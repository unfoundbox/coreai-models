import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation
import Tokenizers

/// Text embeddings from an EmbeddingGemma 2 `.aimodel` exported by `models/embeddinggemma2/export.py`.
///
/// The asset holds two functions per static sequence bucket `L`:
/// `embed_tokens_{L}` (int32 ids -> fp16 embeddings) and `embed_text_{L}` (embeddings + mask ->
/// unnormalized 768-d mean). This type adds the task prefix, tokenizes, picks the smallest bucket
/// that fits, pads, runs both functions, L2-normalizes in fp32 and applies MRL truncation.
public actor EmbeddingGemma2 {
    public static let fullDimension = 768
    public static let supportedDimensions = [768, 512, 256, 128]
    static let bosID = 2
    static let eosID = 1
    static let padID = 0
    static let imageStartID = 255999  // <|image>
    static let imageTokenID = 258880  // <|image|>
    static let imageEndID = 258882  // <image|>
    static let videoTokenID = 258884  // <|video|>

    private struct Bucket {
        let length: Int
        let tokens: InferenceFunction
        let encoder: InferenceFunction
        let idsDescriptor: NDArrayDescriptor
        let maskDescriptor: NDArrayDescriptor
        let embedsDescriptor: NDArrayDescriptor
    }

    private struct Vision {
        let positions: InferenceFunction
        let encoder: InferenceFunction
        let descriptors: [String: NDArrayDescriptor]
    }

    private let tokenizer: any Tokenizer
    private let buckets: [Bucket]
    /// Vision functions per patch budget (2520 = images, 1260 = video frames).
    private let visions: [Int: Vision]

    /// True when the asset has the image functions (`image_positions_2520`, `embed_image_2520`).
    public var supportsImages: Bool { visions[GemmaImageProcessor.maxPatches] != nil }
    /// True when the asset has the video-frame functions (`*_1260`).
    public var supportsVideo: Bool { visions[GemmaImageProcessor.videoFramePatches] != nil }

    /// Longest input in tokens, special tokens included. Longer inputs are truncated.
    public var maxTokens: Int { buckets.last?.length ?? 0 }
    public var bucketLengths: [Int] { buckets.map(\.length) }

    /// - Parameters:
    ///   - modelURL: the `.aimodel` (or AOT-compiled `.aimodelc`).
    ///   - tokenizerFolder: folder with `tokenizer.json` (the HF snapshot of `google/embeddinggemma-2`).
    ///   - computeUnit: preferred compute unit. `.neuralEngine` by default; `nil` uses Core AI's default.
    public init(
        modelURL: URL, tokenizerFolder: URL, computeUnit: ComputeUnitKind? = .neuralEngine
    ) async throws {
        let options = computeUnit.map { SpecializationOptions(preferredComputeUnitKind: $0) } ?? .default
        let model = try await AIModel(contentsOf: modelURL, options: options)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)

        // Buckets come from the asset's function names (embed_text_{L}).
        let lengths = model.functionNames.compactMap { name -> Int? in
            name.hasPrefix("embed_text_") ? Int(name.dropFirst("embed_text_".count)) : nil
        }.sorted()
        var found: [Bucket] = []
        for length in lengths {
            let tokName = "embed_tokens_\(length)"
            let encName = "embed_text_\(length)"
            guard let tokFn = try model.loadFunction(named: tokName),
                let encFn = try model.loadFunction(named: encName),
                case .ndArray(let idsDesc) = model.functionDescriptor(for: tokName)?.inputDescriptor(of: "input_ids"),
                case .ndArray(let maskDesc) = model.functionDescriptor(for: encName)?.inputDescriptor(of: "attention_mask"),
                case .ndArray(let embDesc) = model.functionDescriptor(for: encName)?.inputDescriptor(of: "inputs_embeds")
            else { continue }
            found.append(
                Bucket(
                    length: length, tokens: tokFn, encoder: encFn, idsDescriptor: idsDesc, maskDescriptor: maskDesc,
                    embedsDescriptor: embDesc))
        }
        guard !found.isEmpty else {
            throw EmbeddingError.missingFunctions(modelURL.lastPathComponent)
        }
        self.buckets = found

        var visions: [Int: Vision] = [:]
        for budget in [GemmaImageProcessor.maxPatches, GemmaImageProcessor.videoFramePatches] {
            let posName = "image_positions_\(budget)"
            let imgName = "embed_image_\(budget)"
            guard let posFn = try model.loadFunction(named: posName), let imgFn = try model.loadFunction(named: imgName),
                let posDesc = model.functionDescriptor(for: posName), let imgDesc = model.functionDescriptor(for: imgName)
            else { continue }
            var descriptors: [String: NDArrayDescriptor] = [:]
            for name in ["x_ids", "y_ids"] {
                if case .ndArray(let d) = posDesc.inputDescriptor(of: name) { descriptors[name] = d }
            }
            for name in ["pixel_patches", "rope_cos", "rope_sin", "patch_mask"] {
                if case .ndArray(let d) = imgDesc.inputDescriptor(of: name) { descriptors[name] = d }
            }
            if descriptors.count == 6 {
                visions[budget] = Vision(positions: posFn, encoder: imgFn, descriptors: descriptors)
            }
        }
        self.visions = visions
    }

    // MARK: Tokenization

    /// Token ids exactly as SentenceTransformers builds them: `<bos> prefix+text <eos>`,
    /// truncated to `maxTokens` with `<eos>` kept last.
    public func tokenize(_ text: String, task: EmbeddingTask = .none) -> [Int] {
        Self.truncate(tokenIDs: tokenizer.encode(text: task.prefix + text), maxTokens: maxTokens)
    }

    static func truncate(tokenIDs ids: [Int], maxTokens: Int) -> [Int] {
        guard ids.count > maxTokens else { return ids }
        return Array(ids.prefix(maxTokens - 1)) + [eosID]
    }

    // MARK: Embedding

    /// L2-normalized embedding of `text`, truncated to `dimension` (768, 512, 256 or 128).
    public func embed(_ text: String, task: EmbeddingTask = .none, dimension: Int = fullDimension)
        async throws -> [Float]
    {
        try await embed(tokenIDs: tokenize(text, task: task), dimension: dimension)
    }

    /// Embeds a document: `title: {title or "none"} | text: {text}`.
    public func embedDocument(_ text: String, title: String? = nil, dimension: Int = fullDimension)
        async throws -> [Float]
    {
        try await embed(EmbeddingTask.documentText(text, title: title), task: .none, dimension: dimension)
    }

    /// L2-normalized embedding of pre-tokenized ids (special tokens included).
    public func embed(tokenIDs: [Int], dimension: Int = fullDimension) async throws -> [Float] {
        try await embed(tokenIDs: tokenIDs, dimension: dimension, editEmbeddings: nil)
    }

    /// Core text path. `editEmbeddings` may overwrite columns of the channel-major (512, L)
    /// `inputs_embeds` before the encoder runs (used to place image soft tokens).
    private func embed(
        tokenIDs: [Int], dimension: Int,
        editEmbeddings: ((inout [Float16], Int) -> Void)?
    ) async throws -> [Float] {
        guard Self.supportedDimensions.contains(dimension) else {
            throw EmbeddingError.unsupportedDimension(dimension)
        }
        let ids = Self.truncate(tokenIDs: tokenIDs, maxTokens: maxTokens)
        guard let bucket = buckets.first(where: { ids.count <= $0.length }) else {
            throw EmbeddingError.inputTooLong(ids.count, maxTokens)
        }
        let length = bucket.length

        var idsArray = NDArray(descriptor: bucket.idsDescriptor)
        fillNDArray(&idsArray, as: Int32.self, count: length) {
            guard $0 < ids.count else { return Int32(Self.padID) }
            let id = ids[$0]
            return id == Self.imageTokenID || id == Self.videoTokenID ? Int32(Self.padID) : Int32(id)  // HF: slots use pad
        }
        var tokOut = try await bucket.tokens.run(inputs: ["input_ids": idsArray])
        guard var inputsEmbeds = tokOut.remove("inputs_embeds")?.ndArray else {
            throw EmbeddingError.missingOutput("inputs_embeds")
        }
        if let editEmbeddings {
            var values = readNDArray(inputsEmbeds, as: Float16.self, count: inputsEmbeds.shape.reduce(1, *))
            editEmbeddings(&values, length)
            inputsEmbeds = NDArray(descriptor: bucket.embedsDescriptor)
            fillNDArray(&inputsEmbeds, as: Float16.self, count: values.count) { values[$0] }
        }

        var maskArray = NDArray(descriptor: bucket.maskDescriptor)
        fillNDArray(&maskArray, as: Float16.self, count: length) { $0 < ids.count ? 1 : 0 }
        var encOut = try await bucket.encoder.run(inputs: ["inputs_embeds": inputsEmbeds, "attention_mask": maskArray])
        guard let pooled = encOut.remove("embedding")?.ndArray else {
            throw EmbeddingError.missingOutput("embedding")
        }
        return Self.matryoshka(flattenAsFloat(pooled), dimension: dimension)
    }

    // MARK: Images

    /// L2-normalized embedding of an image (no text prefix), truncated to `dimension`.
    public func embed(image: CGImage, dimension: Int = fullDimension) async throws -> [Float] {
        try await embed(imageInputs: GemmaImageProcessor.inputs(for: image), dimension: dimension)
    }

    /// Same as `embed(image:)`, from a file (PNG, JPEG, HEIC, ...).
    public func embed(imageAt url: URL, dimension: Int = fullDimension) async throws -> [Float] {
        try await embed(image: CGImageUtils.load(from: url), dimension: dimension)
    }

    /// Image embedding from already-preprocessed inputs.
    public func embed(imageInputs inputs: GemmaImageInputs, dimension: Int = fullDimension) async throws -> [Float] {
        guard supportsImages else { throw EmbeddingError.imagesUnsupported }
        let soft = try await softTokens(inputs)
        let n = inputs.softTokenCount
        let ids = [Self.bosID, Self.imageStartID] + Array(repeating: Self.imageTokenID, count: n) + [Self.imageEndID, Self.eosID]
        return try await embed(tokenIDs: ids, dimension: dimension, slotTokens: [soft])
    }

    // MARK: Video

    /// L2-normalized embedding of a video file: 1 frame per second, at most 32 frames (HF defaults).
    public func embed(videoAt url: URL, dimension: Int = fullDimension) async throws -> [Float] {
        try await embed(videoFrames: GemmaVideoSampler.frames(from: url), dimension: dimension)
    }

    /// L2-normalized embedding of already-sampled frames, in order.
    public func embed(videoFrames frames: [CGImage], dimension: Int = fullDimension) async throws -> [Float] {
        guard supportsVideo else { throw EmbeddingError.videoUnsupported }
        guard !frames.isEmpty else { throw EmbeddingError.noFrames }
        var perFrame: [[Float16]] = []
        var ids = [Self.bosID]
        for frame in frames {
            let inputs = try GemmaImageProcessor.inputs(for: frame, budget: GemmaImageProcessor.videoFramePatches)
            perFrame.append(try await softTokens(inputs))
            ids += [Self.imageStartID] + Array(repeating: Self.videoTokenID, count: inputs.softTokenCount) + [Self.imageEndID]
        }
        ids.append(Self.eosID)
        return try await embed(tokenIDs: ids, dimension: dimension, slotTokens: perFrame)
    }

    /// Runs the vision functions for one image or frame. Returns channel-major (512, N) soft tokens.
    private func softTokens(_ inputs: GemmaImageInputs) async throws -> [Float16] {
        guard let vision = visions[inputs.budget] else { throw EmbeddingError.imagesUnsupported }
        let d = vision.descriptors
        func array<T: BitwiseCopyable>(_ name: String, _ values: [T]) -> NDArray {
            var a = NDArray(descriptor: d[name]!)
            fillNDArray(&a, as: T.self, count: values.count) { values[$0] }
            return a
        }
        var posOut = try await vision.positions.run(inputs: ["x_ids": array("x_ids", inputs.xIDs), "y_ids": array("y_ids", inputs.yIDs)])
        guard let posEmbeds = posOut.remove("pos_embeds")?.ndArray else { throw EmbeddingError.missingOutput("pos_embeds") }
        var imgOut = try await vision.encoder.run(inputs: [
            "pixel_patches": array("pixel_patches", inputs.pixelPatches),
            "pos_embeds": posEmbeds,
            "rope_cos": array("rope_cos", inputs.ropeCos),
            "rope_sin": array("rope_sin", inputs.ropeSin),
            "patch_mask": array("patch_mask", inputs.patchMask),
        ])
        guard let softArray = imgOut.remove("soft_tokens")?.ndArray else { throw EmbeddingError.missingOutput("soft_tokens") }
        let slots = softArray.shape[3], channels = softArray.shape[1]
        let all = readNDArray(softArray, as: Float16.self, count: softArray.shape.reduce(1, *))
        let n = inputs.softTokenCount
        var out = [Float16](repeating: 0, count: channels * n)
        for c in 0..<channels {
            for j in 0..<n { out[c * n + j] = all[c * slots + j] }
        }
        return out
    }

    /// Text path with soft tokens placed into the image/video slot columns, in order.
    /// `slotTokens`: one channel-major (512, N_i) block per image or frame.
    private func embed(tokenIDs ids: [Int], dimension: Int, slotTokens: [[Float16]]) async throws -> [Float] {
        // Never truncate multimodal input: cutting through the slots would misplace soft tokens.
        guard ids.count <= maxTokens else { throw EmbeddingError.inputTooLong(ids.count, maxTokens) }
        let columns = ids.indices.filter { ids[$0] == Self.imageTokenID || ids[$0] == Self.videoTokenID }
        return try await embed(tokenIDs: ids, dimension: dimension) { embeds, length in
            let channels = embeds.count / length
            var col = 0
            for block in slotTokens {
                let n = block.count / channels
                for j in 0..<n {
                    for c in 0..<channels { embeds[c * length + columns[col]] = block[c * n + j] }
                    col += 1
                }
            }
        }
    }

    // MARK: Vector math

    /// Keep the leading `dimension` values, then L2-normalize (MRL truncation).
    public static func matryoshka(_ vector: [Float], dimension: Int) -> [Float] {
        normalized(Array(vector.prefix(dimension)))
    }

    public static func normalized(_ vector: [Float]) -> [Float] {
        var sum: Double = 0
        for v in vector { sum += Double(v) * Double(v) }
        let inv = sum > 0 ? Float(1 / sum.squareRoot()) : 0
        return vector.map { $0 * inv }
    }

    /// Cosine similarity of two L2-normalized vectors of the same dimension.
    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "cosine: dimensions differ (\(a.count) vs \(b.count))")
        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }
        return dot
    }
}

/// Task instruction prefixes from the model card (`config_sentence_transformers.json`).
public enum EmbeddingTask: String, CaseIterable, Sendable {
    case none
    case searchQuery
    case document
    case questionAnswering
    case factChecking
    case codeRetrieval
    case classification
    case clustering
    case sentenceSimilarity

    public var prefix: String {
        switch self {
        case .none: ""
        case .searchQuery: "task: search result | query: "
        case .document: "title: none | text: "
        case .questionAnswering: "task: question answering | query: "
        case .factChecking: "task: fact checking | query: "
        case .codeRetrieval: "task: code retrieval | query: "
        case .classification: "task: classification | query: "
        case .clustering: "task: clustering | query: "
        case .sentenceSimilarity: "task: sentence similarity | query: "
        }
    }

    /// Document formatting for asymmetric retrieval: `title: {title} | text: {text}`.
    public static func documentText(_ text: String, title: String?) -> String {
        "title: \(title ?? "none") | text: \(text)"
    }

    /// SentenceTransformers prompt name (as stored in the reference fixtures) -> task.
    public init?(promptName: String) {
        switch promptName {
        case "": self = .none
        case "SearchQuery", "Retrieval-query", "query": self = .searchQuery
        case "Document", "Retrieval-document", "document": self = .document
        case "QuestionAnswering": self = .questionAnswering
        case "FactChecking": self = .factChecking
        case "CodeRetrieval": self = .codeRetrieval
        case "Classification": self = .classification
        case "Clustering": self = .clustering
        case "SentenceSimilarity", "STS": self = .sentenceSimilarity
        default: return nil
        }
    }
}

public enum EmbeddingError: Error, CustomStringConvertible {
    case missingFunctions(String)
    case missingOutput(String)
    case unsupportedDimension(Int)
    case inputTooLong(Int, Int)
    case imagesUnsupported
    case videoUnsupported
    case noFrames
    case imageDecodingFailed

    public var description: String {
        switch self {
        case .missingFunctions(let name): "\(name) has no embed_tokens_{L} / embed_text_{L} function pair"
        case .missingOutput(let name): "the model did not return '\(name)'"
        case .unsupportedDimension(let d): "dimension \(d) is not one of \(EmbeddingGemma2.supportedDimensions)"
        case .inputTooLong(let n, let max): "\(n) tokens exceed the largest bucket (\(max))"
        case .imagesUnsupported: "this .aimodel has no image functions (export with --modalities text,image)"
        case .imageDecodingFailed: "could not read the image pixels"
        case .videoUnsupported: "this .aimodel has no video-frame functions (export with --image-budgets 2520,1260)"
        case .noFrames: "the video produced no frames"
        }
    }
}

#if os(macOS)
/// Snapshot folder of `google/embeddinggemma-2` in the local Hugging Face cache, if present.
public func embeddingGemma2TokenizerFolder(
    hubCache: URL = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub")
) -> URL? {
    let repo = hubCache.appending(path: "models--google--embeddinggemma-2")
    guard let ref = try? String(contentsOf: repo.appending(path: "refs/main"), encoding: .utf8) else {
        return nil
    }
    let snapshot = repo.appending(path: "snapshots/\(ref.trimmingCharacters(in: .whitespacesAndNewlines))")
    return FileManager.default.fileExists(atPath: snapshot.appending(path: "tokenizer.json").path) ? snapshot : nil
}
#endif
