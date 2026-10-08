import CoreAI
import CoreAIShared
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
    static let candidateBuckets = [128, 512, 2048, 8192]
    static let bosID = 2
    static let eosID = 1
    static let padID = 0

    private struct Bucket {
        let length: Int
        let tokens: InferenceFunction
        let encoder: InferenceFunction
        let idsDescriptor: NDArrayDescriptor
        let maskDescriptor: NDArrayDescriptor
    }

    private let tokenizer: any Tokenizer
    private let buckets: [Bucket]

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

        var found: [Bucket] = []
        for length in Self.candidateBuckets {
            let tokName = "embed_tokens_\(length)"
            let encName = "embed_text_\(length)"
            guard let tokFn = try model.loadFunction(named: tokName),
                let encFn = try model.loadFunction(named: encName),
                case .ndArray(let idsDesc) = model.functionDescriptor(for: tokName)?.inputDescriptor(of: "input_ids"),
                case .ndArray(let maskDesc) = model.functionDescriptor(for: encName)?.inputDescriptor(of: "attention_mask")
            else { continue }
            found.append(
                Bucket(length: length, tokens: tokFn, encoder: encFn, idsDescriptor: idsDesc, maskDescriptor: maskDesc))
        }
        guard !found.isEmpty else {
            throw EmbeddingError.missingFunctions(modelURL.lastPathComponent)
        }
        self.buckets = found
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
        guard Self.supportedDimensions.contains(dimension) else {
            throw EmbeddingError.unsupportedDimension(dimension)
        }
        let ids = Self.truncate(tokenIDs: tokenIDs, maxTokens: maxTokens)
        guard let bucket = buckets.first(where: { ids.count <= $0.length }) else {
            throw EmbeddingError.inputTooLong(ids.count, maxTokens)
        }
        let length = bucket.length

        var idsArray = NDArray(descriptor: bucket.idsDescriptor)
        fillNDArray(&idsArray, as: Int32.self, count: length) { $0 < ids.count ? Int32(ids[$0]) : Int32(Self.padID) }
        var tokOut = try await bucket.tokens.run(inputs: ["input_ids": idsArray])
        guard let inputsEmbeds = tokOut.remove("inputs_embeds")?.ndArray else {
            throw EmbeddingError.missingOutput("inputs_embeds")
        }

        var maskArray = NDArray(descriptor: bucket.maskDescriptor)
        fillNDArray(&maskArray, as: Float16.self, count: length) { $0 < ids.count ? 1 : 0 }
        var encOut = try await bucket.encoder.run(inputs: ["inputs_embeds": inputsEmbeds, "attention_mask": maskArray])
        guard let pooled = encOut.remove("embedding")?.ndArray else {
            throw EmbeddingError.missingOutput("embedding")
        }
        return Self.matryoshka(flattenAsFloat(pooled), dimension: dimension)
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

    public var description: String {
        switch self {
        case .missingFunctions(let name): "\(name) has no embed_tokens_{L} / embed_text_{L} function pair"
        case .missingOutput(let name): "the model did not return '\(name)'"
        case .unsupportedDimension(let d): "dimension \(d) is not one of \(EmbeddingGemma2.supportedDimensions)"
        case .inputTooLong(let n, let max): "\(n) tokens exceed the largest bucket (\(max))"
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
