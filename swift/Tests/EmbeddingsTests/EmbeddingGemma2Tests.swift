import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation
import Testing
import Tokenizers

@testable import CoreAIEmbeddings

/// Reference vectors from `models/embeddinggemma2/reference/make_text_reference.py`
/// (HF transformers 5.19 + SentenceTransformers, fp32).
struct TextReference: Decodable {
    struct Item: Decodable {
        let prompt_name: String
        let text: String
        let input_ids: [Int]
        let embedding: [Float]
    }
    let max_tokens: Int
    let items: [Item]

    static let url = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "models/embeddinggemma2/reference/fixtures/text_reference.json")

    static func load() throws -> TextReference {
        try JSONDecoder().decode(TextReference.self, from: Data(contentsOf: url))
    }
}

private let modelURL: URL? = ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_MODEL"].map { URL(filePath: $0) }
private let tokenizerFolder: URL? = embeddingGemma2TokenizerFolder()

@Suite struct EmbeddingGemma2UnitTests {
    @Test func matryoshkaKeepsPrefixAndRenormalizes() {
        let v: [Float] = [3, 4, 12, 0]
        let t = EmbeddingGemma2.matryoshka(v, dimension: 2)
        #expect(t.count == 2)
        #expect(abs(t[0] - 0.6) < 1e-6 && abs(t[1] - 0.8) < 1e-6)
    }

    @Test func truncationKeepsEOSLast() {
        let ids = [2] + Array(10..<20) + [1]
        let t = EmbeddingGemma2.truncate(tokenIDs: ids, maxTokens: 5)
        #expect(t == [2, 10, 11, 12, 1])
        #expect(EmbeddingGemma2.truncate(tokenIDs: [2, 5, 1], maxTokens: 5) == [2, 5, 1])
    }

    @Test func prefixesMatchModelCard() {
        #expect(EmbeddingTask.searchQuery.prefix == "task: search result | query: ")
        #expect(EmbeddingTask.document.prefix == "title: none | text: ")
        #expect(EmbeddingTask.documentText("body", title: "T") == "title: T | text: body")
        #expect(EmbeddingTask(promptName: "CodeRetrieval") == .codeRetrieval)
    }
}

@Suite(.enabled(if: tokenizerFolder != nil, "needs the google/embeddinggemma-2 HF cache"))
struct EmbeddingGemma2TokenizerTests {
    @Test func tokenIDsMatchSentenceTransformers() async throws {
        let ref = try TextReference.load()
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder!)
        for item in ref.items {
            let task = try #require(EmbeddingTask(promptName: item.prompt_name))
            let ids = EmbeddingGemma2.truncate(
                tokenIDs: tokenizer.encode(text: task.prefix + item.text), maxTokens: ref.max_tokens)
            #expect(ids == item.input_ids, "token ids differ for \(item.text.prefix(40))")
        }
    }
}

@Suite(.enabled(if: modelURL != nil && tokenizerFolder != nil, "set EMBEDDINGGEMMA2_MODEL to an exported .aimodel"))
struct EmbeddingGemma2ModelTests {
    static func embedder() async throws -> EmbeddingGemma2 {
        try await EmbeddingGemma2(modelURL: modelURL!, tokenizerFolder: tokenizerFolder!)
    }

    /// Every reference text through the full Swift path (prefix, tokenizer, buckets, both functions).
    @Test func parityWithFP32Reference() async throws {
        let ref = try TextReference.load()
        let embedder = try await Self.embedder()
        let maxTokens = await embedder.maxTokens
        for item in ref.items where item.input_ids.count <= maxTokens {
            let task = try #require(EmbeddingTask(promptName: item.prompt_name))
            // The reference truncated at its own max_tokens (2048); truncate the same way.
            let ids = EmbeddingGemma2.truncate(tokenIDs: await embedder.tokenize(item.text, task: task), maxTokens: ref.max_tokens)
            let v = try await embedder.embed(tokenIDs: ids)
            let cos = EmbeddingGemma2.cosine(v, item.embedding)
            #expect(cos >= 0.999, "cos \(cos) for \(item.text.prefix(40))")
        }
    }

    /// MRL: each reduced dimension matches the reference truncated and re-normalized the same way.
    @Test(arguments: [512, 256, 128])
    func reducedDimensions(dimension: Int) async throws {
        let ref = try TextReference.load()
        let embedder = try await Self.embedder()
        for item in ref.items.prefix(6) {
            let v = try await embedder.embed(tokenIDs: item.input_ids, dimension: dimension)
            let r = EmbeddingGemma2.matryoshka(item.embedding, dimension: dimension)
            #expect(v.count == dimension)
            #expect(EmbeddingGemma2.cosine(v, r) >= 0.999)
        }
    }

    /// text <-> text retrieval: the query ranks its answer above an unrelated document.
    @Test func queryRanksMatchingDocumentFirst() async throws {
        let embedder = try await Self.embedder()
        let q = try await embedder.embed("What causes the northern lights?", task: .searchQuery)
        let hit = try await embedder.embedDocument("The northern lights are caused by charged particles from the sun.")
        let miss = try await embedder.embedDocument("Photosynthesis converts light energy into chemical energy in plants.")
        #expect(EmbeddingGemma2.cosine(q, hit) > EmbeddingGemma2.cosine(q, miss) + 0.1)
    }
}

// MARK: - Images

struct ImageReference: Decodable {
    struct Item: Decodable {
        let stem: String
        let caption: String
        let num_soft_tokens: Int
        let embedding: [Float]
        let caption_embedding: [Float]
    }
    let items: [Item]

    static let dir = TextReference.url.deletingLastPathComponent()

    static func load() throws -> ImageReference {
        try JSONDecoder().decode(ImageReference.self, from: Data(contentsOf: dir.appending(path: "image_reference.json")))
    }

    static func image(_ name: String) throws -> CGImage {
        try CGImageUtils.load(from: dir.appending(path: "images/\(name).png"))
    }
}

@Suite struct GemmaImageProcessorTests {
    /// Swift resize vs HF's (torchvision antialiased bicubic) resized pixels, for up- and downscaling.
    @Test(arguments: ["red_circle", "green_triangle", "yellow_star_tall", "sunset_gradient", "large_pattern"])
    func resizeMatchesHF(stem: String) throws {
        let files = try FileManager.default.contentsOfDirectory(atPath: ImageReference.dir.appending(path: "images").path)
        let hfName = try #require(files.first { $0.hasPrefix("\(stem)__hf_") }).replacingOccurrences(of: ".png", with: "")
        let (src, w, h) = try GemmaImageProcessor.rgbPixels(ImageReference.image(stem))
        let (hf, tw, th) = try GemmaImageProcessor.rgbPixels(ImageReference.image(hfName))
        let target = GemmaImageProcessor.targetSize(height: h, width: w)
        #expect(target.height == th && target.width == tw)
        let mine = GemmaImageProcessor.resize(src, width: w, height: h, outWidth: tw, outHeight: th)
        var maxDiff = 0, off = 0
        for i in 0..<mine.count {
            let d = abs(Int(mine[i]) - Int(hf[i]))
            maxDiff = max(maxDiff, d)
            if d > 0 { off += 1 }
        }
        #expect(maxDiff <= 2, "max pixel difference \(maxDiff)")
        #expect(Double(off) / Double(mine.count) < 0.005, "\(off) of \(mine.count) values differ")
    }

    @Test func blockOrderAndCounts() throws {
        let inputs = try GemmaImageProcessor.inputs(for: ImageReference.image("green_triangle"))
        #expect(inputs.softTokenCount == 260)  // HF: 260 soft tokens for 384x256
        // First block: patches (0,0) (1,0) (2,0) (0,1) ... in x, y.
        #expect(Array(inputs.xIDs.prefix(9)) == [0, 1, 2, 0, 1, 2, 0, 1, 2])
        #expect(Array(inputs.yIDs.prefix(9)) == [0, 0, 0, 1, 1, 1, 2, 2, 2])
        #expect(inputs.patchMask.filter { $0 == 1 }.count == 260 * 9)
    }
}

@Suite(.enabled(if: modelURL != nil && tokenizerFolder != nil, "set EMBEDDINGGEMMA2_MODEL to an exported .aimodel"))
struct EmbeddingGemma2ImageTests {
    /// Full Swift image path (decode, resize, patches, RoPE, both functions, text encoder) vs fp32 HF.
    @Test func imageParityWithFP32Reference() async throws {
        let ref = try ImageReference.load()
        let embedder = try await EmbeddingGemma2ModelTests.embedder()
        try #require(await embedder.supportsImages)
        for item in ref.items {
            let v = try await embedder.embed(image: ImageReference.image(item.stem))
            let cos = EmbeddingGemma2.cosine(v, item.embedding)
            #expect(cos >= 0.999, "cos \(cos) for \(item.stem)")
        }
    }

    /// text -> image: each caption ranks its own image first (the fp32 reference gets 7 of 7).
    @Test func captionFindsItsImage() async throws {
        let ref = try ImageReference.load()
        let embedder = try await EmbeddingGemma2ModelTests.embedder()
        var images: [[Float]] = []
        for item in ref.items { images.append(try await embedder.embed(image: ImageReference.image(item.stem))) }
        for (i, item) in ref.items.enumerated() {
            let q = try await embedder.embed(item.caption)
            let best = images.indices.max { EmbeddingGemma2.cosine(q, images[$0]) < EmbeddingGemma2.cosine(q, images[$1]) }
            #expect(best == i, "'\(item.caption)' matched \(ref.items[best!].stem)")
        }
    }

    /// image <-> image: a half-size copy of an image is nearest to its original, at 768 and 256 dims.
    @Test(arguments: [768, 256])
    func resizedCopyFindsOriginal(dimension: Int) async throws {
        let ref = try ImageReference.load()
        let embedder = try await EmbeddingGemma2ModelTests.embedder()
        let original = try ImageReference.image("stop_sign")
        let ctx = try #require(CGContext(
            data: nil, width: original.width / 2, height: original.height / 2, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.interpolationQuality = .high
        ctx.draw(original, in: CGRect(x: 0, y: 0, width: original.width / 2, height: original.height / 2))
        let copy = try await embedder.embed(image: try #require(ctx.makeImage()), dimension: dimension)
        var scores: [(String, Float)] = []
        for item in ref.items {
            let v = try await embedder.embed(image: ImageReference.image(item.stem), dimension: dimension)
            scores.append((item.stem, EmbeddingGemma2.cosine(copy, v)))
        }
        #expect(scores.max { $0.1 < $1.1 }?.0 == "stop_sign", "\(scores)")
    }
}

// MARK: - Video

struct VideoReference: Decodable {
    struct Item: Decodable {
        let stem: String
        let caption: String
        let fps: Double
        let total_frames: Int
        let sampled_indices: [Int]
        let embedding: [Float]
    }
    let items: [Item]

    static let dir = TextReference.url.deletingLastPathComponent()

    static func load() throws -> VideoReference {
        try JSONDecoder().decode(VideoReference.self, from: Data(contentsOf: dir.appending(path: "video_reference.json")))
    }

    static func url(_ stem: String) -> URL { dir.appending(path: "videos/\(stem).mp4") }

    /// The frames HF sampled, decoded by PyAV.
    static func hfFrames(_ stem: String) throws -> [CGImage] {
        let folder = dir.appending(path: "videos/\(stem)")
        let names = try FileManager.default.contentsOfDirectory(atPath: folder.path).filter { $0.hasSuffix(".png") }.sorted()
        return try names.map { try CGImageUtils.load(from: folder.appending(path: $0)) }
    }
}

@Suite struct GemmaVideoSamplerTests {
    @Test func indicesMatchHF() throws {
        for item in try VideoReference.load().items {
            let idx = GemmaVideoSampler.sampleIndices(totalFrames: item.total_frames, fps: item.fps)
            #expect(idx == item.sampled_indices, "\(item.stem)")
        }
    }

    /// AVFoundation decode vs PyAV decode of the same H.264 frames (YUV -> RGB conversions differ slightly).
    @Test func decodedFramesCloseToHF() async throws {
        for item in try VideoReference.load().items.prefix(3) {
            let mine = try await GemmaVideoSampler.frames(from: VideoReference.url(item.stem))
            let hf = try VideoReference.hfFrames(item.stem)
            #expect(mine.count == hf.count, "\(item.stem): \(mine.count) vs \(hf.count) frames")
            var total = 0.0, count = 0
            for (a, b) in zip(mine, hf) {
                let pa = try GemmaImageProcessor.rgbPixels(a).pixels, pb = try GemmaImageProcessor.rgbPixels(b).pixels
                for i in 0..<min(pa.count, pb.count) { total += Double(abs(Int(pa[i]) - Int(pb[i]))) }
                count += pa.count
            }
            #expect(total / Double(count) < 3, "\(item.stem): mean |diff| \(total / Double(count))")
        }
    }
}

@Suite(.enabled(if: modelURL != nil && tokenizerFolder != nil, "set EMBEDDINGGEMMA2_MODEL to an exported .aimodel"))
struct EmbeddingGemma2VideoTests {
    /// Model parity from HF's own frames (isolates the model from video decoding).
    @Test func videoParityFromHFFrames() async throws {
        let embedder = try await EmbeddingGemma2ModelTests.embedder()
        try #require(await embedder.supportsVideo)
        for item in try VideoReference.load().items {
            let v = try await embedder.embed(videoFrames: VideoReference.hfFrames(item.stem))
            let cos = EmbeddingGemma2.cosine(v, item.embedding)
            #expect(cos >= 0.999, "cos \(cos) for \(item.stem)")
        }
    }

    /// Full path from the .mp4 (AVFoundation decode + sampling) vs the fp32 reference.
    @Test func videoParityFromFile() async throws {
        let embedder = try await EmbeddingGemma2ModelTests.embedder()
        for item in try VideoReference.load().items {
            let v = try await embedder.embed(videoAt: VideoReference.url(item.stem))
            let cos = EmbeddingGemma2.cosine(v, item.embedding)
            #expect(cos >= 0.995, "cos \(cos) for \(item.stem)")
        }
    }

    /// text -> video: each caption ranks its own clip first.
    @Test func captionFindsItsVideo() async throws {
        let ref = try VideoReference.load()
        let embedder = try await EmbeddingGemma2ModelTests.embedder()
        var videos: [[Float]] = []
        for item in ref.items { videos.append(try await embedder.embed(videoFrames: VideoReference.hfFrames(item.stem))) }
        for (i, item) in ref.items.enumerated() {
            let q = try await embedder.embed(item.caption)
            let best = videos.indices.max { EmbeddingGemma2.cosine(q, videos[$0]) < EmbeddingGemma2.cosine(q, videos[$1]) }
            #expect(best == i, "'\(item.caption)' matched \(ref.items[best!].stem)")
        }
    }

    /// image <-> video: a still image finds the clip with the same object.
    @Test func imageFindsMatchingVideo() async throws {
        let ref = try VideoReference.load()
        let embedder = try await EmbeddingGemma2ModelTests.embedder()
        var videos: [String: [Float]] = [:]
        for item in ref.items { videos[item.stem] = try await embedder.embed(videoFrames: VideoReference.hfFrames(item.stem)) }
        let pairs = [("red_circle", "red_ball_moving"), ("blue_square", "blue_square_growing"),
                     ("green_triangle", "green_triangle_rotating"), ("yellow_star_tall", "yellow_star_falling"),
                     ("checkerboard", "checkerboard_flashing")]
        for (image, video) in pairs {
            let v = try await embedder.embed(image: ImageReference.image(image))
            let best = videos.max { EmbeddingGemma2.cosine(v, $0.value) < EmbeddingGemma2.cosine(v, $1.value) }!.key
            #expect(best == video, "\(image) matched \(best)")
        }
    }
}
