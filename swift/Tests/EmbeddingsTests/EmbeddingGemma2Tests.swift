import CoreAI
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
            let v = try await embedder.embed(item.text, task: task)
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
