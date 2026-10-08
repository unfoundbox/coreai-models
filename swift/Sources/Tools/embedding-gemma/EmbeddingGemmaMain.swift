import ArgumentParser
import CoreAI
import CoreAIEmbeddings
import Foundation

@main
struct EmbeddingGemmaCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embedding-gemma",
        abstract: "Text embeddings and semantic search with an EmbeddingGemma 2 .aimodel.",
        subcommands: [Embed.self, Search.self, MediaSearch.self]
    )
}

struct ModelOptions: ParsableArguments {
    @Option(name: .long, help: "Path to the EmbeddingGemma 2 .aimodel.")
    var model: String

    @Option(name: .long, help: "Folder with tokenizer.json. Default: the google/embeddinggemma-2 HF cache snapshot.")
    var tokenizer: String?

    @Option(name: .long, help: "Preferred compute unit: neural-engine, gpu, cpu or default.")
    var compute: String = "neural-engine"

    @Option(name: .long, help: "Output dimension: 768, 512, 256 or 128.")
    var dim: Int = 768

    func load() async throws -> EmbeddingGemma2 {
        guard let folder = tokenizer.map({ URL(filePath: $0) }) ?? embeddingGemma2TokenizerFolder() else {
            throw ValidationError("No tokenizer found. Pass --tokenizer or download google/embeddinggemma-2.")
        }
        let unit: ComputeUnitKind?
        switch compute {
        case "neural-engine": unit = .neuralEngine
        case "gpu": unit = .gpu
        case "cpu": unit = .cpu
        case "default": unit = nil
        default: throw ValidationError("--compute must be neural-engine, gpu, cpu or default")
        }
        let start = ContinuousClock.now
        let embedder = try await EmbeddingGemma2(modelURL: URL(filePath: model), tokenizerFolder: folder, computeUnit: unit)
        let buckets = await embedder.bucketLengths
        FileHandle.standardError.write(
            Data("loaded in \(ContinuousClock.now - start); buckets \(buckets); compute \(compute)\n".utf8))
        return embedder
    }
}

struct Embed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Print the embedding of each text as JSON lines.")

    @OptionGroup var options: ModelOptions

    @Option(name: .long, help: "Task prefix: \(EmbeddingTask.allCases.map(\.rawValue).joined(separator: ", ")).")
    var task: String = "none"

    @Option(name: .long, help: "Image file to embed (repeatable).")
    var image: [String] = []

    @Option(name: .long, help: "Video file to embed (repeatable): 1 frame per second, at most 32 frames.")
    var video: [String] = []

    @Argument(help: "Texts to embed.")
    var texts: [String] = []

    func run() async throws {
        guard let task = EmbeddingTask(rawValue: task) else { throw ValidationError("unknown --task \(task)") }
        let embedder = try await options.load()
        for path in image {
            let start = ContinuousClock.now
            let vector = try await embedder.embed(imageAt: URL(filePath: path), dimension: options.dim)
            let elapsed = ContinuousClock.now - start
            let values = vector.map { String(format: "%.6f", $0) }.joined(separator: ",")
            print("{\"image\":\(jsonString(path)),\"dim\":\(vector.count),\"ms\":\(milliseconds(elapsed)),\"embedding\":[\(values)]}")
        }
        for path in video {
            let start = ContinuousClock.now
            let vector = try await embedder.embed(videoAt: URL(filePath: path), dimension: options.dim)
            let elapsed = ContinuousClock.now - start
            let values = vector.map { String(format: "%.6f", $0) }.joined(separator: ",")
            print("{\"video\":\(jsonString(path)),\"dim\":\(vector.count),\"ms\":\(milliseconds(elapsed)),\"embedding\":[\(values)]}")
        }
        for text in texts {
            let start = ContinuousClock.now
            let vector = try await embedder.embed(text, task: task, dimension: options.dim)
            let elapsed = ContinuousClock.now - start
            let values = vector.map { String(format: "%.6f", $0) }.joined(separator: ",")
            print("{\"text\":\(jsonString(text)),\"dim\":\(vector.count),\"ms\":\(milliseconds(elapsed)),\"embedding\":[\(values)]}")
        }
    }
}

struct Search: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Rank the lines of a text file against a query (query uses the search prefix, lines the document format).")

    @OptionGroup var options: ModelOptions

    @Option(name: .long, help: "Query text.")
    var query: String

    @Option(name: .long, help: "Text file; each non-empty line is one document.")
    var docs: String

    @Option(name: .long, help: "Number of results.")
    var top: Int = 5

    func run() async throws {
        let lines = try String(contentsOf: URL(filePath: docs), encoding: .utf8)
            .split(separator: "\n").map(String.init).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let embedder = try await options.load()
        let start = ContinuousClock.now
        var docVectors: [[Float]] = []
        for line in lines {
            docVectors.append(try await embedder.embedDocument(line, dimension: options.dim))
        }
        let q = try await embedder.embed(query, task: .searchQuery, dimension: options.dim)
        let elapsed = ContinuousClock.now - start
        let ranked = docVectors.enumerated()
            .map { (index: $0.offset, score: EmbeddingGemma2.cosine(q, $0.element)) }
            .sorted { $0.score > $1.score }
        print("query: \(query)   (\(lines.count) docs, dim \(options.dim), \(milliseconds(elapsed)) ms total)")
        for (rank, hit) in ranked.prefix(top).enumerated() {
            print(String(format: "%2d. %.4f  ", rank + 1, hit.score) + lines[hit.index].prefix(100))
        }
    }
}

struct MediaSearch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "media-search", abstract: "Rank the images and videos in a folder against a text query.")

    @OptionGroup var options: ModelOptions

    @Option(name: .long, help: "Query text.")
    var query: String

    @Option(name: .long, help: "Folder of images (png, jpg, jpeg, heic) and videos (mp4, mov, m4v).")
    var folder: String

    @Option(name: .long, help: "Number of results.")
    var top: Int = 5

    func run() async throws {
        let imageTypes: Set = ["png", "jpg", "jpeg", "heic"], videoTypes: Set = ["mp4", "mov", "m4v"]
        let files = try FileManager.default.contentsOfDirectory(at: URL(filePath: folder), includingPropertiesForKeys: nil)
            .filter { imageTypes.union(videoTypes).contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let embedder = try await options.load()
        let start = ContinuousClock.now
        var vectors: [[Float]] = []
        for file in files {
            vectors.append(videoTypes.contains(file.pathExtension.lowercased())
                ? try await embedder.embed(videoAt: file, dimension: options.dim)
                : try await embedder.embed(imageAt: file, dimension: options.dim))
        }
        let q = try await embedder.embed(query, task: .none, dimension: options.dim)
        let elapsed = ContinuousClock.now - start
        let ranked = vectors.enumerated().map { ($0.offset, EmbeddingGemma2.cosine(q, $0.element)) }.sorted { $0.1 > $1.1 }
        print("query: \(query)   (\(files.count) files, dim \(options.dim), \(milliseconds(elapsed)) ms total)")
        for (rank, hit) in ranked.prefix(top).enumerated() {
            print(String(format: "%2d. %.4f  ", rank + 1, hit.1) + files[hit.0].lastPathComponent)
        }
    }
}

private func milliseconds(_ d: Duration) -> Int {
    Int(d.components.seconds * 1000) + Int(d.components.attoseconds / 1_000_000_000_000_000)
}

private func jsonString(_ s: String) -> String {
    let data = (try? JSONEncoder().encode(s)) ?? Data("\"\"".utf8)
    return String(decoding: data, as: UTF8.self)
}
