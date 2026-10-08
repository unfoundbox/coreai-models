import AVFoundation
import CoreGraphics
import CoreImage
import Foundation

/// Frame sampling for EmbeddingGemma 2 video, matching HF's `EmbeddingGemma2VideoProcessor.sample_frames`
/// with its defaults (1 frame per second, at most 32 frames, uniform thinning above that).
public enum GemmaVideoSampler {
    public static let targetFPS = 1.0
    public static let maxFrames = 32

    /// Frame indices HF samples. `duration` is `totalFrames / fps`, as in HF's PyAV reader.
    public static func sampleIndices(
        totalFrames: Int, fps: Double, targetFPS: Double = targetFPS, maxFrames: Int = maxFrames
    ) -> [Int] {
        guard totalFrames > 0, fps > 0 else { return [] }
        let duration = Double(totalFrames) / fps
        let step = fps / targetFPS
        let count = max(1, Int(duration * targetFPS))
        var indices = (0..<count).map { min(totalFrames - 1, Int(Double($0) * step)) }
        if indices.count > maxFrames {
            // numpy.linspace(0, n - 1, maxFrames, dtype=int): start + i * step, last value exact, truncated.
            let n = indices.count
            let lin = (0..<maxFrames).map { i -> Int in
                i == maxFrames - 1 ? n - 1 : Int(Double(i) * (Double(n - 1) / Double(maxFrames - 1)))
            }
            indices = lin.map { indices[$0] }
        }
        return indices
    }

    /// Decode the sampled frames of a video file, in order.
    public static func frames(from url: URL, maxFrames: Int = maxFrames) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw EmbeddingError.noFrames
        }
        let fps = Double(try await track.load(.nominalFrameRate))

        // Pass 1: count frames without decoding.
        let total = try countFrames(asset: asset, track: track)
        let wanted = sampleIndices(totalFrames: total, fps: fps, maxFrames: maxFrames)
        guard !wanted.isEmpty else { throw EmbeddingError.noFrames }

        // Pass 2: decode, keeping the wanted indices.
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()
        let context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
        var keep = Set(wanted)
        var byIndex: [Int: CGImage] = [:]
        var index = 0
        while !keep.isEmpty, let sample = output.copyNextSampleBuffer() {
            defer { index += 1 }
            guard keep.contains(index), let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            let image = CIImage(cvPixelBuffer: pixels)
            if let cg = context.createCGImage(image, from: image.extent, format: .RGBA8,
                                              colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) {
                byIndex[index] = cg
            }
            keep.remove(index)
        }
        reader.cancelReading()
        return wanted.compactMap { byIndex[$0] }
    }

    private static func countFrames(asset: AVAsset, track: AVAssetTrack) throws -> Int {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)  // compressed: no decode
        reader.add(output)
        reader.startReading()
        var count = 0
        while let sample = output.copyNextSampleBuffer() {
            count += CMSampleBufferGetNumSamples(sample)
        }
        return count
    }
}
