import AVFoundation
import CoreGraphics
import CoreImage
import Foundation
import VideoToolbox

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
    ///
    /// Frame index `i` is the `i`-th frame in presentation order, as PyAV numbers them. The timestamps come
    /// from the container's sample table (no decoding); each sampled frame is then decoded by an exact-time
    /// seek. If a seek does not land on its frame, this falls back to decoding the whole track.
    public static func frames(from url: URL, maxFrames: Int = maxFrames) async throws -> [CGImage] {
        try await sample(url, maxFrames: maxFrames, mode: .seekOrDecodeAll)
    }

    enum Mode { case seekOrDecodeAll, seekOnly, decodeAll }

    /// One decoding strategy, without fallback (tests compare them).
    static func frames(from url: URL, mode: Mode, maxFrames: Int = maxFrames) async throws -> [CGImage] {
        try await sample(url, maxFrames: maxFrames, mode: mode)
    }

    private static func sample(_ url: URL, maxFrames: Int, mode: Mode) async throws -> [CGImage] {
        registerSupplementalDecoders
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw EmbeddingError.noFrames
        }
        let fps = Double(try await track.load(.nominalFrameRate))
        let codec = try await track.load(.formatDescriptions).first.map(CMFormatDescriptionGetMediaSubType)
        if let codec, !VTIsHardwareDecodeSupported(codec), !canDecode(asset: asset, track: track) {
            throw EmbeddingError.unsupportedCodec(fourCC(codec))
        }

        let times = try await presentationTimes(asset: asset, track: track)
        let wanted = sampleIndices(totalFrames: times.count, fps: fps, maxFrames: maxFrames)
        guard !wanted.isEmpty else { throw EmbeddingError.noFrames }

        let wantedTimes = wanted.map { times[$0] }
        if mode == .seekOnly { return try seekFrames(asset: asset, track: track, times: wantedTimes) }
        if mode == .seekOrDecodeAll, let images = try? seekFrames(asset: asset, track: track, times: wantedTimes) {
            return images
        }
        let images = try decodeAll(asset: asset, track: track, wanted: wanted)
        if images.isEmpty, let codec { throw EmbeddingError.unsupportedCodec(fourCC(codec)) }
        return images
    }

    /// VP9 and AV1 decoders that macOS ships but does not register for AVFoundation by default.
    private static let registerSupplementalDecoders: Void = {
        VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_VP9)
        VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_AV1)
    }()

    private static let context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])

    private static func rgba(_ image: CIImage) -> CGImage? {
        context.createCGImage(image, from: image.extent, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }

    /// Presentation timestamps of every frame, sorted. Uses sample cursors (sample table only) when the
    /// track offers them, else reads the compressed samples.
    private static func presentationTimes(asset: AVAsset, track: AVAssetTrack) async throws -> [CMTime] {
        var times: [CMTime] = []
        let segments = try await track.load(.segments)
        if track.canProvideSampleCursors, segments.count == 1, let map = segments.first?.timeMapping,
           let cursor = track.makeSampleCursorAtFirstSampleInDecodeOrder() {
            // Cursors report media time; readers report track time (after the edit list).
            let offset = map.target.start - map.source.start
            repeat { times.append(cursor.presentationTimeStamp + offset) } while cursor.stepInDecodeOrder(byCount: 1) == 1
        } else {
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)  // compressed: no decode
            reader.add(output)
            reader.startReading()
            while let sample = output.copyNextSampleBuffer() {
                let timing = (try? sample.sampleTimingInfos()) ?? []
                times += timing.map(\.presentationTimeStamp)
            }
        }
        return times.sorted()
    }

    /// One exact-time decode per frame. Throws if any frame is missing or lands on another time.
    /// A reader limited to each frame's time range decodes from the nearest keyframe and returns that frame.
    private static func seekFrames(asset: AVAsset, track: AVAssetTrack, times: [CMTime]) throws -> [CGImage] {
        let oneTick = CMTime(value: 1, timescale: times.first?.timescale ?? 600)
        return try times.map { time in
            let reader = try AVAssetReader(asset: asset)
            reader.timeRange = CMTimeRange(start: time, duration: oneTick)
            let output = AVAssetReaderTrackOutput(
                track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            output.alwaysCopiesSampleData = false
            reader.add(output)
            reader.startReading()
            defer { reader.cancelReading() }
            while let sample = output.copyNextSampleBuffer() {
                guard sample.outputPresentationTimeStamp == time, let pixels = CMSampleBufferGetImageBuffer(sample) else {
                    continue
                }
                if let cg = rgba(CIImage(cvPixelBuffer: pixels)) { return cg }
            }
            throw EmbeddingError.noFrames
        }
    }

    /// Decode the track front to back, keeping the wanted indices (the slow path).
    private static func decodeAll(asset: AVAsset, track: AVAssetTrack, wanted: [Int]) throws -> [CGImage] {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()
        var keep = Set(wanted)
        var byIndex: [Int: CGImage] = [:]
        var index = 0
        while !keep.isEmpty, let sample = output.copyNextSampleBuffer() {
            defer { index += 1 }
            guard keep.contains(index), let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            byIndex[index] = rgba(CIImage(cvPixelBuffer: pixels))
            keep.remove(index)
        }
        reader.cancelReading()
        return wanted.compactMap { byIndex[$0] }
    }

    /// Whether AVFoundation can decode the first frame of the track (software decoders included).
    private static func canDecode(asset: AVAsset, track: AVAssetTrack) -> Bool {
        guard let reader = try? AVAssetReader(asset: asset) else { return false }
        let output = AVAssetReaderTrackOutput(
            track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(output)
        reader.startReading()
        defer { reader.cancelReading() }
        return output.copyNextSampleBuffer().flatMap(CMSampleBufferGetImageBuffer) != nil
    }

    private static func fourCC(_ code: FourCharCode) -> String {
        let chars = [24, 16, 8, 0].map { Character(UnicodeScalar(UInt8((code >> $0) & 0xFF))) }
        return String(chars).trimmingCharacters(in: .whitespaces)
    }
}
