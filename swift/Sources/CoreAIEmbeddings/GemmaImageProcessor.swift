import CoreGraphics
import Foundation

/// Inputs for `image_positions_2520` and `embed_image_2520`, built the way HF's Gemma4ImageProcessor
/// builds them, with patches reordered into 3x3 blocks (see `models/embeddinggemma2/CONTRACT.md`).
public struct GemmaImageInputs: Sendable {
    /// Patch grid of the resized image.
    public let gridHeight: Int
    public let gridWidth: Int
    /// Number of soft tokens (one per 3x3 block of patches).
    public var softTokenCount: Int { gridHeight * gridWidth / (GemmaImageProcessor.pool * GemmaImageProcessor.pool) }
    /// Channel-major (768, P): pixel values 0...1, patch flattened as (row, col, channel).
    public let pixelPatches: [Float16]
    public let xIDs: [Int32]
    public let yIDs: [Int32]
    /// Channel-major (64, P).
    public let ropeCos: [Float16]
    public let ropeSin: [Float16]
    /// (P): 1 for real patches, 0 for padding.
    public let patchMask: [Float16]
}

public enum GemmaImageProcessor {
    public static let patch = 16
    public static let pool = 3
    public static let maxPatches = 2520
    static let patchValues = 3 * patch * patch  // 768
    static let headDim = 64
    static let ropeTheta: Float = 100

    // MARK: Resize target

    /// HF `get_aspect_ratio_preserving_size`: the largest size with sides that are multiples of 48 and
    /// at most `maxPatches` 16x16 patches. Returns (height, width).
    public static func targetSize(height: Int, width: Int) -> (height: Int, width: Int) {
        let factor = (Double(maxPatches * patch * patch) / Double(height * width)).squareRoot()
        let side = pool * patch
        var th = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var tw = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        let maxSide = (maxPatches / (pool * pool)) * side
        if th == 0 {
            th = side
            tw = min((width / height) * side, maxSide)
        } else if tw == 0 {
            tw = side
            th = min((height / width) * side, maxSide)
        }
        return (th, tw)
    }

    // MARK: Pixels

    /// RGB8 pixels of `image` at its own size, in sRGB. Row-major (h, w, 3).
    public static func rgbPixels(_ image: CGImage) throws -> (pixels: [UInt8], width: Int, height: Int) {
        let w = image.width, h = image.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw EmbeddingError.imageDecodingFailed }
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return (rgb, w, h)
    }

    // MARK: Resize (PIL / torchvision antialiased bicubic)

    private static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        return 0
    }

    /// PIL `precompute_coeffs`: per output index, the first input index and normalized weights.
    private static func coefficients(inSize: Int, outSize: Int) -> [(start: Int, weights: [Double])] {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        return (0..<outSize).map { i in
            let center = (Double(i) + 0.5) * scale
            let xmin = max(Int(center - support + 0.5), 0)
            let count = min(Int(center + support + 0.5), inSize) - xmin
            var w = (0..<count).map { cubic((Double($0 + xmin) - center + 0.5) / filterScale) }
            let sum = w.reduce(0, +)
            w = w.map { $0 / sum }
            return (xmin, w)
        }
    }

    /// One pass along x (`horizontal`) or y, RGB8 in and out, rounded and clamped like PIL.
    private static func resamplePass(
        _ src: [UInt8], width: Int, height: Int, outSize: Int, horizontal: Bool
    ) -> [UInt8] {
        let outW = horizontal ? outSize : width
        let outH = horizontal ? height : outSize
        let coeffs = coefficients(inSize: horizontal ? width : height, outSize: outSize)
        var dst = [UInt8](repeating: 0, count: outW * outH * 3)
        for y in 0..<outH {
            for x in 0..<outW {
                let (start, weights) = coeffs[horizontal ? x : y]
                var acc: (Double, Double, Double) = (0, 0, 0)
                for (k, wk) in weights.enumerated() {
                    let sx = horizontal ? start + k : x
                    let sy = horizontal ? y : start + k
                    let s = (sy * width + sx) * 3
                    acc.0 += wk * Double(src[s])
                    acc.1 += wk * Double(src[s + 1])
                    acc.2 += wk * Double(src[s + 2])
                }
                let d = (y * outW + x) * 3
                dst[d] = UInt8(max(0, min(255, acc.0.rounded())))
                dst[d + 1] = UInt8(max(0, min(255, acc.1.rounded())))
                dst[d + 2] = UInt8(max(0, min(255, acc.2.rounded())))
            }
        }
        return dst
    }

    /// Resize RGB8 (h, w, 3) to (outH, outW, 3): horizontal pass, then vertical.
    public static func resize(_ rgb: [UInt8], width: Int, height: Int, outWidth: Int, outHeight: Int) -> [UInt8] {
        var out = rgb
        var w = width
        if w != outWidth {
            out = resamplePass(out, width: w, height: height, outSize: outWidth, horizontal: true)
            w = outWidth
        }
        if height != outHeight {
            out = resamplePass(out, width: w, height: height, outSize: outHeight, horizontal: false)
        }
        return out
    }

    // MARK: Patches, positions, RoPE

    /// Full preprocessing for one image.
    public static func inputs(for image: CGImage) throws -> GemmaImageInputs {
        let (rgb, w, h) = try rgbPixels(image)
        let (th, tw) = targetSize(height: h, width: w)
        let resized = resize(rgb, width: w, height: h, outWidth: tw, outHeight: th)
        return inputs(resizedRGB: resized, width: tw, height: th)
    }

    /// Preprocessing after the resize (exposed for tests that start from HF's resized pixels).
    public static func inputs(resizedRGB rgb: [UInt8], width: Int, height: Int) -> GemmaImageInputs {
        let gh = height / patch, gw = width / patch
        let p = maxPatches
        var pixels = [Float16](repeating: 0, count: patchValues * p)
        var xIDs = [Int32](repeating: 0, count: p)
        var yIDs = [Int32](repeating: 0, count: p)
        var mask = [Float16](repeating: 0, count: p)
        var cos = [Float16](repeating: 0, count: headDim * p)
        var sin = [Float16](repeating: 0, count: headDim * p)
        let freqs = (0..<(headDim / 4)).map { 1 / pow(ropeTheta, Float(2 * $0) / Float(headDim / 2)) }  // 16

        // 3x3 blocks in row-major block order; inside a block, row-major patches.
        var slot = 0
        for by in 0..<(gh / pool) {
            for bx in 0..<(gw / pool) {
                for dy in 0..<pool {
                    for dx in 0..<pool {
                        let py = by * pool + dy, px = bx * pool + dx
                        // Patch values flatten as (row, col, channel).
                        for r in 0..<patch {
                            for c in 0..<patch {
                                let src = ((py * patch + r) * width + (px * patch + c)) * 3
                                for ch in 0..<3 {
                                    let v = (r * patch + c) * 3 + ch
                                    pixels[v * p + slot] = Float16(Float(rgb[src + ch]) / 255)
                                }
                            }
                        }
                        xIDs[slot] = Int32(px)
                        yIDs[slot] = Int32(py)
                        mask[slot] = 1
                        for (i, f) in freqs.enumerated() {
                            let ax = Float(px) * f, ay = Float(py) * f
                            for (off, a) in [(0, ax), (16, ax), (32, ay), (48, ay)] {
                                cos[(off + i) * p + slot] = Float16(Foundation.cos(a))
                                sin[(off + i) * p + slot] = Float16(Foundation.sin(a))
                            }
                        }
                        slot += 1
                    }
                }
            }
        }
        // Padding slots: angle 0 (cos 1, sin 0), mask 0, ids 0.
        for s in slot..<p {
            for d in 0..<headDim { cos[d * p + s] = 1 }
        }
        return GemmaImageInputs(
            gridHeight: gh, gridWidth: gw, pixelPatches: pixels, xIDs: xIDs, yIDs: yIDs,
            ropeCos: cos, ropeSin: sin, patchMask: mask)
    }
}
