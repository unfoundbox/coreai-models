import AVFoundation
import Accelerate
import Foundation

/// Log-mel features as HF's `Gemma4AudioFeatureExtractor` computes them (EmbeddingGemma 2 defaults):
/// 16 kHz mono, samples zero-padded to a multiple of 128, 160 zeros prepended, 320-sample periodic-Hann
/// frames every 160 samples, 512-point FFT magnitude, 128 HTK mel bins (0-8 kHz, no norm), log(x + 0.001).
/// At most 30 s (480,000 samples) per clip.
public enum GemmaAudioProcessor {
    public static let sampleRate = 16_000.0
    public static let maxSamples = 480_000
    public static let melBins = 128
    static let frameLength = 320
    static let hop = 160
    static let fftLength = 512
    static let melFloor = 0.001
    static let padMultiple = 128

    public struct Features: Sendable {
        /// Row-major (frames, 128).
        public let mel: [Float]
        /// Per frame: true when the frame lies inside the real samples.
        public let mask: [Bool]
        public var frames: Int { mask.count }
        public var validFrames: Int { mask.filter { $0 }.count }
        /// Audio soft tokens after the encoder's two stride-2 convolutions.
        public var tokenCount: Int { GemmaAudioProcessor.tokenCount(validFrames: validFrames) }
    }

    // MARK: Loading

    /// Mono float samples at 16 kHz from any file AVFoundation reads (wav, m4a, mp3, aiff, video soundtracks).
    public static func loadMono16k(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let inFormat = file.processingFormat
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: inFormat, to: outFormat),
            let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { throw EmbeddingError.audioDecodingFailed }
        try file.read(into: inBuffer)
        let ratio = sampleRate / inFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(inBuffer.frameLength) * ratio) + 1024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else {
            throw EmbeddingError.audioDecodingFailed
        }
        var fed = false
        var error: NSError?
        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if fed {
                outStatus.pointee = .endOfStream
                return nil
            }
            fed = true
            outStatus.pointee = .haveData
            return inBuffer
        }
        guard status != .error, let channel = outBuffer.floatChannelData?[0] else { throw EmbeddingError.audioDecodingFailed }
        return Array(UnsafeBufferPointer(start: channel, count: Int(outBuffer.frameLength)))
    }

    // MARK: Mel filterbank (HF audio_utils.mel_filter_bank, htk, norm=None)

    static let melFilters: [Double] = {
        let nFreq = fftLength / 2 + 1
        func toMel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        func toHz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let mMin = toMel(0), mMax = toMel(8000)
        let filterFreqs = (0..<(melBins + 2)).map { toHz(mMin + Double($0) * (mMax - mMin) / Double(melBins + 1)) }
        let fftFreqs = (0..<nFreq).map { Double($0) * (sampleRate / 2) / Double(nFreq - 1) }
        var bank = [Double](repeating: 0, count: nFreq * melBins)  // (freq, mel)
        for f in 0..<nFreq {
            for m in 0..<melBins {
                let down = -(filterFreqs[m] - fftFreqs[f]) / (filterFreqs[m + 1] - filterFreqs[m])
                let up = (filterFreqs[m + 2] - fftFreqs[f]) / (filterFreqs[m + 2] - filterFreqs[m + 1])
                bank[f * melBins + m] = max(0, min(down, up))
            }
        }
        return bank
    }()

    static let window: [Double] = (0..<frameLength).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(frameLength)) }

    // MARK: Features

    public static func features(samples raw: [Float]) -> Features {
        let samples = Array(raw.prefix(maxSamples))
        let padded = (samples.count + padMultiple - 1) / padMultiple * padMultiple
        // Prepend frameLength / 2 zeros (mask 0), append zeros up to the multiple of 128 (mask 0).
        let left = frameLength / 2
        var wave = [Double](repeating: 0, count: left + padded)
        for i in 0..<samples.count { wave[left + i] = Double(samples[i]) }
        let unfold = frameLength + 1
        let frames = max(0, (wave.count - unfold) / hop + 1)
        let realEnd = left + samples.count  // first index past the real samples

        let nFreq = fftLength / 2 + 1
        let log2n = vDSP_Length(9)  // 512
        guard let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else { return Features(mel: [], mask: []) }
        defer { vDSP_destroy_fftsetupD(setup) }
        var mel = [Float](repeating: 0, count: frames * melBins)
        var mask = [Bool](repeating: false, count: frames)
        var real = [Double](repeating: 0, count: fftLength / 2)
        var imag = [Double](repeating: 0, count: fftLength / 2)
        var frame = [Double](repeating: 0, count: fftLength)
        var magnitude = [Double](repeating: 0, count: nFreq)

        for t in 0..<frames {
            let start = t * hop
            mask[t] = start + unfold - 1 < realEnd
            for i in 0..<fftLength { frame[i] = i < frameLength ? wave[start + i] * window[i] : 0 }
            // Packed real FFT: even samples -> real, odd -> imag.
            for i in 0..<(fftLength / 2) {
                real[i] = frame[2 * i]
                imag[i] = frame[2 * i + 1]
            }
            real.withUnsafeMutableBufferPointer { rp in
                imag.withUnsafeMutableBufferPointer { ip in
                    var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    vDSP_fft_zripD(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                }
            }
            // vDSP scales by 2; DC in real[0], Nyquist in imag[0].
            magnitude[0] = abs(real[0]) / 2
            magnitude[nFreq - 1] = abs(imag[0]) / 2
            for k in 1..<(fftLength / 2) { magnitude[k] = (real[k] * real[k] + imag[k] * imag[k]).squareRoot() / 2 }
            for m in 0..<melBins {
                var acc = 0.0
                for k in 0..<nFreq { acc += magnitude[k] * melFilters[k * melBins + m] }
                mel[t * melBins + m] = Float(log(acc + melFloor))
            }
        }
        return Features(mel: mel, mask: mask)
    }

    /// Soft-token count for `validFrames` real mel frames: two stride-2 convolutions, ceil each time.
    public static func tokenCount(validFrames: Int) -> Int {
        let afterFirst = (validFrames + 1) / 2
        return (afterFirst + 1) / 2
    }
}
