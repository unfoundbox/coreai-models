# EmbeddingGemma 2

EmbeddingGemma 2 maps text, images, video and audio into one 768-dimensional embedding space (740M parameters, 8K context, Matryoshka dimensions 768 / 512 / 256 / 128).[^1] This port runs all four modalities on the Neural Engine in fp16.

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run python export.py
```

Saves to `<repo-root>/exports/embeddinggemma2_fp16.aimodel`. One asset holds every function:

| Modality | Functions | Notes |
| -------- | --------- | ----- |
| Text | `embed_tokens_{L}`, `embed_text_{L}` for L = 128, 512, 2048, 4608 | static buckets, right-padded |
| Image | `image_positions_2520`, `embed_image_2520` | 280 soft tokens per image |
| Video | `image_positions_1260`, `embed_image_1260` | 140 soft tokens per frame, 1 frame/s, at most 32 frames |
| Audio | `embed_audio_{1024,3072}` | mel-frame buckets (10 s, 30 s) |

**Options:**

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--buckets` | Text buckets | `128,512,2048,4608` |
| `--modalities` | `text`, `image`, `audio` (video uses the image path) | `text,image,audio` |
| `--audio-buckets` | Audio buckets in mel frames (100 = 1 s) | `1024,3072` |
| `--compute` | `neural-engine`, `gpu`, `default` | `neural-engine` |
| `--output-dir` | Output directory | `<repo-root>/exports/` |
| `--check` | Parity against the fp32 fixtures (needs `USE_OS_COREAI=1`) | — |

## fp16 on the Neural Engine

The model card advises against fp16: in a naive fp16 run, `x²` inside the RMS norms overflows. No activation exceeds the fp16 range. This port fixes the overflow exactly: a per-norm power-of-2 prescale (`reference/fixtures/fp16_scales.json`), divide-first pooling, pool-then-project, and no `×√768` before the vision pooler's weightless norm. See [`CONTRACT.md`](CONTRACT.md) for the interfaces and findings.

Minimum cosine similarity against the fp32 Hugging Face model (M1 Max):

| Modality | Fixtures | Min cosine |
| -------- | -------- | ---------- |
| Text | 17 texts | 0.99997 |
| Image | 7 images | 0.999989 |
| Video | 6 clips, up to 32 frames | 0.99996 |
| Audio | 9 clips | 0.99983 |

## Swift

```sh
swift run -c release embedding-gemma embed --model exports/embeddinggemma2_fp16.aimodel "What causes the northern lights?"
swift run -c release embedding-gemma media-search --model exports/embeddinggemma2_fp16.aimodel --query "a red circle" --folder <folder>
```

The `CoreAIEmbeddings` library exposes `EmbeddingGemma2` with `embed(_:task:dimension:)`, `embed(imageAt:)`, `embed(videoAt:)` and `embed(audioAt:)`.

Known limits: macOS 27 or later (asset compatibility 27.0), tested on macOS 27.2 beta only; iOS untested. The first load compiles the asset for the device and can take several minutes.

## License

The export code follows this repository's license. The model weights are Google's and are subject to the Gemma terms linked from the model card, including the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Converted weights are not part of this repository.

[^1]: [Model card](https://huggingface.co/google/embeddinggemma-2) · [Documentation](https://ai.google.dev/gemma/docs/embeddinggemma)
