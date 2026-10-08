# EmbeddingGemma 2 port: working contract

Status: working agreement for the text-only macOS phase. Written 2026-10-06.

## Goal of this phase

Text embeddings from `google/embeddinggemma-2` on Core AI, with as many ops as possible on
the Neural Engine (fp16 only), matching the PyTorch fp32 reference at cosine >= 0.999.

## Facts (verified 2026-10-06)

- Checkpoint: `google/embeddinggemma-2`, snapshot `914f7f89142e33e77833254d9c9b90c3cef7303b`,
  bf16 weights, in the HF cache (`~/.cache/huggingface/hub`).
- Reference code: `transformers==5.19.0` (`transformers/models/embedding_gemma2/`). The repo
  pins 5.12.1, which lacks this model. Run reference code with:
  `uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 --with "sentence-transformers>=6.1" python ...`
  Do not change the repo pins.
- Text backbone: 24 bidirectional encoder layers, d=512, 4 heads, GQA. Sliding layers: head_dim
  256, 2 KV heads, window 512. Full layers (5, 11, 17, 23): head_dim 512, 1 KV head.
  q/k/v RMSNorm, attention scaling 1.0, no softcap. RoPE theta 1e4 (sliding) / 1e6 (full).
- PLE is projection-only: `ple.per_layer_model_projection` (512 -> 24*512) * 512**-0.5, then
  RMSNorm. No PLE lookup table.
- Each layer ends with `hidden_states *= layer_scalar` (values 0.11 to 0.97).
- Head: final RMSNorm -> `embedding_projection` (512 -> 768) per token -> masked mean pool
  -> L2 normalize. Truncation to 512/256/128 happens after, then re-normalize.
- Neural Engine element types: fp16, f8E4M3, si8, ui8, si16, ui16. No bf16, no fp32.
  Check placement with `coreai_torch.debugging.compute_plan.ComputePlan` and `USE_OS_COREAI=1`.
- Only this Mac (M1 Max, macOS 27.2) has a Neural Engine. All work runs here.

## Core AI interface (all lanes build to this)

One `.aimodel`, two functions per sequence bucket L (built: 128, 512, 2048; 8192 later).
The token lookup is separate because int32 ids cannot run on the Neural Engine.

| Function | Input | Output |
|---|---|---|
| `embed_tokens_{L}` | `input_ids` int32 [1, L], right-padded with pad id 0 | `inputs_embeds` fp16 [1, 512, 1, L] |
| `embed_text_{L}` | `inputs_embeds` fp16 [1, 512, 1, L], `attention_mask` fp16 [1, L] (1 = real, 0 = pad) | `embedding` fp16 [1, 768], masked mean, **not** normalized |

The caller (Swift) does: prompt prefix, tokenization, bucket choice, padding, L2 normalization
in fp32, MRL truncation and re-normalization.

### Image functions (fixed budget P = 2520 patches = 280 soft tokens)

| Function | Input | Output |
|---|---|---|
| `image_positions_2520` | `x_ids`, `y_ids` int32 [1, P] (patch column / row; 0 for padding) | `pos_embeds` fp16 [1, 768, 1, P] |
| `embed_image_2520` | `pixel_patches` fp16 [1, 768, 1, P] (pixels 0..1, 3x3-block order), `pos_embeds`, `rope_cos` / `rope_sin` fp16 [1, 64, 1, P], `patch_mask` fp16 [1, P] | `soft_tokens` fp16 [1, 512, 1, 280]; first N real |

Caller steps for one image:
1. Resize, keeping aspect ratio, to H x W with H, W multiples of 48 and (H/16)(W/16) <= 2520
   (HF `get_aspect_ratio_preserving_size`, bicubic, antialiased). Pixels / 255, no normalization.
2. Cut 16x16 patches; each patch flattens as (row, col, channel). Reorder patches so each 3x3
   block is contiguous, blocks in row-major order. Pad to P with zeros.
3. RoPE: per patch, angles = (x, y) * inv_freq (theta 100, 16 freqs), laid out [x, x, y, y]
   (64 values); cos/sin in fp32, then fp16. `patch_mask` 1 for real patches.
4. Tokens: `<bos> <|image> ` + N x `<|image|>` + ` <image|> <eos>` (N = real patches / 9). Run
   `embed_tokens_{L}` with image-token ids replaced by 0, overwrite the N image columns of
   `inputs_embeds` with `soft_tokens[:, :, :, :N]`, then `embed_text_{L}`.

Exact rewrites in the vision path: per-norm prescale; dropped pooler x sqrt(768) (it feeds a
weightless RMSNorm); 3x3-block patch order so pooling is a static mean of 9 neighbours.

### Placement lessons (vision)

- A function the Neural Engine compiler fails on (`CompilationFailure`) moves **every** function
  in the asset to the GPU. The first positions function (int32 lookup + clamp/where + fp32 trig)
  did this. Fix: lookup-only positions function; RoPE and mask computed by the caller.
- Neural Engine vs GPU on M1 Max for the vision encoder: ~467 ms vs ~280 ms. The Neural Engine
  is slower here; its value is power and background use.
- Export trap: tracing with the same example tensor for two inputs makes `torch.export` alias them.

## Placement findings (2026-10-07, M1 Max, macOS 27.2 beta)

- `ComputePlan` reports every op on the GPU. It loads the model in debug mode, which appears
  to keep the model off the Neural Engine. Do not use it to judge placement here.
- Without debug mode, `embed_text_128` runs in 8.6 ms with the Neural Engine preferred and
  ~29 ms with GPU preferred, default, or CPU only, and the outputs differ slightly. So the
  Neural Engine-preferred run uses different hardware.
- Proven with `sudo powermetrics --samplers cpu_power,gpu_power`: during a loop of
  `embed_text_512` (Neural Engine preferred), ANE Power is ~3.9–4.1 W per sample and GPU Power
  stays at idle (~90–130 mW). Instruments' `ane-hw-intervals` stays empty without root.
- `coreai-build` (Metal Toolchain 27A266a) fails to launch on macOS 27.2 beta: missing symbol
  `CoreAIAsset.AIModelAsset.delegatesFolderURL(artifactName:)`. Toolchain/OS mismatch.

## Paths

| Path | Contents |
|---|---|
| `models/embeddinggemma2/reference/` | fp32 reference, fp16 overflow map, fp16 fix test |
| `python/src/coreai_models/models/ios/embedding_gemma2_text.py` | Neural Engine text model |
| `python/tests/test_model_units/test_embedding_gemma2_text.py` | unit tests |
| `models/embeddinggemma2/export.py` | Core AI export |
| `swift/Sources/CoreAIEmbeddings/` | Swift wrapper |

## fp16 findings (measured 2026-10-07, 17 public texts, PyTorch MPS)

| Variant | Min cosine vs fp32 | Broken vectors |
|---|---|---|
| bf16, HF as shipped | 0.9972 | 0 |
| fp16, HF as shipped | 0.0 | 3 (the 3 longest: pooling sum overflows) |
| fp16, norms in fp16, no fixes | 0.0 | 16 (x² overflows in RMSNorm) |
| fp16, norms in fp16, prescaled norms + divide-first pooling | **0.9995** | 0 |

No activation exceeds 65,504 (max 3,935). Residual stream max is 78. Prescale per norm is
the smallest power of 2 that brings the calibrated max input to <= 8 (`fixtures/text_fp16_fix.json`).
Calibration set is small; re-check scales on a larger corpus before shipping.
