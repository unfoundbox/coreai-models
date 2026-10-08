"""Export EmbeddingGemma 2 to a Core AI ``.aimodel`` (fp16 encoders in Neural Engine layout).

Text: two functions per static bucket L.
    embed_tokens_{L}      input_ids int32 [1, L] (right-padded with 0) -> inputs_embeds fp16 [1, 512, 1, L]
    embed_text_{L}        inputs_embeds fp16 [1, 512, 1, L], attention_mask fp16 [1, L]
                          -> embedding fp16 [1, 768], masked mean, NOT L2-normalized (normalize in the caller)
Image: two functions for a fixed budget of P = 2520 patches (280 soft tokens).
    image_positions_2520  x_ids, y_ids int32 [1, P] (patch column / row; 0 for padding) -> pos_embeds fp16 [1, 768, 1, P]
    embed_image_2520      pixel_patches fp16 [1, 768, 1, P] (0..1, patches in 3x3-block order), pos_embeds,
                          rope_cos / rope_sin fp16 [1, 64, 1, P], patch_mask fp16 [1, P] (1 = real, 0 = padding)
                          -> soft_tokens fp16 [1, 512, 1, 280]; the first N are real
    The caller computes rope_cos / rope_sin / patch_mask from the patch coordinates in fp32
    (``embedding_gemma2_vision.rope_and_mask``).
An image embedding is the text model run on `<bos><|image>` + N image slots + `<image|><eos>`, with the
slots' inputs_embeds replaced by the soft tokens. Lookups (int32) and RoPE angles (fp32) run apart from
the fp16 encoders, which target the Neural Engine.

    uv run python export.py                                   # text 128/512/2048 + image
    USE_OS_COREAI=1 uv run python export.py --check \\
        --image-inputs <npz from check_image_model.py --dump-inputs>   # + parity vs fp32 fixtures
"""

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import torch
from coreai_torch import TorchConverter, get_decomp_table
from huggingface_hub import hf_hub_download

from coreai_models.models.ios.embedding_gemma2_text import (
    HIDDEN,
    EmbeddingGemma2Text,
    TextEncoder,
    TokenEmbedding,
    load_scales,
    load_weights,
)
from coreai_models.models.ios.embedding_gemma2_audio import AudioEncoder, load_audio_weights
from coreai_models.models.ios.embedding_gemma2_vision import (
    HEAD_DIM,
    VisionEncoder,
    VisionPositions,
    load_vision_weights,
    rope_and_mask,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIX = HERE / "reference" / "fixtures"
SCALES = FIX / "fp16_scales.json"
MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
NUM_PATCHES = 2520  # image budget (280 soft tokens)
VIDEO_PATCHES = 1260  # per-frame budget for video (140 soft tokens)
IMAGE_TOKEN_ID = 258880
VIDEO_TOKEN_ID = 258884
AUDIO_TOKEN_ID = 258881
MEL_BINS = 128
VISION_HIDDEN = 768


def tokens_fn(seq_len: int) -> str:
    return f"embed_tokens_{seq_len}"


def text_fn(seq_len: int) -> str:
    return f"embed_text_{seq_len}"


def positions_fn(p: int) -> str:
    return f"image_positions_{p}"


def image_fn(p: int) -> str:
    return f"embed_image_{p}"


def audio_fn(mel_frames: int) -> str:
    return f"embed_audio_{mel_frames}"


POSITIONS_FN = positions_fn(NUM_PATCHES)
IMAGE_FN = image_fn(NUM_PATCHES)


def weights_path() -> str:
    return hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION)


def build_text(buckets: list[int]) -> tuple[TokenEmbedding, dict[int, TextEncoder]]:
    """One encoder per bucket. Parameters are shared, so the asset stores the weights once."""
    scales = load_scales(SCALES, "text")
    base = load_weights(EmbeddingGemma2Text(buckets[0], scales), weights_path()).eval().half()
    encoders = {buckets[0]: base.encoder}
    shared = dict(base.encoder.named_parameters())
    for b in buckets[1:]:
        enc = TextEncoder(b, scales).eval().half()
        for name, param in shared.items():
            owner, _, leaf = name.rpartition(".")
            setattr(enc.get_submodule(owner) if owner else enc, leaf, param)
        encoders[b] = enc
    return base.embed, encoders


def build_vision(budgets: list[int]) -> tuple[VisionPositions, dict[int, VisionEncoder]]:
    """One encoder per patch budget; parameters are shared, so the asset stores them once."""
    scales = load_scales(SCALES, "vision")
    positions = VisionPositions()
    base = VisionEncoder(budgets[0], scales)
    load_vision_weights(positions, base, weights_path())
    encoders = {budgets[0]: base.eval().half()}
    shared = dict(encoders[budgets[0]].named_parameters())
    for p in budgets[1:]:
        enc = VisionEncoder(p, scales).eval().half()
        for name, param in shared.items():
            owner, _, leaf = name.rpartition(".")
            setattr(enc.get_submodule(owner) if owner else enc, leaf, param)
        encoders[p] = enc
    return positions.eval().half(), encoders


def build_audio(buckets: list[int]) -> dict[int, AudioEncoder]:
    """One audio encoder per mel-frame bucket; parameters and constants shared."""
    scales = load_scales(SCALES, "audio")
    base = AudioEncoder(buckets[0], scales)
    load_audio_weights(base, weights_path())
    encoders = {buckets[0]: base.eval().half()}
    shared = dict(encoders[buckets[0]].named_parameters())
    shared.update({k: v for k, v in encoders[buckets[0]].named_buffers()})
    for t in buckets[1:]:
        enc = AudioEncoder(t, scales).eval().half()
        for name, tensor in shared.items():
            owner, _, leaf = name.rpartition(".")
            mod = enc.get_submodule(owner) if owner else enc
            if isinstance(tensor, torch.nn.Parameter):
                setattr(mod, leaf, tensor)
            else:
                getattr(mod, leaf).copy_(tensor)
        encoders[t] = enc
    return encoders


def convert(text, vision, debug: bool, audio=None):
    decomp = get_decomp_table()
    converter = TorchConverter(mode=TorchConverter.Mode.DEBUG if debug else TorchConverter.Mode.RELEASE)

    def add(module, example, inputs, outputs, name):
        t = time.time()
        ep = torch.export.export(module, example).run_decompositions(decomp)
        converter.add_exported_program(ep, input_names=inputs, output_names=outputs, entrypoint_name=name)
        print(f"[export] traced {name} in {time.time() - t:.1f}s")

    with torch.no_grad():
        if text:
            embed, encoders = text
            for b, enc in encoders.items():
                add(embed, (torch.zeros(1, b, dtype=torch.int32),), ["input_ids"], ["inputs_embeds"], tokens_fn(b))
                mask = torch.zeros(1, b, dtype=torch.float16)
                mask[0, :8] = 1
                add(enc, (torch.zeros(1, HIDDEN, 1, b, dtype=torch.float16), mask),
                    ["inputs_embeds", "attention_mask"], ["embedding"], text_fn(b))
        if vision:
            vpos, vencs = vision
            f16 = lambda *s: torch.zeros(*s, dtype=torch.float16)  # noqa: E731
            for p, venc in vencs.items():
                # Two distinct tensors: passing one tensor twice makes torch.export alias the inputs,
                # and both lookups then read the same ids.
                x_ids = torch.zeros(1, p, dtype=torch.int32)
                y_ids = torch.zeros(1, p, dtype=torch.int32)
                add(vpos, (x_ids, y_ids), ["x_ids", "y_ids"], ["pos_embeds"], positions_fn(p))
                add(venc, (f16(1, VISION_HIDDEN, 1, p), f16(1, VISION_HIDDEN, 1, p),
                           f16(1, HEAD_DIM, 1, p), f16(1, HEAD_DIM, 1, p), f16(1, p)),
                    ["pixel_patches", "pos_embeds", "rope_cos", "rope_sin", "patch_mask"], ["soft_tokens"], image_fn(p))
        if audio:
            for t, enc in audio.items():
                mask = torch.zeros(1, t, dtype=torch.float16)
                mask[0, :100] = 1
                add(enc, (torch.zeros(1, 1, t, MEL_BINS, dtype=torch.float16), mask),
                    ["mel", "frame_mask"], ["soft_tokens"], audio_fn(t))
    return converter.to_coreai()


async def check(asset_path: Path, buckets: list[int], image_inputs: Path | None, compute: str,
                video_inputs: Path | None = None, audio_inputs: Path | None = None) -> None:
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    kinds = {"neural-engine": ComputeUnitKind.neural_engine(), "gpu": ComputeUnitKind.gpu()}
    opts = SpecializationOptions.from_preferred_compute_unit_kind(kinds[compute]) if compute in kinds \
        else SpecializationOptions.default()
    t = time.time()
    model = await AIModel.load(asset_path, opts)
    print(f"[check] loaded + specialized ({compute}) in {time.time() - t:.1f}s; functions {model.function_names}")
    names = set(model.function_names)
    tok_fns = {b: model.load_function(tokens_fn(b)) for b in buckets if tokens_fn(b) in names}
    txt_fns = {b: model.load_function(text_fn(b)) for b in buckets if text_fn(b) in names}

    async def text_embedding(ids: np.ndarray, soft: np.ndarray | None = None) -> tuple[np.ndarray, int, float]:
        n = len(ids)
        b = next(b for b in sorted(txt_fns) if n <= b)
        slots = (ids == IMAGE_TOKEN_ID) | (ids == VIDEO_TOKEN_ID) | (ids == AUDIO_TOKEN_ID)
        x = np.zeros((1, b), dtype=np.int32)
        x[0, :n] = np.where(slots, 0, ids)
        m = np.zeros((1, b), dtype=np.float16)
        m[0, :n] = 1
        emb = (await tok_fns[b]({"input_ids": NDArray(x)}))["inputs_embeds"]
        if soft is not None:
            e = np.array(emb.numpy(), dtype=np.float16)
            cols = e[0, :, 0]  # (512, L) view
            cols[:, np.nonzero(slots)[0]] = soft
            emb = NDArray(np.ascontiguousarray(e))
        t0 = time.time()
        out = await txt_fns[b]({"inputs_embeds": emb, "attention_mask": NDArray(m)})
        v = np.asarray(out["embedding"].numpy(), dtype=np.float32)[0]
        return v / np.linalg.norm(v), b, time.time() - t0

    if txt_fns:
        ref = np.load(FIX / "text_reference.npz")
        cos = []
        for i, n in enumerate(ref["lengths"]):
            n = int(n)
            if n > max(txt_fns):
                continue
            v, b, dt = await text_embedding(ref["input_ids"][i, :n])
            cos.append(float(np.dot(v, ref["embeddings"][i])))
            print(f"[check] text  len {n:5d} bucket {b:5d} cos {cos[-1]:.6f}  encoder {dt * 1000:.1f} ms")
        print(f"[check] text  min cos {min(cos):.6f}  below 0.999: {sum(c < 0.999 for c in cos)}/{len(cos)}")

    async def soft_tokens(p: int, patches: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float]:
        """One image/frame: patches (P, 768) fp16 block order, positions (P, 2) -> (512, N) fp16."""
        pos_fn, img_fn = model.load_function(positions_fn(p)), model.load_function(image_fn(p))
        xy = np.clip(positions, 0, None).astype(np.int32)
        pe = await pos_fn({"x_ids": NDArray(np.ascontiguousarray(xy[None, :, 0])),
                           "y_ids": NDArray(np.ascontiguousarray(xy[None, :, 1]))})
        cos_t, sin_t, mask_t = (t.numpy() for t in rope_and_mask(torch.from_numpy(positions)))
        t0 = time.time()
        out = await img_fn({"pixel_patches": NDArray(np.ascontiguousarray(patches.T[None, :, None, :])),
                            "pos_embeds": pe["pos_embeds"], "rope_cos": NDArray(np.ascontiguousarray(cos_t)),
                            "rope_sin": NDArray(np.ascontiguousarray(sin_t)),
                            "patch_mask": NDArray(np.ascontiguousarray(mask_t))})
        n = int((positions[:, 0] >= 0).sum()) // 9
        return np.asarray(out["soft_tokens"].numpy(), dtype=np.float16)[0, :, 0, :n], time.time() - t0

    if video_inputs and image_fn(VIDEO_PATCHES) in names and txt_fns:
        import json
        ref = json.loads((FIX / "video_reference.json").read_text())["items"]
        data = np.load(video_inputs)
        cos = []
        for item in ref:
            stem = item["stem"]
            patches, positions, ids = data[f"{stem}__patches"], data[f"{stem}__positions"], data[f"{stem}__ids"]
            frames, dt_vis = [], 0.0
            for f in range(patches.shape[0]):
                s, dt = await soft_tokens(VIDEO_PATCHES, patches[f], positions[f])
                frames.append(s)
                dt_vis += dt
            v, b, dt_txt = await text_embedding(ids, np.concatenate(frames, axis=1))
            cos.append(float(np.dot(v, np.array(item["embedding"]))))
            print(f"[check] video {stem:24s} frames {len(frames):2d} bucket {b:5d}  embedding cos {cos[-1]:.6f}  "
                  f"vision {dt_vis * 1000:.0f} ms + text {dt_txt * 1000:.0f} ms")
        print(f"[check] video min cos {min(cos):.6f}  below 0.999: {sum(c < 0.999 for c in cos)}/{len(cos)}")

    if audio_inputs and txt_fns and any(n.startswith("embed_audio_") for n in names):
        import json
        ref = json.loads((FIX / "audio_reference.json").read_text())["items"]
        data = np.load(audio_inputs)
        cos = []
        for item in ref:
            stem = item["stem"]
            mel, mask, ids = data[f"{stem}__mel"], data[f"{stem}__mask"], data[f"{stem}__ids"]
            fn = model.load_function(audio_fn(mel.shape[0]))
            t0 = time.time()
            out = await fn({"mel": NDArray(np.ascontiguousarray(mel[None, None].astype(np.float16))),
                            "frame_mask": NDArray(np.ascontiguousarray(mask[None]))})
            dt_aud = time.time() - t0
            n = int((ids == AUDIO_TOKEN_ID).sum())
            soft = np.asarray(out["soft_tokens"].numpy(), dtype=np.float16)[0, :, 0, :n]
            v, b, dt_txt = await text_embedding(ids, soft)
            cos.append(float(np.dot(v, np.array(item["embedding"]))))
            print(f"[check] audio {stem:20s} mel {mel.shape[0]:4d} tokens {n:3d}  embedding cos {cos[-1]:.6f}  "
                  f"audio {dt_aud * 1000:.0f} ms + text({b}) {dt_txt * 1000:.0f} ms")
        print(f"[check] audio min cos {min(cos):.6f}  below 0.999: {sum(c < 0.999 for c in cos)}/{len(cos)}")

    if image_inputs and POSITIONS_FN in names:
        pos_fn, img_fn = model.load_function(POSITIONS_FN), model.load_function(IMAGE_FN)
        ref = np.load(FIX / "image_reference.npz", allow_pickle=True)
        data = np.load(image_inputs)
        cos = []
        for i, stem in enumerate(ref["stems"]):
            patches = data[f"{stem}__patches"]  # (P, 768) fp16, block order
            ids = data[f"{stem}__ids"]
            n_soft = int((ids == IMAGE_TOKEN_ID).sum())
            positions = data[f"{stem}__positions"]  # (P, 2), -1 = padding
            xy = np.clip(positions, 0, None).astype(np.int32)
            p = await pos_fn({"x_ids": NDArray(np.ascontiguousarray(xy[None, :, 0])),
                              "y_ids": NDArray(np.ascontiguousarray(xy[None, :, 1]))})
            cos_t, sin_t, mask_t = (t.numpy() for t in rope_and_mask(torch.from_numpy(positions)))
            t0 = time.time()
            out = await img_fn({"pixel_patches": NDArray(np.ascontiguousarray(patches.T[None, :, None, :])),
                                "pos_embeds": p["pos_embeds"], "rope_cos": NDArray(np.ascontiguousarray(cos_t)),
                                "rope_sin": NDArray(np.ascontiguousarray(sin_t)),
                                "patch_mask": NDArray(np.ascontiguousarray(mask_t))})
            dt_img = time.time() - t0
            soft = np.asarray(out["soft_tokens"].numpy(), dtype=np.float16)[0, :, 0, :n_soft]
            r = ref["soft_tokens"][i].astype(np.float32)
            s32 = soft.astype(np.float32).T
            tok_cos = (s32 * r).sum(1) / (np.linalg.norm(s32, axis=1) * np.linalg.norm(r, axis=1))
            if not txt_fns:  # image-only asset: soft tokens only
                print(f"[check] image {stem:18s} soft-token cos min {tok_cos.min():.5f}  vision {dt_img * 1000:.0f} ms")
                continue
            v, b, dt_txt = await text_embedding(ids, soft)
            cos.append(float(np.dot(v, ref["embeddings"][i])))
            print(f"[check] image {stem:18s} soft-token cos min {tok_cos.min():.5f}  embedding cos {cos[-1]:.6f}  "
                  f"vision {dt_img * 1000:.0f} ms + text({b}) {dt_txt * 1000:.0f} ms")
        if cos:
            print(f"[check] image min cos {min(cos):.6f}  below 0.999: {sum(c < 0.999 for c in cos)}/{len(cos)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="128,512,2048,4608")
    ap.add_argument("--image-budgets", default=f"{NUM_PATCHES},{VIDEO_PATCHES}",
                    help="vision patch budgets: 2520 = images (280 tokens), 1260 = video frames (140 tokens)")
    ap.add_argument("--modalities", default="text,image,audio")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "exports")
    ap.add_argument("--name", default="embeddinggemma2_fp16")
    ap.add_argument("--check", action="store_true", help="parity vs fp32 fixtures (needs USE_OS_COREAI=1)")
    ap.add_argument("--check-only", type=Path, default=None, help="skip export; check this existing .aimodel")
    ap.add_argument("--image-inputs", type=Path, default=None, help="npz from check_image_model.py --dump-inputs")
    ap.add_argument("--video-inputs", type=Path, default=None, help="npz from check_video_model.py --dump-inputs")
    ap.add_argument("--audio-inputs", type=Path, default=None, help="npz from check_audio_model.py --dump-inputs")
    ap.add_argument("--audio-buckets", default="1024,3072", help="mel-frame buckets (100 frames = 1 s; 4 frames = 1 token)")
    ap.add_argument("--compute", default="neural-engine", choices=["neural-engine", "gpu", "default"])
    ap.add_argument("--debug-info", action="store_true")
    args = ap.parse_args()
    buckets = sorted(int(b) for b in args.buckets.split(","))
    modalities = set(args.modalities.split(","))

    path = args.check_only
    if path is None:
        text = build_text(buckets) if "text" in modalities else None
        budgets = [int(p) for p in args.image_budgets.split(",")]
        vision = build_vision(budgets) if "image" in modalities else None
        audio = build_audio([int(t) for t in args.audio_buckets.split(",")]) if "audio" in modalities else None
        program = convert(text, vision, debug=args.debug_info, audio=audio)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"{args.name}.aimodel"
        program.save_asset(path)
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
        print(f"[export] saved {path} ({size / 2**20:.0f} MiB)")
    if args.check or args.check_only:
        asyncio.run(check(path, buckets, args.image_inputs, args.compute, args.video_inputs, args.audio_inputs))


if __name__ == "__main__":
    main()
