"""Parity check: Neural Engine-authored vision encoder (+ text encoder) vs the fp32 HF image reference.

Uses HF's own image processor for pixels and positions, so this measures the model port only.
Runs in the reference env (needs transformers 5.19):

    uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 \
        --with "sentence-transformers>=6.1" --with pillow --with torchvision --with safetensors \
        python check_image_model.py --device mps --dtype float16
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor

HERE = Path(__file__).resolve().parent
FIX = HERE / "reference" / "fixtures"
SRC = HERE.parents[1] / "python" / "src" / "coreai_models" / "models" / "ios"
MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
NUM_PATCHES = 2520
TEXT_BUCKET = 512


def _load(name: str, file: str):
    """Import a module file directly: ``coreai_models.models`` pulls in the coreai wheel, absent here."""
    spec = importlib.util.spec_from_file_location(name, SRC / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


text_mod = _load("coreai_models.models.ios.embedding_gemma2_text", "embedding_gemma2_text.py")
vision_mod = _load("coreai_models.models.ios.embedding_gemma2_vision", "embedding_gemma2_vision.py")


def prepare(processor, img: Image.Image):
    """HF processor output -> block-ordered, padded patches/positions, token ids, soft-token count."""
    inputs = processor(text=["<|image|>"], images=[[img]], return_tensors="pt")
    pv, pos = inputs["pixel_values"][0], inputs["image_position_ids"][0]  # (2520, 768), (2520, 2)
    real = pos[:, 0] >= 0
    grid_w, grid_h = int(pos[real, 0].max()) + 1, int(pos[real, 1].max()) + 1
    order = vision_mod.block_order(grid_h, grid_w)
    n_real = grid_h * grid_w
    patches = torch.zeros(NUM_PATCHES, pv.shape[1])
    positions = torch.full((NUM_PATCHES, 2), -1, dtype=torch.int32)
    patches[:n_real] = pv[:n_real][order]
    positions[:n_real] = pos[:n_real][order].int()
    return patches, positions, inputs["input_ids"][0], n_real // 9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--dump-inputs", type=Path, default=None,
                    help="write prepared inputs (block-ordered patches, positions, token ids) to this .npz and exit")
    args = ap.parse_args()
    dtype, dev = getattr(torch, args.dtype), args.device

    if args.dump_inputs:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        ref = np.load(FIX / "image_reference.npz", allow_pickle=True)
        out = {}
        for stem in ref["stems"]:
            img = Image.open(FIX / "images" / f"{stem}.png").convert("RGB")
            patches, positions, ids, n = prepare(processor, img)
            out[f"{stem}__patches"] = patches.numpy().astype(np.float16)
            out[f"{stem}__positions"] = positions.numpy()
            out[f"{stem}__ids"] = ids.numpy().astype(np.int32)
        np.savez_compressed(args.dump_inputs, **out)
        print(f"wrote {args.dump_inputs}")
        return

    weights = hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION)
    scales_path = FIX / "fp16_scales.json"
    vpos = vision_mod.VisionPositions()
    venc = vision_mod.VisionEncoder(NUM_PATCHES, text_mod.load_scales(scales_path, "vision"))
    vision_mod.load_vision_weights(vpos, venc, weights)
    vpos = vpos.eval()  # the GPU-side lookup function; stays on CPU here
    venc = venc.eval().to(device=dev, dtype=dtype)
    text = text_mod.load_weights(text_mod.EmbeddingGemma2Text(TEXT_BUCKET, text_mod.load_scales(scales_path)), weights)
    text = text.eval().to(device=dev, dtype=dtype)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    image_token_id = 258880

    ref = np.load(FIX / "image_reference.npz", allow_pickle=True)
    for i, stem in enumerate(ref["stems"]):
        img = Image.open(FIX / "images" / f"{stem}.png").convert("RGB")
        patches, positions, ids, n = prepare(processor, img)
        with torch.no_grad():
            ids_xy = positions.clamp(min=0)
            pos_emb = vpos(ids_xy[None, :, 0], ids_xy[None, :, 1])
            cos, sin, patch_mask = vision_mod.rope_and_mask(positions)
            to = lambda t: t.to(device=dev, dtype=dtype)  # noqa: E731
            soft = venc(to(patches.t()[None, :, None, :]), to(pos_emb), to(cos), to(sin), to(patch_mask))
            soft = soft[0, :, 0, :n].float().cpu().t()  # (n, 512)
            r = torch.from_numpy(ref["soft_tokens"][i].astype(np.float32))
            tok_cos = torch.nn.functional.cosine_similarity(soft, r, dim=-1)

            # Full path: text tokens with image positions filled by the soft tokens.
            L = len(ids)
            x = torch.zeros(1, TEXT_BUCKET, dtype=torch.long)
            x[0, :L] = torch.where(ids == image_token_id, 0, ids)
            emb = text.embed(x.to(dev))  # (1, 512, 1, TEXT_BUCKET)
            img_cols = torch.nonzero(ids == image_token_id).flatten()
            emb[0, :, 0, img_cols.to(dev)] = soft.t().to(device=dev, dtype=dtype)
            mask = torch.zeros(1, TEXT_BUCKET, device=dev, dtype=dtype)
            mask[0, :L] = 1
            v = text.encoder(emb, mask)[0].float().cpu().numpy()
        v = v / np.linalg.norm(v)
        c = float(np.dot(v, ref["embeddings"][i]))
        print(f"{stem:18s} n {n:3d}  soft-token cos min {tok_cos.min():.6f} mean {tok_cos.mean():.6f}  "
              f"embedding cos {c:.6f}")


if __name__ == "__main__":
    main()
