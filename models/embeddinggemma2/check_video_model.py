"""Parity check: Neural Engine-authored vision (P=1260 per frame) + text encoder vs the fp32 HF video reference.

Uses HF's processor on the sampled frames (fixtures/videos/<stem>/frame_XX.png) for pixels and positions.
Runs in the reference env:

    uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 \
        --with "sentence-transformers>=6.1" --with pillow --with torchvision --with av --with safetensors \
        python check_video_model.py --device mps --dtype float16 [--dump-inputs OUT.npz]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor

from check_image_model import FIX, MODEL_ID, REVISION, text_mod, vision_mod

NUM_PATCHES = 1260
BUCKETS = (512, 2048, 4608)
VIDEO_TOKEN_ID = 258884


def prepare(processor, frames: list[np.ndarray]):
    inputs = processor(text=["<|video|>"], videos=[np.stack(frames)], do_sample_frames=False, return_tensors="pt")
    pv, pos = inputs["pixel_values_videos"], inputs["video_position_ids"]  # (F, 1260, 768), (F, 1260, 2)
    out_p, out_pos = [], []
    for f in range(pv.shape[0]):
        real = pos[f, :, 0] >= 0
        gw, gh = int(pos[f, real, 0].max()) + 1, int(pos[f, real, 1].max()) + 1
        order = vision_mod.block_order(gh, gw)
        n_real = gh * gw
        p = torch.zeros(NUM_PATCHES, pv.shape[2])
        q = torch.full((NUM_PATCHES, 2), -1, dtype=torch.int32)
        p[:n_real] = pv[f, :n_real][order]
        q[:n_real] = pos[f, :n_real][order].int()
        out_p.append(p)
        out_pos.append(q)
    return torch.stack(out_p), torch.stack(out_pos), inputs["input_ids"][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--dump-inputs", type=Path, default=None)
    args = ap.parse_args()
    dtype, dev = getattr(torch, args.dtype), args.device

    ref = json.loads((FIX / "video_reference.json").read_text())["items"]
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    clips = []
    for item in ref:
        fdir = FIX / "videos" / item["stem"]
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(fdir.glob("frame_*.png"))]
        clips.append((item, *prepare(processor, frames)))

    if args.dump_inputs:
        out = {}
        for item, patches, positions, ids in clips:
            out[f"{item['stem']}__patches"] = patches.numpy().astype(np.float16)  # (F, P, 768)
            out[f"{item['stem']}__positions"] = positions.numpy()
            out[f"{item['stem']}__ids"] = ids.numpy().astype(np.int32)
        np.savez_compressed(args.dump_inputs, **out)
        print(f"wrote {args.dump_inputs}")
        return

    weights = hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION)
    scales = FIX / "fp16_scales.json"
    vpos = vision_mod.VisionPositions()
    venc = vision_mod.VisionEncoder(NUM_PATCHES, text_mod.load_scales(scales, "vision"))
    vision_mod.load_vision_weights(vpos, venc, weights)
    vpos, venc = vpos.eval(), venc.eval().to(device=dev, dtype=dtype)
    texts = {}

    for item, patches, positions, ids in clips:
        to = lambda t: t.to(device=dev, dtype=dtype)  # noqa: E731
        soft = []
        with torch.no_grad():
            for f in range(patches.shape[0]):
                xy = positions[f].clamp(min=0)
                pe = vpos(xy[None, :, 0], xy[None, :, 1])
                cos, sin, pm = vision_mod.rope_and_mask(positions[f])
                n = int((positions[f, :, 0] >= 0).sum()) // 9
                s = venc(to(patches[f].t()[None, :, None, :]), to(pe), to(cos), to(sin), to(pm))
                soft.append(s[0, :, 0, :n])
            soft = torch.cat(soft, dim=1)  # (512, F * n)

            L = len(ids)
            b = next(b for b in BUCKETS if L <= b)
            if b not in texts:
                texts[b] = text_mod.load_weights(
                    text_mod.EmbeddingGemma2Text(b, text_mod.load_scales(scales)), weights).eval().to(device=dev, dtype=dtype)
            text = texts[b]
            x = torch.zeros(1, b, dtype=torch.long)
            x[0, :L] = torch.where(ids == VIDEO_TOKEN_ID, 0, ids)
            emb = text.embed(x.to(dev))
            cols = torch.nonzero(ids == VIDEO_TOKEN_ID).flatten()
            assert len(cols) == soft.shape[1], (len(cols), soft.shape)
            emb[0, :, 0, cols.to(dev)] = soft
            mask = torch.zeros(1, b, device=dev, dtype=dtype)
            mask[0, :L] = 1
            v = text.encoder(emb, mask)[0].float().cpu().numpy()
        v = v / np.linalg.norm(v)
        c = float(np.dot(v, np.array(item["embedding"])))
        print(f"{item['stem']:24s} frames {patches.shape[0]:2d}  seq {L:5d} bucket {b:5d}  embedding cos {c:.6f}")


if __name__ == "__main__":
    main()
