"""Parity check: the Neural Engine-authored PyTorch text model vs the fp32 HF reference fixtures.

    uv run python check_text_model.py --device cpu --dtype float32
    uv run python check_text_model.py --device mps --dtype float16
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from coreai_models.models.ios.embedding_gemma2_text import EmbeddingGemma2Text, load_scales, load_weights

HERE = Path(__file__).resolve().parent
FIX = HERE / "reference" / "fixtures"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
BUCKETS = (128, 512, 2048)


def bucket_for(n: int) -> int:
    return next(b for b in BUCKETS if n <= b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    ref = np.load(FIX / "text_reference.npz")
    ids, lengths, emb32 = ref["input_ids"], ref["lengths"], ref["embeddings"]
    weights = hf_hub_download("google/embeddinggemma-2", "model.safetensors", revision=REVISION)
    scales = load_scales(FIX / "text_fp16_fix.json")

    models: dict[int, EmbeddingGemma2Text] = {}
    cos = []
    for row, n in zip(ids, lengths):
        b = bucket_for(int(n))
        if b not in models:
            m = load_weights(EmbeddingGemma2Text(b, scales), weights).eval()
            models[b] = m.to(device=args.device, dtype=dtype)
        x = np.zeros((1, b), dtype=np.int64)
        x[0, :n] = row[:n]
        mask = np.zeros((1, b), dtype=np.float32)
        mask[0, :n] = 1
        with torch.no_grad():
            out = models[b](torch.from_numpy(x).to(args.device),
                            torch.from_numpy(mask).to(device=args.device, dtype=dtype))
        e = out.float().cpu().numpy()[0]
        e = e / np.linalg.norm(e)
        c = float(np.dot(e, emb32[len(cos)]))
        cos.append(c)
        print(f"len {int(n):5d}  bucket {b:5d}  cos {c:.6f}")
    print(f"\n{args.device} {args.dtype}: min cos {min(cos):.6f}  mean {np.mean(cos):.6f}  "
          f"below 0.999: {sum(c < 0.999 for c in cos)}/{len(cos)}")


if __name__ == "__main__":
    main()
