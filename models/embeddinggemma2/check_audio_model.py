"""Parity check: Neural Engine-authored audio encoder (+ text encoder) vs HF (soft tokens) and the fp32 reference.

    uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 \
        --with "sentence-transformers>=6.1" --with pillow --with torchvision --with safetensors \
        python check_audio_model.py --device mps --dtype float16 [--dump-inputs OUT.npz]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from check_image_model import FIX, MODEL_ID, REVISION, SRC, text_mod

spec = importlib.util.spec_from_file_location("coreai_models.models.ios.embedding_gemma2_audio", SRC / "embedding_gemma2_audio.py")
audio_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audio_mod
spec.loader.exec_module(audio_mod)

sys.path.insert(0, str(FIX.parent))
from make_audio_reference import read_wav  # noqa: E402

MEL_BUCKETS = (1024, 3072)
TEXT_BUCKETS = (128, 512, 2048)
AUDIO_TOKEN_ID = 258881


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--dump-inputs", type=Path, default=None)
    args = ap.parse_args()
    dtype, dev = getattr(torch, args.dtype), args.device

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu", model_kwargs={"torch_dtype": torch.float32},
                             config_kwargs={"vision_config": None})
    hf, processor = st[0].model.eval(), st[0].processor
    ref = json.loads((FIX / "audio_reference.json").read_text())["items"]

    weights = hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION)
    scales = FIX / "fp16_scales.json"
    encoders, texts, dump = {}, {}, {}
    for item in ref:
        x = read_wav(FIX / "audio" / f"{item['stem']}.wav")
        inputs = processor(text=["<|audio|>"], audio=[x], sampling_rate=16000, return_tensors="pt")
        mel, fmask, ids = inputs["input_features"][0], inputs["input_features_mask"][0], inputs["input_ids"][0]
        n_mel = mel.shape[0]
        tb = next(b for b in MEL_BUCKETS if n_mel <= b)
        mel_p = torch.zeros(tb, 128)
        mel_p[:n_mel] = mel
        m_p = torch.zeros(tb)
        m_p[:n_mel] = fmask.float()
        n_tok = int((ids == AUDIO_TOKEN_ID).sum())
        if args.dump_inputs:
            dump[f"{item['stem']}__mel"] = mel_p.numpy().astype(np.float32)
            dump[f"{item['stem']}__mask"] = m_p.numpy().astype(np.float16)
            dump[f"{item['stem']}__ids"] = ids.numpy().astype(np.int32)
            continue

        with torch.no_grad():
            hf_soft = hf.get_audio_features(inputs["input_features"], inputs["input_features_mask"]).pooler_output[0, :n_tok]
            if tb not in encoders:
                enc = audio_mod.AudioEncoder(tb, text_mod.load_scales(scales, "audio"))
                audio_mod.load_audio_weights(enc, weights)
                encoders[tb] = enc.eval().to(device=dev, dtype=dtype)
            to = lambda t: t.to(device=dev, dtype=dtype)  # noqa: E731
            soft = encoders[tb](to(mel_p[None, None]), to(m_p[None]))[0, :, 0, :n_tok].float().cpu().t()  # (n, 512)
            tok_cos = torch.nn.functional.cosine_similarity(soft, hf_soft, dim=-1)

            L = len(ids)
            b = next(b for b in TEXT_BUCKETS if L <= b)
            if b not in texts:
                texts[b] = text_mod.load_weights(text_mod.EmbeddingGemma2Text(b, text_mod.load_scales(scales)),
                                                 weights).eval().to(device=dev, dtype=dtype)
            text = texts[b]
            xids = torch.zeros(1, b, dtype=torch.long)
            xids[0, :L] = torch.where(ids == AUDIO_TOKEN_ID, 0, ids)
            emb = text.embed(xids.to(dev))
            cols = torch.nonzero(ids == AUDIO_TOKEN_ID).flatten().to(dev)
            emb[0, :, 0, cols] = soft.t().to(device=dev, dtype=dtype)
            mask = torch.zeros(1, b, device=dev, dtype=dtype)
            mask[0, :L] = 1
            v = text.encoder(emb, mask)[0].float().cpu().numpy()
        v = v / np.linalg.norm(v)
        c = float(np.dot(v, np.array(item["embedding"])))
        print(f"{item['stem']:20s} mel {n_mel:4d}->{tb}  tokens {n_tok:3d}  soft-token cos min {tok_cos.min():.6f} "
              f"mean {tok_cos.mean():.6f}  embedding cos {c:.6f}")
    if args.dump_inputs:
        np.savez_compressed(args.dump_inputs, **dump)
        print(f"wrote {args.dump_inputs}")


if __name__ == "__main__":
    main()
