"""Export the EmbeddingGemma 2 text encoder to a Core AI ``.aimodel`` (fp16, Neural Engine layout).

Two functions per static bucket L:
    embed_tokens_{L}  input_ids int32 [1, L] (right-padded with 0) -> inputs_embeds fp16 [1, 512, 1, L]
    embed_text_{L}    inputs_embeds fp16 [1, 512, 1, L], attention_mask fp16 [1, L]
                      -> embedding fp16 [1, 768], masked mean, NOT L2-normalized (normalize in the caller)
The token lookup is its own function because int32 ids cannot run on the Neural Engine.

    uv run python export.py                        # buckets 128, 512, 2048
    USE_OS_COREAI=1 uv run python export.py --check  # + placement summary and parity vs fp32 fixtures
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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIX = HERE / "reference" / "fixtures"
MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def tokens_fn(seq_len: int) -> str:
    return f"embed_tokens_{seq_len}"


def text_fn(seq_len: int) -> str:
    return f"embed_text_{seq_len}"


def build_models(buckets: list[int]) -> tuple[TokenEmbedding, dict[int, TextEncoder]]:
    """One encoder per bucket. Parameters are shared, so the asset stores the weights once."""
    weights = hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION)
    scales = load_scales(FIX / "text_fp16_fix.json")
    base = load_weights(EmbeddingGemma2Text(buckets[0], scales), weights).eval().half()
    encoders = {buckets[0]: base.encoder}
    shared = dict(base.encoder.named_parameters())
    for b in buckets[1:]:
        enc = TextEncoder(b, scales).eval().half()
        for name, param in shared.items():
            owner, _, leaf = name.rpartition(".")
            setattr(enc.get_submodule(owner) if owner else enc, leaf, param)
        encoders[b] = enc
    return base.embed, encoders


def convert(embed: TokenEmbedding, encoders: dict[int, TextEncoder], debug: bool):
    decomp = get_decomp_table()
    converter = TorchConverter(mode=TorchConverter.Mode.DEBUG if debug else TorchConverter.Mode.RELEASE)
    with torch.no_grad():
        for b, enc in encoders.items():
            t = time.time()
            ids = torch.zeros(1, b, dtype=torch.int32)
            ep = torch.export.export(embed, (ids,)).run_decompositions(decomp)
            converter.add_exported_program(ep, input_names=["input_ids"], output_names=["inputs_embeds"],
                                           entrypoint_name=tokens_fn(b))
            x = torch.zeros(1, HIDDEN, 1, b, dtype=torch.float16)
            mask = torch.zeros(1, b, dtype=torch.float16)
            mask[0, :8] = 1
            ep = torch.export.export(enc, (x, mask)).run_decompositions(decomp)
            converter.add_exported_program(ep, input_names=["inputs_embeds", "attention_mask"],
                                           output_names=["embedding"], entrypoint_name=text_fn(b))
            print(f"[export] traced bucket {b} in {time.time() - t:.1f}s")
    return converter.to_coreai()


async def check(program, asset_path: Path, buckets: list[int]) -> None:
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
    from coreai_torch.debugging.compute_plan import ComputePlan

    opts = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())
    plan = await ComputePlan.from_program(program, opts)
    plan.write_summary(program)
    from coreai_torch.debugging.utils import _walk_operations

    reasons: dict[str, set[str]] = {}
    for op in _walk_operations(program):
        for msg in plan.get_validation_messages(op):
            reasons.setdefault(str(msg)[:240], set()).add(op.name)
    print("[check] Neural Engine validation messages:")
    for msg, ops in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"   {sorted(ops)[:6]} -> {msg}")

    ref = np.load(FIX / "text_reference.npz")
    ids, lengths, emb32 = ref["input_ids"], ref["lengths"], ref["embeddings"]
    t = time.time()
    model = await AIModel.load(asset_path, opts)
    print(f"[check] loaded + specialized in {time.time() - t:.1f}s; functions {model.function_names}")
    tok_fns = {b: model.load_function(tokens_fn(b)) for b in buckets}
    fns = {b: model.load_function(text_fn(b)) for b in buckets}
    cos = []
    for i, n in enumerate(lengths):
        n = int(n)
        b = next((b for b in buckets if n <= b), None)
        if b is None:
            continue
        x = np.zeros((1, b), dtype=np.int32)
        x[0, :n] = ids[i, :n]
        m = np.zeros((1, b), dtype=np.float16)
        m[0, :n] = 1
        t = time.time()
        emb = await tok_fns[b]({"input_ids": NDArray(x)})
        t1 = time.time()
        out = await fns[b]({"inputs_embeds": emb["inputs_embeds"], "attention_mask": NDArray(m)})
        dt, dt_tok = time.time() - t1, t1 - t
        e = np.asarray(out["embedding"].numpy(), dtype=np.float32)[0]
        e = e / np.linalg.norm(e)
        cos.append(float(np.dot(e, emb32[i])))
        print(f"[check] len {n:5d} bucket {b:5d} cos {cos[-1]:.6f}  tokens {dt_tok * 1000:.1f} ms  encoder {dt * 1000:.1f} ms")
    print(f"[check] min cos {min(cos):.6f}  mean {np.mean(cos):.6f}  below 0.999: {sum(c < 0.999 for c in cos)}/{len(cos)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="128,512,2048")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "exports")
    ap.add_argument("--check", action="store_true", help="placement summary and parity (needs USE_OS_COREAI=1)")
    ap.add_argument("--debug-info", action="store_true")
    args = ap.parse_args()
    buckets = sorted(int(b) for b in args.buckets.split(","))

    embed, encoders = build_models(buckets)
    program = convert(embed, encoders, debug=args.debug_info or args.check)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"embeddinggemma2_text_fp16_{'_'.join(map(str, buckets))}.aimodel"
    program.save_asset(path)
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
    print(f"[export] saved {path} ({size / 2**20:.0f} MiB)")
    if args.check:
        asyncio.run(check(program, path, buckets))


if __name__ == "__main__":
    main()
