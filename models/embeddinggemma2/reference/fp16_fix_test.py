"""Test fp16 rewrites for EmbeddingGemma 2 text in PyTorch (MPS) against the fp32 reference.

Variants (all fp16 weights and activations on MPS):
    A  hf_naive      HF code as shipped: RMSNorm upcasts to fp32, pooling sums then divides
    B  strict        RMSNorm in fp16, pooling sums then divides, normalize in fp16
    C  strict_fixed  RMSNorm in fp16 with per-norm prescale, pooling divides first, normalize prescaled
Plus bf16 HF as shipped, for comparison.

Prescale per norm: S = smallest power of 2 with max_in / S <= 8, from fixtures/text_overflow.json.
RMSNorm is scale-invariant, so x/S gives the same output (eps becomes eps/S^2).

    uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 \
        --with "sentence-transformers>=6.1" python fp16_fix_test.py
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as hf
from sentence_transformers import SentenceTransformer

from make_text_reference import MAX_TOKENS, MODEL_ID, REVISION, compare

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
TARGET_MAX_AFTER_SCALE = 8.0

MODE = {"norm": "hf"}  # "hf" | "strict" | "fixed"
_orig_forward = hf.EmbeddingGemma2RMSNorm.forward


def _rmsnorm_forward(self, x):
    if MODE["norm"] == "hf":
        return _orig_forward(self, x)
    s = getattr(self, "_fx_scale", 1.0) if MODE["norm"] == "fixed" else 1.0
    xs = x / s if s != 1.0 else x
    ms = (xs * xs).mean(-1, keepdim=True) + self.eps / (s * s)
    out = xs * torch.rsqrt(ms)
    return out * self.weight if self.with_scale else out


hf.EmbeddingGemma2RMSNorm.forward = _rmsnorm_forward


def pool_normalize(tok: torch.Tensor, mask: torch.Tensor, fixed: bool) -> torch.Tensor:
    """Masked mean pool + L2 normalize, in tok's dtype."""
    m = mask.to(tok.dtype).unsqueeze(-1)
    n = m.sum(1)
    if fixed:
        pooled = (tok * (m / n.unsqueeze(1))).sum(1)  # divide first: no large partial sums
        s = pooled.abs().amax(-1, keepdim=True).clamp_min(1e-4)  # prescale before squaring
        ps = pooled / s
        return ps * torch.rsqrt((ps * ps).sum(-1, keepdim=True))
    pooled = (tok * m).sum(1) / n
    return pooled * torch.rsqrt((pooled * pooled).sum(-1, keepdim=True))


def main() -> None:
    ref = np.load(FIX / "text_reference.npz")
    stats = json.loads((FIX / "text_overflow.json").read_text())["modules"]
    emb32 = ref["embeddings"]
    prompts, texts = list(ref["prompts"]), list(ref["texts"])

    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="mps",
                                model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"vision_config": None, "audio_config": None})
    model.eval()
    model.max_seq_length = MAX_TOKENS
    prefixed = [(model.prompts.get(p, "") if p else "") + t for p, t in zip(prompts, texts)]

    lm = model[0].model.language_model
    scales = {}
    for name, mod in lm.named_modules():
        if isinstance(mod, hf.EmbeddingGemma2RMSNorm):
            mx = stats.get(name, {}).get("in", 0.0)
            s = 2.0 ** max(0, math.ceil(math.log2(mx / TARGET_MAX_AFTER_SCALE))) if mx > 0 else 1.0
            mod._fx_scale = s
            scales[name] = s

    # Watch for non-finite values inside the fp16 runs (catches overflow we did not hook in fp32).
    nonfinite: dict[str, set] = {}
    current = {"variant": ""}

    def watch(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                nonfinite.setdefault(current["variant"], set()).add(name)
        return hook

    for name, mod in lm.named_modules():
        if name:
            mod.register_forward_hook(watch(name))

    def run(variant: str, dtype, norm_mode: str, fixed_pool: bool, st_pool: bool) -> np.ndarray:
        MODE["norm"] = norm_mode
        current["variant"] = variant
        model.to(dtype)
        out = []
        for text in prefixed:  # one at a time: no padding effects, max 2048 tokens
            feats = model.tokenize([text])
            feats = {k: v.to("mps") if torch.is_tensor(v) else v for k, v in feats.items()}
            with torch.no_grad():
                if st_pool:
                    e = model(feats)["sentence_embedding"]
                else:
                    tok = model[0](feats)["token_embeddings"]
                    e = pool_normalize(tok, feats["attention_mask"], fixed_pool)
            out.append(np.nan_to_num(e.float().cpu().numpy()[0], nan=0.0, posinf=0.0, neginf=0.0))
            torch.mps.empty_cache()
        return np.stack(out)

    variants = {
        "bf16_hf": run("bf16_hf", torch.bfloat16, "hf", False, True),
        "A_fp16_hf_naive": run("A_fp16_hf_naive", torch.float16, "hf", False, True),
        "B_fp16_strict": run("B_fp16_strict", torch.float16, "strict", False, False),
        "C_fp16_strict_fixed": run("C_fp16_strict_fixed", torch.float16, "fixed", True, False),
    }

    lengths = [len(model.tokenize([t])["input_ids"][0]) for t in prefixed]
    print("lengths:", lengths)
    summary = {}
    for k, e in variants.items():
        per = (emb32 * e).sum(-1)
        summary[k] = compare(emb32, e) | {"per_text": [round(float(c), 5) for c in per]}
        print(f"\n{k}: {json.dumps({kk: v for kk, v in summary[k].items() if kk != 'per_text'})}")
        print("  per text:", " ".join(f"{c:.4f}" for c in per))
        if k in nonfinite:
            print("  non-finite in:", sorted(nonfinite[k])[:12], "..." if len(nonfinite[k]) > 12 else "")
    print("\nprescale used (norms with S>1):", {n: s for n, s in scales.items() if s > 1})
    (FIX / "text_fp16_fix.json").write_text(json.dumps(
        {"summary": summary, "scales": scales, "nonfinite": {k: sorted(v) for k, v in nonfinite.items()},
         "lengths": lengths}, indent=1))


if __name__ == "__main__":
    main()
