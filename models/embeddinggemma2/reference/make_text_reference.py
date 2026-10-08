"""Build the fp32 text reference for EmbeddingGemma 2 and map fp16 overflow risk.

Runs in an isolated env (the repo pins an older transformers):

    uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 \
        --with "sentence-transformers>=6.1" python make_text_reference.py [--corpus-dir DIR]

Passes:
    fp32 on MPS, with hooks  -> reference embeddings + per-module max |activation|
    bf16 and fp16 on MPS     -> naive low-precision cosine and top-10 agreement vs fp32

Writes:
    fixtures/text_reference.npz   public test texts only: token ids, lengths, fp32 embeddings
    fixtures/text_overflow.json   aggregate numbers only

--corpus-dir adds local passages (PDF/md/txt) to the statistics. Their text, names and
embeddings are never written anywhere.
"""

import argparse
import inspect
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
FP16_MAX = 65504.0
TOKENS_PER_BATCH = 4096  # batch by total padded tokens, so long inputs don't blow up memory
MAX_TOKENS = 2048  # cap per input; the 8192 bucket gets its own test later

HERE = Path(__file__).resolve().parent


def public_texts() -> list[tuple[str, str]]:
    """(prompt_name, text) pairs safe to commit: queries, documents, code, multilingual, long."""
    import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as m

    prose = (
        "The northern lights are caused by charged particles from the sun striking atoms in "
        "Earth's upper atmosphere. Oxygen emits green and red light, nitrogen emits blue and purple. "
    )
    return [
        ("SearchQuery", "What causes the northern lights?"),
        ("Document", "The northern lights are caused by charged particles from the sun."),
        ("Document", "Photosynthesis converts light energy into chemical energy in plants."),
        ("SearchQuery", "how do plants make food"),
        ("CodeRetrieval", "rotary position embedding applied to query and key"),
        ("Document", inspect.getsource(m.EmbeddingGemma2Attention)),
        ("SentenceSimilarity", "A man is playing a guitar on stage."),
        ("SentenceSimilarity", "Someone performs music with a guitar in front of an audience."),
        ("Classification", "This product broke after two days. Very disappointed."),
        ("Clustering", "Les aurores boréales sont causées par des particules chargées du soleil."),
        ("SearchQuery", "日本の首都はどこですか"),
        ("Document", "東京は日本の首都です。"),
        ("", "a"),
        ("", ""),
        ("Document", prose * 12),
        ("Document", prose * 50),
        ("Document", inspect.getsource(m)[:8000]),
    ]


def corpus_texts(root: Path, n_short: int, n_long: int, seed: int = 0) -> list[tuple[str, str]]:
    """Passages from local PDF/md/txt files. Kept in memory only."""
    docs = []
    for p in sorted(root.rglob("*")):
        if p.name.startswith(".") or not p.is_file():
            continue
        if p.suffix.lower() == ".pdf":
            r = subprocess.run(["pdftotext", "-q", str(p), "-"], capture_output=True, text=True)
            text = r.stdout
        elif p.suffix.lower() in (".md", ".txt"):
            text = p.read_text(errors="ignore")
        else:
            continue
        text = " ".join(text.split())
        if len(text) > 200:
            docs.append(text)
    rng = random.Random(seed)
    short, long_ = [], []
    for text in docs:
        for start in range(0, len(text), 1500):
            short.append(text[start : start + 1500])
        for start in range(0, len(text) - 8000, 8000):
            long_.append(text[start : start + 8000])
    rng.shuffle(short)
    rng.shuffle(long_)
    print(f"corpus: {len(docs)} files, {len(short)} short and {len(long_)} long passages available")
    return [("Document", t) for t in short[:n_short] + long_[:n_long]]


def tokenize_all(model, texts):
    prefixed = [(model.prompts.get(p, "") if p else "") + t for p, t in texts]
    return prefixed, [model.tokenize([t])["input_ids"][0] for t in prefixed]


def encode(model, prefixed, device, hooks_into=None) -> np.ndarray:
    """Batched by length. Returns L2-normalized sentence embeddings in input order."""
    order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))
    batches, cur = [], []
    for i in order:
        n_tok = min(MAX_TOKENS, len(model.tokenize([prefixed[i]])["input_ids"][0]))
        if cur and n_tok * (len(cur) + 1) > TOKENS_PER_BATCH:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    out = [None] * len(prefixed)
    for idx in batches:
        feats = model.tokenize([prefixed[i] for i in idx])
        feats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feats.items()}
        with torch.no_grad():
            emb = model(feats)["sentence_embedding"].float().cpu().numpy()
        if device == "mps":
            torch.mps.empty_cache()
        for j, i in enumerate(idx):
            out[i] = emb[j]
    return np.nan_to_num(np.stack(out), nan=0.0).astype(np.float32)


def attach_stats_hooks(model, stats):
    """Max |output| per module, max |input| per RMSNorm. Padding rows count: buckets compute them too."""
    handles = []
    lm = model[0].model.language_model

    def record(name, key, t):
        if isinstance(t, (tuple, list)):
            t = t[0]
        if torch.is_tensor(t) and t.is_floating_point():
            s = stats.setdefault(name, {})
            s[key] = max(s.get(key, 0.0), float(t.detach().abs().max()))

    for name, mod in lm.named_modules():
        if not name:
            continue

        def hook(mod, inp, out, name=name):
            record(name, "out", out)
            if "RMSNorm" in type(mod).__name__ and inp:
                record(name, "in", inp[0])

        handles.append(mod.register_forward_hook(hook))
    return handles


def compare(ref: np.ndarray, other: np.ndarray, k: int = 10) -> dict:
    cos = (ref * other).sum(-1)
    out = {"min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
           "n_below_0.999": int((cos < 0.999).sum()), "n_nan_or_zero": int((np.abs(other).sum(-1) == 0).sum())}
    if len(ref) > k:
        s_ref, s_oth = ref @ ref.T, other @ other.T
        np.fill_diagonal(s_ref, -np.inf)
        np.fill_diagonal(s_oth, -np.inf)
        top_ref = np.argsort(-s_ref, axis=1)[:, :k]
        top_oth = np.argsort(-s_oth, axis=1)[:, :k]
        out[f"top{k}_overlap"] = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(top_ref, top_oth)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "fixtures")
    ap.add_argument("--corpus-dir", type=Path, default=None)
    ap.add_argument("--n-short", type=int, default=160)
    ap.add_argument("--n-long", type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="mps",
                                model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"vision_config": None, "audio_config": None})
    model.eval()
    model.max_seq_length = MAX_TOKENS

    groups = {"public": public_texts()}
    if args.corpus_dir:
        groups["corpus"] = corpus_texts(args.corpus_dir, args.n_short, args.n_long)

    stats: dict = {}
    results: dict = {}
    embs32: dict = {}
    for gname, texts in groups.items():
        prefixed, ids = tokenize_all(model, texts)
        lengths = [len(i) for i in ids]
        handles = attach_stats_hooks(model, stats)
        embs32[gname] = encode(model, prefixed, "mps")
        for h in handles:
            h.remove()
        results[gname] = {"n": len(texts), "max_len": max(lengths), "mean_len": float(np.mean(lengths))}
        if gname == "public":
            padded = np.zeros((len(ids), max(lengths)), dtype=np.int32)
            for row, i in zip(padded, ids):
                row[: len(i)] = i.numpy()
            np.savez_compressed(args.out_dir / "text_reference.npz", embeddings=embs32[gname],
                                input_ids=padded, lengths=np.array(lengths, dtype=np.int32),
                                prompts=np.array([p for p, _ in texts]), texts=np.array([t for _, t in texts]))
        groups[gname] = prefixed

    for label, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        model.to(dtype)
        for gname, prefixed in groups.items():
            results[gname][label] = compare(embs32[gname], encode(model, prefixed, "mps"))

    report = {"model": MODEL_ID, "revision": REVISION, "fp16_max": FP16_MAX,
              "groups": results, "modules": stats}
    (args.out_dir / "text_overflow.json").write_text(json.dumps(report, indent=1))

    print(json.dumps(results, indent=1))
    top = sorted(((max(s.values()), n, s) for n, s in stats.items()), reverse=True)[:25]
    for v, n, s in top:
        print(f"{v:12.1f}  {n}  {s}")


if __name__ == "__main__":
    main()
