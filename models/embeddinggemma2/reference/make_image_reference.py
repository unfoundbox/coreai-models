"""fp32 image reference for EmbeddingGemma 2, vision fp16 overflow map, and text<->image checks.

    uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 \
        --with "sentence-transformers>=6.1" --with pillow --with torchvision python make_image_reference.py

Images: synthetic drawings generated here (saved to fixtures/images/, safe to commit) plus, when
--sample-dir is given, extra local photos used only for statistics (never written anywhere).

Writes:
    fixtures/images/*.png            synthetic test images
    fixtures/image_reference.npz     per image: size, soft-token count, fp32 768-d embedding,
                                     fp32 vision soft tokens (N x 512); text query embeddings
    fixtures/image_overflow.json     per-module max |activation| (vision tower, embedder, text
                                     backbone on image input) and text<->image similarity table
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"

# (file stem, caption used as a text query, size)
SYNTHETIC = [
    ("red_circle", "a red circle", (320, 320)),
    ("blue_square", "a blue square", (320, 320)),
    ("green_triangle", "a green triangle", (384, 256)),
    ("stop_sign", "a red stop sign with the word STOP", (300, 300)),
    ("checkerboard", "a black and white checkerboard pattern", (256, 256)),
    ("sunset_gradient", "an orange and purple sunset sky gradient", (480, 270)),
    ("yellow_star_tall", "a yellow star", (240, 480)),
]


def draw(stem: str, size: tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    m = min(w, h)
    cx, cy = w // 2, h // 2
    if stem == "red_circle":
        d.ellipse([cx - m // 3, cy - m // 3, cx + m // 3, cy + m // 3], fill=(220, 20, 30))
    elif stem == "blue_square":
        d.rectangle([cx - m // 3, cy - m // 3, cx + m // 3, cy + m // 3], fill=(20, 60, 220))
    elif stem == "green_triangle":
        d.polygon([(cx, cy - m // 3), (cx - m // 3, cy + m // 3), (cx + m // 3, cy + m // 3)], fill=(30, 170, 60))
    elif stem == "stop_sign":
        r = m // 3
        pts = [(cx + r * np.cos(np.pi / 8 + k * np.pi / 4), cy + r * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)]
        d.polygon(pts, fill=(200, 10, 20))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", m // 7)
        except OSError:
            font = ImageFont.load_default()
        d.text((cx, cy), "STOP", fill="white", font=font, anchor="mm")
    elif stem == "checkerboard":
        s = m // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2:
                    d.rectangle([i * s, j * s, (i + 1) * s - 1, (j + 1) * s - 1], fill="black")
    elif stem == "sunset_gradient":
        top, bottom = np.array([90, 30, 140]), np.array([250, 140, 30])
        for y in range(h):
            c = tuple(int(v) for v in top + (bottom - top) * y / (h - 1))
            d.line([(0, y), (w, y)], fill=c)
    elif stem == "yellow_star_tall":
        r1, r2 = m // 2.5, m // 6
        pts = []
        for k in range(10):
            r = r1 if k % 2 == 0 else r2
            a = -np.pi / 2 + k * np.pi / 5
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        d.polygon(pts, fill=(245, 200, 20))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=None, help="extra local images for statistics only")
    args = ap.parse_args()
    (FIX / "images").mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu",
                                model_kwargs={"torch_dtype": torch.float32},
                                config_kwargs={"audio_config": None})
    model.eval()
    st = model[0]
    hf_model = st.model
    processor = st.processor

    images: list[tuple[str, Image.Image, bool]] = []
    for stem, _, size in SYNTHETIC:
        img = draw(stem, size)
        img.save(FIX / "images" / f"{stem}.png")
        images.append((stem, img, True))
    if args.sample_dir:
        for p in sorted(args.sample_dir.glob("*.jp*g")):
            images.append((p.stem, Image.open(p).convert("RGB"), False))

    # Overflow hooks on vision tower, vision embedder and text backbone.
    stats: dict[str, dict[str, float]] = {}

    def record(name, key, t):
        if isinstance(t, (tuple, list)):
            t = t[0]
        if hasattr(t, "last_hidden_state"):
            t = t.last_hidden_state
        if torch.is_tensor(t) and t.is_floating_point() and t.numel():
            s = stats.setdefault(name, {})
            s[key] = max(s.get(key, 0.0), float(t.detach().abs().max()))

    for prefix, root in (("vision.", hf_model.vision_tower), ("embed_vision.", hf_model.embed_vision),
                         ("text.", hf_model.language_model)):
        for name, mod in root.named_modules():
            def hook(mod, inp, out, name=prefix + name):
                record(name, "out", out)
                if "RMSNorm" in type(mod).__name__ and inp:
                    record(name, "in", inp[0])
            mod.register_forward_hook(hook)

    emb, soft, sizes, n_soft, committed = [], [], [], [], []
    for stem, img, synthetic in images:
        inputs = processor(text=["<|image|>"], images=[[img]], return_tensors="pt")
        n_tok = int((inputs["input_ids"][0] == hf_model.config.image_token_id).sum())
        with torch.no_grad():
            out = hf_model(**inputs)
            feats = hf_model.get_image_features(inputs["pixel_values"], inputs["image_position_ids"]).pooler_output
        h = out.last_hidden_state[0]
        m = inputs["attention_mask"][0].float()
        v = (h * m[:, None]).sum(0) / m.sum()
        v = (v / v.norm()).numpy()
        # Cross-check against SentenceTransformers' own image path.
        v_st = model.encode([{"image": img}], convert_to_numpy=True, normalize_embeddings=True)[0]
        print(f"{stem:18s} size {img.size}  soft tokens {n_tok}  "
              f"seq {int(m.sum())}  cos(manual, ST) {float(np.dot(v, v_st)):.6f}")
        if synthetic:
            emb.append(v)
            soft.append(feats[0].numpy().astype(np.float16))
            sizes.append(img.size)
            n_soft.append(n_tok)
            committed.append(stem)

    captions = [c for _, c, _ in SYNTHETIC]
    q_none = model.encode(captions, convert_to_numpy=True, normalize_embeddings=True)
    q_search = model.encode(captions, prompt_name="SearchQuery", convert_to_numpy=True, normalize_embeddings=True)
    emb = np.stack(emb)
    sim_none, sim_search = q_none @ emb.T, q_search @ emb.T
    for label, sim in (("no prefix", sim_none), ("SearchQuery", sim_search)):
        top1 = (sim.argmax(1) == np.arange(len(captions))).mean()
        print(f"text->image top-1 ({label}): {top1:.2f}")

    np.savez_compressed(
        FIX / "image_reference.npz",
        stems=np.array(committed), sizes=np.array(sizes, dtype=np.int32), num_soft_tokens=np.array(n_soft, dtype=np.int32),
        embeddings=emb.astype(np.float32), soft_tokens=np.array(soft, dtype=object),
        captions=np.array(captions), query_embeddings=q_none.astype(np.float32),
        search_query_embeddings=q_search.astype(np.float32),
    )
    (FIX / "image_overflow.json").write_text(json.dumps({
        "model": MODEL_ID, "revision": REVISION, "images": [s for s, _, _ in images],
        "sim_no_prefix": sim_none.round(4).tolist(), "sim_search_query": sim_search.round(4).tolist(),
        "modules": stats}, indent=1))

    for group in ("vision.", "embed_vision.", "text."):
        mods = {n: s for n, s in stats.items() if n.startswith(group)}
        norms = [(s["in"], n) for n, s in mods.items() if "in" in s]
        print(f"\n{group}: max {max(max(s.values()) for s in mods.values()):.1f}; "
              f"norm inputs >256: {sum(v > 256 for v, _ in norms)}/{len(norms)}; >65504: {sum(v > 65504 for v, _ in norms)}")
        for v, n, s in sorted(((max(s.values()), n, s) for n, s in mods.items()), reverse=True)[:8]:
            print(f"   {v:10.1f}  {n}  {s}")


if __name__ == "__main__":
    main()
