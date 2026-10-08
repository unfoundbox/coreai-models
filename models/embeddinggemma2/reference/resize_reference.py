"""Reference for the Swift image preprocessing: HF-equivalent resize, checked against HF's own output.

Prototypes the PIL-style antialiased bicubic resampler (what torchvision's antialias=True follows)
in NumPy, compares it with HF Gemma4ImageProcessor pixels, and writes the HF-resized images as
lossless PNG fixtures for the Swift test (fixtures/images/<stem>__hf_<W>x<H>.png).

    uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 \
        --with pillow --with torchvision python resize_reference.py
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
FIX = Path(__file__).resolve().parent / "fixtures"
PATCH, POOL, MAX_PATCHES = 16, 3, 2520


def target_size(h: int, w: int) -> tuple[int, int]:
    """HF get_aspect_ratio_preserving_size for patch 16, pooling 3, 2520 patches."""
    factor = math.sqrt(MAX_PATCHES * PATCH**2 / (h * w))
    side = POOL * PATCH
    th, tw = int(math.floor(factor * h / side)) * side, int(math.floor(factor * w / side)) * side
    max_side = (MAX_PATCHES // POOL**2) * side
    if th == 0:
        th, tw = side, min(int(math.floor(w / h)) * side, max_side)
    elif tw == 0:
        tw, th = side, min(int(math.floor(h / w)) * side, max_side)
    return th, tw


def _cubic(x: np.ndarray, a: float = -0.5) -> np.ndarray:
    x = np.abs(x)
    return np.where(x < 1, ((a + 2) * x - (a + 3)) * x * x + 1,
                    np.where(x < 2, ((a * x - 5 * a) * x + 8 * a) * x - 4 * a, 0.0))


def _weights(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """PIL precompute_coeffs: (n_out, k) weights and start indices."""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = 2.0 * fs
    k = int(math.ceil(support)) * 2 + 1
    w = np.zeros((n_out, k))
    start = np.zeros(n_out, dtype=np.int64)
    for i in range(n_out):
        center = (i + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), n_in) - xmin
        x = (np.arange(xmax) + xmin - center + 0.5) / fs
        ww = _cubic(x)
        ww = ww / ww.sum()
        w[i, :xmax] = ww
        start[i] = xmin
    return w, start


def _pass(img: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """One separable pass along axis, uint8 in and out (rounded, clamped) like PIL."""
    src = np.moveaxis(img.astype(np.float64), axis, 0)
    w, start = _weights(src.shape[0], n_out)
    out = np.zeros((n_out,) + src.shape[1:])
    for i in range(n_out):
        k = np.count_nonzero(w[i]) and w.shape[1]
        seg = src[start[i] : start[i] + k]
        out[i] = np.tensordot(w[i, : seg.shape[0]], seg, axes=1)
    return np.moveaxis(np.clip(np.round(out), 0, 255).astype(np.uint8), 0, axis)


def resize(rgb: np.ndarray, th: int, tw: int) -> np.ndarray:
    """(H, W, 3) uint8 -> (th, tw, 3) uint8: horizontal pass, then vertical."""
    out = rgb
    if out.shape[1] != tw:
        out = _pass(out, tw, 1)
    if out.shape[0] != th:
        out = _pass(out, th, 0)
    return out


def hf_resized(processor, img: Image.Image) -> np.ndarray:
    """Rebuild HF's resized uint8 image from its patches and positions."""
    inputs = processor(text=["<|image|>"], images=[[img]], return_tensors="pt")
    pv, pos = inputs["pixel_values"][0].numpy(), inputs["image_position_ids"][0].numpy()
    real = pos[:, 0] >= 0
    gw, gh = pos[real, 0].max() + 1, pos[real, 1].max() + 1
    out = np.zeros((gh * PATCH, gw * PATCH, 3), dtype=np.float64)
    for p, (x, y) in zip(pv[real], pos[real]):
        out[y * PATCH : (y + 1) * PATCH, x * PATCH : (x + 1) * PATCH] = p.reshape(PATCH, PATCH, 3)
    return np.round(out * 255).astype(np.uint8)


def large_test_image() -> Image.Image:
    """A 1600x1200 synthetic photo-like image, so the resize downsamples (antialias path)."""
    w, h = 1600, 1200
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) % 97) * 2.6], -1).astype(np.uint8)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    for i in range(12):
        d.ellipse([100 + i * 110, 200 + (i % 3) * 250, 180 + i * 110, 280 + (i % 3) * 250], fill=(255, 255 - i * 20, i * 20))
    d.line([(0, 0), (w, h)], fill="black", width=5)
    return img


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    imgs = {p.stem: Image.open(p).convert("RGB") for p in sorted((FIX / "images").glob("*.png")) if "__" not in p.stem}
    big = large_test_image()
    big.save(FIX / "images" / "large_pattern.png")
    imgs["large_pattern"] = big
    for stem, img in imgs.items():
        ref = hf_resized(processor, img)
        th, tw = target_size(img.height, img.width)
        assert (th, tw) == ref.shape[:2], (stem, (th, tw), ref.shape)
        mine = resize(np.asarray(img), th, tw)
        diff = np.abs(mine.astype(int) - ref.astype(int))
        print(f"{stem:18s} {img.width}x{img.height} -> {tw}x{th}  max|diff| {diff.max()}  "
              f"mean {diff.mean():.4f}  pixels off {np.count_nonzero(diff)}/{diff.size}")
        Image.fromarray(ref).save(FIX / "images" / f"{stem}__hf_{tw}x{th}.png")


if __name__ == "__main__":
    main()
