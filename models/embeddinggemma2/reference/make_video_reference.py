"""fp32 video reference for EmbeddingGemma 2 (synthetic clips), plus fp16 overflow stats on video input.

    uv run --no-project --python 3.11 --with transformers==5.19.0 --with torch==2.9.0 \
        --with "sentence-transformers>=6.1" --with pillow --with torchvision --with av \
        python make_video_reference.py

Writes:
    fixtures/videos/<stem>.mp4               synthetic clips (H.264, yuv420p), safe to commit
    fixtures/videos/<stem>/frame_XX.png       the frames HF samples (1 fps, max 32), decoded by PyAV
    fixtures/video_reference.json            per clip: caption, fps, frame count, sampled indices, soft
                                              tokens per frame, sequence length, fp32 768-d embedding,
                                              caption embedding
    fixtures/video_overflow.json             per-module max |activation| on video input
"""

import json
import math
import subprocess
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
FIX = Path(__file__).resolve().parent / "fixtures"
VID = FIX / "videos"

# stem, caption, seconds, fps, (width, height)
CLIPS = [
    ("red_ball_moving", "a red ball moving from left to right", 6, 10, (320, 240)),
    ("blue_square_growing", "a blue square growing larger", 4, 10, (320, 240)),
    ("green_triangle_rotating", "a green triangle spinning around", 5, 10, (320, 240)),
    ("checkerboard_flashing", "a black and white checkerboard flashing", 3, 10, (320, 240)),
    ("yellow_star_falling", "a yellow star falling down", 4, 10, (240, 360)),
    ("long_color_cycle", "a screen slowly changing colors through the rainbow", 45, 2, (320, 240)),
]


def frame(stem: str, t: float, size: tuple[int, int]) -> Image.Image:
    """Frame at time fraction t in [0, 1]."""
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    m = min(w, h)
    if stem == "red_ball_moving":
        r = m // 8
        cx = int(r + t * (w - 2 * r))
        d.ellipse([cx - r, h // 2 - r, cx + r, h // 2 + r], fill=(220, 20, 30))
    elif stem == "blue_square_growing":
        s = int(m * (0.1 + 0.35 * t))
        d.rectangle([w // 2 - s, h // 2 - s, w // 2 + s, h // 2 + s], fill=(20, 60, 220))
    elif stem == "green_triangle_rotating":
        a0 = t * 2 * math.pi
        pts = [(w / 2 + m / 3 * math.cos(a0 + k * 2 * math.pi / 3), h / 2 + m / 3 * math.sin(a0 + k * 2 * math.pi / 3)) for k in range(3)]
        d.polygon(pts, fill=(30, 170, 60))
    elif stem == "checkerboard_flashing":
        s = m // 6
        flip = int(t * 6) % 2
        for i in range(w // s + 1):
            for j in range(h // s + 1):
                if (i + j + flip) % 2:
                    d.rectangle([i * s, j * s, (i + 1) * s - 1, (j + 1) * s - 1], fill="black")
    elif stem == "yellow_star_falling":
        cy = int(m / 6 + t * (h - m / 3))
        pts = []
        for k in range(10):
            r = m / 5 if k % 2 == 0 else m / 12
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((w / 2 + r * math.cos(a), cy + r * math.sin(a)))
        d.polygon(pts, fill=(245, 200, 20))
    elif stem == "long_color_cycle":
        hue = t
        k = lambda n: (n + hue * 6) % 6  # noqa: E731
        f = lambda n: 1 - max(0, min(k(n), 4 - k(n), 1))  # noqa: E731
        img.paste(tuple(int(255 * f(n)) for n in (5, 3, 1)), [0, 0, w, h])
    return img


def write_clip(stem: str, seconds: int, fps: int, size: tuple[int, int]) -> Path:
    path = VID / f"{stem}.mp4"
    n = seconds * fps
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-movflags", "+faststart",
         str(path)], stdin=subprocess.PIPE)
    for i in range(n):
        proc.stdin.write(np.asarray(frame(stem, i / max(n - 1, 1), size)).tobytes())
    proc.stdin.close()
    proc.wait()
    return path


def decode(path: Path) -> tuple[list[np.ndarray], float, float]:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        fps = float(s.average_rate)
    return frames, fps, len(frames) / fps


def sample_indices(total: int, fps: float, duration: float, target_fps: float = 1, max_frames: int = 32) -> np.ndarray:
    """EmbeddingGemma2VideoProcessor.sample_frames with fps=1, max_frames=32, uniform overflow."""
    step = fps / target_fps
    n = max(1, int(duration * target_fps))
    idx = np.array([min(total - 1, int(i * step)) for i in range(n)], dtype=int)
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames, dtype=int)]
    return idx


def main() -> None:
    VID.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu",
                                model_kwargs={"torch_dtype": torch.float32}, config_kwargs={"audio_config": None})
    model.eval()
    hf_model, processor = model[0].model, model[0].processor

    stats: dict[str, dict[str, float]] = {}

    def record(name, key, t):
        if isinstance(t, (tuple, list)):
            t = t[0]
        if hasattr(t, "last_hidden_state"):
            t = t.last_hidden_state
        if torch.is_tensor(t) and t.is_floating_point() and t.numel():
            s = stats.setdefault(name, {})
            s[key] = max(s.get(key, 0.0), float(t.detach().abs().max()))

    for prefix, root in (("vision.", hf_model.vision_tower), ("text.", hf_model.language_model)):
        for name, mod in root.named_modules():
            def hook(mod, inp, out, name=prefix + name):
                record(name, "out", out)
                if "RMSNorm" in type(mod).__name__ and inp:
                    record(name, "in", inp[0])
            mod.register_forward_hook(hook)

    items = []
    for stem, caption, seconds, fps, size in CLIPS:
        path = write_clip(stem, seconds, fps, size)
        frames, real_fps, duration = decode(path)
        idx = sample_indices(len(frames), real_fps, duration)
        fdir = VID / stem
        fdir.mkdir(exist_ok=True)
        for old in fdir.glob("*.png"):
            old.unlink()
        sampled = [frames[i] for i in idx]
        for k, f in enumerate(sampled):
            Image.fromarray(f).save(fdir / f"frame_{k:02d}.png")

        # Path 1: HF's own decode + sampling from the file. Path 2: our sampled frames, sampling off.
        inputs = processor(text=["<|video|>"], videos=[str(path)], return_tensors="pt")
        inputs2 = processor(text=["<|video|>"], videos=[np.stack(sampled)], do_sample_frames=False, return_tensors="pt")
        vids = []
        for inp in (inputs, inputs2):
            with torch.no_grad():
                out = hf_model(**inp)
            m = inp["attention_mask"][0].float()
            v = (out.last_hidden_state[0] * m[:, None]).sum(0) / m.sum()
            vids.append((v / v.norm()).numpy())
        n_video_tok = int((inputs["input_ids"][0] == hf_model.config.video_token_id).sum())
        n_frames = int(inputs["num_frames_per_video"][0])
        cap = model.encode([caption], convert_to_numpy=True, normalize_embeddings=True)[0]
        print(f"{stem:24s} {len(frames)} frames @ {real_fps:g} fps -> sampled {len(idx)} (HF {n_frames}); "
              f"soft tokens/frame {n_video_tok // n_frames}; seq {int(inputs['attention_mask'].sum())}; "
              f"cos(HF decode, our frames) {float(np.dot(vids[0], vids[1])):.6f}; cos(caption) {float(np.dot(cap, vids[0])):.3f}")
        items.append({
            "stem": stem, "caption": caption, "fps": real_fps, "total_frames": len(frames),
            "sampled_indices": idx.tolist(), "num_frames": n_frames, "soft_tokens_per_frame": n_video_tok // n_frames,
            "seq_len": int(inputs["attention_mask"].sum()),
            "embedding": [round(float(x), 7) for x in vids[1]],
            "embedding_hf_decode": [round(float(x), 7) for x in vids[0]],
            "caption_embedding": [round(float(x), 7) for x in cap],
        })

    emb = np.array([it["embedding"] for it in items])
    caps = np.array([it["caption_embedding"] for it in items])
    sim = caps @ emb.T
    print(f"text->video top-1: {(sim.argmax(1) == np.arange(len(items))).mean():.2f}")
    (FIX / "video_reference.json").write_text(json.dumps({"items": items, "caption_video_sim": sim.round(4).tolist()}))
    (FIX / "video_overflow.json").write_text(json.dumps({"modules": stats}, indent=1))
    for group in ("vision.", "text."):
        mods = {n: s for n, s in stats.items() if n.startswith(group)}
        norms = [s["in"] for s in mods.values() if "in" in s]
        print(f"{group} max {max(max(s.values()) for s in mods.values()):.1f}; norm inputs >256: "
              f"{sum(v > 256 for v in norms)}/{len(norms)}; max norm input {max(norms):.1f}")


if __name__ == "__main__":
    main()
