"""Per-RMSNorm fp16 prescales from the measured overflow maps (text and image inputs combined).

Scale S = smallest power of 2 with (max |input| / S) <= TARGET, so x^2 stays far below the fp16
limit (65504). RMSNorm is scale-invariant, so dividing by S does not change its output.

Writes fixtures/fp16_scales.json: {"text": {norm name: S}, "vision": {norm name: S}}.
Norm names match the HF module paths under language_model / vision_tower / embed_vision.
"""

import json
import math
from pathlib import Path

FIX = Path(__file__).resolve().parent / "fixtures"
TARGET = 8.0
POOLER_SCALE = 768**0.5  # the port drops the pooler's x sqrt(768); the next norm sees max / this


def scale_for(max_in: float) -> float:
    return 2.0 ** max(0, math.ceil(math.log2(max_in / TARGET))) if max_in > 0 else 1.0


def main() -> None:
    text = json.loads((FIX / "text_overflow.json").read_text())["modules"]
    image = json.loads((FIX / "image_overflow.json").read_text())["modules"]

    text_max: dict[str, float] = {n: s["in"] for n, s in text.items() if "in" in s}
    for n, s in image.items():
        if n.startswith("text.") and "in" in s:
            key = n[len("text."):]
            text_max[key] = max(text_max.get(key, 0.0), s["in"])

    vision_max: dict[str, float] = {}
    for n, s in image.items():
        if "in" not in s:
            continue
        if n.startswith("vision."):
            vision_max[n[len("vision."):]] = s["in"]
        elif n == "embed_vision.embedding_pre_projection_norm":
            vision_max["embedding_pre_projection_norm"] = s["in"] / POOLER_SCALE

    out = {
        "target_max_after_scale": TARGET,
        "text": {n: scale_for(v) for n, v in sorted(text_max.items())},
        "vision": {n: scale_for(v) for n, v in sorted(vision_max.items())},
        "max_input": {"text": text_max, "vision": vision_max},
    }
    (FIX / "fp16_scales.json").write_text(json.dumps(out, indent=1))
    for group in ("text", "vision"):
        big = {n: s for n, s in out[group].items() if s > 1}
        print(f"{group}: {len(out[group])} norms, {len(big)} prescaled, max S {max(out[group].values()):.0f}")


if __name__ == "__main__":
    main()
