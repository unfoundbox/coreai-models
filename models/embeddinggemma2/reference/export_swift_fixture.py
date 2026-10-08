"""Write fixtures/text_reference.json (texts, prompts, token ids, fp32 embeddings) for the Swift tests."""

import json
from pathlib import Path

import numpy as np

FIX = Path(__file__).resolve().parent / "fixtures"

r = np.load(FIX / "text_reference.npz")
items = []
for i, n in enumerate(r["lengths"]):
    items.append({
        "prompt_name": str(r["prompts"][i]),
        "text": str(r["texts"][i]),
        "input_ids": r["input_ids"][i][: int(n)].tolist(),
        "embedding": [round(float(x), 7) for x in r["embeddings"][i]],
    })
(FIX / "text_reference.json").write_text(json.dumps({"max_tokens": 2048, "items": items}))
print(f"wrote {len(items)} items")
