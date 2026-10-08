"""EmbeddingGemma 2 text backbone, authored for the Neural Engine (fp16, BC1S).

Matches ``transformers`` 5.19 ``EmbeddingGemma2TextModel`` + SentenceTransformers mean pooling,
with two exact rewrites so every activation stays inside fp16 range:

  - RMSNorm prescale: each norm divides its input by a power-of-2 constant before squaring.
    RMSNorm is scale-invariant, so the output is unchanged (eps becomes eps / s^2). Without it,
    30 of 194 norm inputs exceed 256 and x^2 overflows fp16.
  - Pool before projecting: masked mean of the final 512-d states (weights mask / count, so no
    large partial sums), then the 512 -> 768 projection once. Projection is linear, so this
    equals projecting every token and pooling.

The encoder returns the pooled 768-d vector *without* L2 normalization; the caller normalizes in
fp32 (it re-normalizes after MRL truncation anyway).

Layout: hidden states are (1, C, 1, L). Every Linear is a 1x1 Conv2d. Attention is per head
with an additive key-major mask (1, L_k, 1, L_q). One encoder instance serves one static bucket length. The token lookup is a separate
module because int32 ids cannot run on the Neural Engine.
"""

import json
import math
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 512
INTERMEDIATE = 2048
NUM_LAYERS = 24
NUM_HEADS = 4
EMBED_DIM = 768
VOCAB = 262144
SLIDING_WINDOW = 512
FULL_LAYERS = (5, 11, 17, 23)
EPS = 1e-6
MASK_NEG = -20000.0  # two masks add to -40000, still finite in fp16
CHUNK = 512  # query chunk for long buckets
CHUNKED_ABOVE = 2048  # buckets longer than this use chunked attention


class BandedMask(NamedTuple):
    """Sliding attention in query chunks. Keys are padded by the window on both sides, so every
    chunk sees a static window of CHUNK + 2 * SLIDING_WINDOW keys."""

    key_mask_padded: torch.Tensor  # (1, L + 2W, 1, 1): padding and the added edges are MASK_NEG
    band: torch.Tensor  # (1, CHUNK + 2W, 1, CHUNK): MASK_NEG where |key - query| > W


class ChunkedMask(NamedTuple):
    """Full attention in query chunks against all keys."""

    key_mask: torch.Tensor  # (1, L, 1, 1)


def _layer_shape(i: int) -> tuple[int, int, float]:
    """(head_dim, num_kv_heads, rope_theta) for layer i."""
    if i in FULL_LAYERS:
        return 512, 1, 1_000_000.0
    return 256, 2, 10_000.0


def _conv(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=1, bias=False)


class RMSNormBC1S(nn.Module):
    """RMSNorm over channels (dim 1) with a constant input prescale."""

    def __init__(self, dim: int, scale: float = 1.0, with_weight: bool = True) -> None:
        super().__init__()
        self.with_weight = with_weight
        if with_weight:
            self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.register_buffer("inv_scale", torch.tensor(1.0 / scale))
        self.register_buffer("eps", torch.tensor(EPS / (scale * scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x * self.inv_scale
        ms = (xs * xs).mean(dim=1, keepdim=True)
        out = xs * torch.rsqrt(ms + self.eps)
        return out * self.weight if self.with_weight else out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[1] // 2
    return torch.cat((-x[:, half:], x[:, :half]), dim=1)


class Attention(nn.Module):
    def __init__(self, layer: int, scales: dict[str, float]) -> None:
        super().__init__()
        self.head_dim, self.num_kv, _ = _layer_shape(layer)
        hd, p = self.head_dim, f"layers.{layer}.self_attn."
        self.q_proj = _conv(HIDDEN, NUM_HEADS * hd)
        self.k_proj = _conv(HIDDEN, self.num_kv * hd)
        self.v_proj = _conv(HIDDEN, self.num_kv * hd)
        self.o_proj = _conv(NUM_HEADS * hd, HIDDEN)
        self.q_norm = RMSNormBC1S(hd, scales.get(p + "q_norm", 1.0))
        self.k_norm = RMSNormBC1S(hd, scales.get(p + "k_norm", 1.0))
        self.v_norm = RMSNormBC1S(hd, scales.get(p + "v_norm", 1.0), with_weight=False)

    def forward(self, x, cos, sin, mask):
        hd = self.head_dim
        group = NUM_HEADS // self.num_kv
        qs = self.q_proj(x).split(hd, dim=1)
        ks = self.k_proj(x).split(hd, dim=1)
        vs = self.v_proj(x).split(hd, dim=1)

        keys, values = [], []
        for k, v in zip(ks, vs):
            k = self.k_norm(k)
            k = k * cos + _rotate_half(k) * sin
            keys.append(k.permute(0, 3, 2, 1))  # (1, L_k, 1, hd)
            values.append(self.v_norm(v).permute(0, 3, 2, 1))

        if isinstance(mask, BandedMask):
            w = SLIDING_WINDOW
            pad = keys[0].new_zeros(1, w, 1, hd)
            keys = [torch.cat([pad, k, pad], dim=1) for k in keys]
            values = [torch.cat([pad, v, pad], dim=1) for v in values]

        outs = []
        for h, q in enumerate(qs):
            q = self.q_norm(q)
            q = q * cos + _rotate_half(q) * sin  # (1, hd, 1, L_q)
            kv = h // group
            if isinstance(mask, BandedMask):
                span = CHUNK + 2 * SLIDING_WINDOW
                parts = []
                for s in range(0, q.shape[3], CHUNK):
                    m = mask.band + mask.key_mask_padded[:, s : s + span]
                    parts.append(_attend(q[..., s : s + CHUNK], keys[kv][:, s : s + span], values[kv][:, s : s + span], m))
                outs.append(torch.cat(parts, dim=3))
            elif isinstance(mask, ChunkedMask):
                parts = [_attend(q[..., s : s + CHUNK], keys[kv], values[kv], mask.key_mask)
                         for s in range(0, q.shape[3], CHUNK)]
                outs.append(torch.cat(parts, dim=3))
            else:
                outs.append(_attend(q, keys[kv], values[kv], mask))
        return self.o_proj(torch.cat(outs, dim=1))


def _attend(q, k, v, mask):
    """q (1, hd, 1, L_q); k, v (1, L_k, 1, hd); additive mask broadcastable to (1, L_k, 1, L_q)."""
    scores = torch.einsum("bchq,bkhc->bkhq", q, k)  # scaling is 1.0
    if mask is not None:
        scores = scores + mask
    return torch.einsum("bkhq,bkhc->bchq", scores.softmax(dim=1), v)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _conv(HIDDEN, INTERMEDIATE)
        self.up_proj = _conv(HIDDEN, INTERMEDIATE)
        self.down_proj = _conv(INTERMEDIATE, HIDDEN)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class PLEBlock(nn.Module):
    def __init__(self, layer: int, scales: dict[str, float]) -> None:
        super().__init__()
        self.per_layer_input_gate = _conv(HIDDEN, HIDDEN)
        self.per_layer_projection = _conv(HIDDEN, HIDDEN)
        self.post_per_layer_input_norm = RMSNormBC1S(
            HIDDEN, scales.get(f"layers.{layer}.ple_block.post_per_layer_input_norm", 1.0)
        )

    def forward(self, x, per_layer_input):
        h = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * per_layer_input
        return x + self.post_per_layer_input_norm(self.per_layer_projection(h))


class EncoderLayer(nn.Module):
    def __init__(self, layer: int, scales: dict[str, float]) -> None:
        super().__init__()
        p = f"layers.{layer}."
        self.is_full = layer in FULL_LAYERS
        self.self_attn = Attention(layer, scales)
        self.mlp = MLP()
        self.input_layernorm = RMSNormBC1S(HIDDEN, scales.get(p + "input_layernorm", 1.0))
        self.post_attention_layernorm = RMSNormBC1S(HIDDEN, scales.get(p + "post_attention_layernorm", 1.0))
        self.pre_feedforward_layernorm = RMSNormBC1S(HIDDEN, scales.get(p + "pre_feedforward_layernorm", 1.0))
        self.post_feedforward_layernorm = RMSNormBC1S(HIDDEN, scales.get(p + "post_feedforward_layernorm", 1.0))
        self.ple_block = PLEBlock(layer, scales)
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(self, x, per_layer_input, cos, sin, mask):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        x = self.ple_block(x, per_layer_input)
        return x * self.layer_scalar


class TokenEmbedding(nn.Module):
    """Token lookup, kept apart from the encoder: int32 ids cannot run on the Neural Engine."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(VOCAB, HIDDEN))  # sqrt(512) folded in at load

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight).transpose(1, 2).unsqueeze(2)  # (1, 512, 1, L)


class TextEncoder(nn.Module):
    """Static-length encoder: (inputs_embeds, attention_mask) -> unnormalized 768-d mean embedding."""

    def __init__(self, seq_len: int, scales: dict[str, float] | None = None) -> None:
        super().__init__()
        scales = scales or {}
        self.seq_len = seq_len
        self.per_layer_model_projection = _conv(HIDDEN, NUM_LAYERS * HIDDEN)  # 512**-0.5 folded in at load
        self.per_layer_projection_norm = RMSNormBC1S(HIDDEN, scales.get("ple.per_layer_projection_norm", 1.0))
        self.layers = nn.ModuleList(EncoderLayer(i, scales) for i in range(NUM_LAYERS))
        self.norm = RMSNormBC1S(HIDDEN, scales.get("norm", 1.0))
        self.embedding_projection = _conv(HIDDEN, EMBED_DIM)
        self.register_buffer("mask_neg", torch.tensor(MASK_NEG))

        pos = torch.arange(seq_len, dtype=torch.float32)
        for hd, theta in ((256, 10_000.0), (512, 1_000_000.0)):
            inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
            ang = torch.cat([pos[None, :] * inv[:, None]] * 2, dim=0)  # (hd, L)
            self.register_buffer(f"cos_{hd}", ang.cos()[None, :, None, :])
            self.register_buffer(f"sin_{hd}", ang.sin()[None, :, None, :])
        self.chunked = seq_len > CHUNKED_ABOVE
        if self.chunked:
            if seq_len % CHUNK:
                raise ValueError(f"chunked buckets must be multiples of {CHUNK}, got {seq_len}")
            k = torch.arange(CHUNK + 2 * SLIDING_WINDOW)[:, None] - SLIDING_WINDOW
            q = torch.arange(CHUNK)[None, :]
            band = ((k - q).abs() > SLIDING_WINDOW).float() * MASK_NEG
            self.register_buffer("band", band[None, :, None, :])  # (1, CHUNK + 2W, 1, CHUNK)
            self.register_buffer("edge_mask", torch.full((1, SLIDING_WINDOW, 1, 1), MASK_NEG))
            self.sliding_mask = None
        elif seq_len > SLIDING_WINDOW + 1:
            i = torch.arange(seq_len)
            far = (i[:, None] - i[None, :]).abs() > SLIDING_WINDOW
            self.register_buffer("sliding_mask", (far.float() * MASK_NEG)[None, :, None, :])
        else:
            self.sliding_mask = None  # every pair is within the window

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = inputs_embeds  # (1, 512, 1, L)
        ple = self.per_layer_model_projection(x)  # (1, 24*512, 1, L)
        ple = ple.reshape(NUM_LAYERS, HIDDEN, 1, self.seq_len)
        ple = self.per_layer_projection_norm(ple)

        key_mask = ((attention_mask - 1.0) * -self.mask_neg).reshape(1, self.seq_len, 1, 1)
        if self.chunked:
            full_mask = ChunkedMask(key_mask)
            sliding_mask = BandedMask(torch.cat([self.edge_mask, key_mask, self.edge_mask], dim=1), self.band)
        else:
            full_mask = key_mask
            sliding_mask = key_mask if self.sliding_mask is None else key_mask + self.sliding_mask

        for i, layer in enumerate(self.layers):
            if layer.is_full:
                cos, sin, mask = self.cos_512, self.sin_512, full_mask
            else:
                cos, sin, mask = self.cos_256, self.sin_256, sliding_mask
            x = layer(x, ple[i : i + 1], cos, sin, mask)

        x = self.norm(x)  # (1, 512, 1, L)
        weights = (attention_mask / attention_mask.sum()).reshape(1, 1, 1, self.seq_len)
        pooled = (x * weights).sum(dim=-1, keepdim=True)  # (1, 512, 1, 1), divide-first mean
        return self.embedding_projection(pooled).reshape(1, EMBED_DIM)


class EmbeddingGemma2Text(nn.Module):
    """Token embedding + encoder, for PyTorch parity checks. Export uses the two parts separately."""

    def __init__(self, seq_len: int, scales: dict[str, float] | None = None) -> None:
        super().__init__()
        self.embed = TokenEmbedding()
        self.encoder = TextEncoder(seq_len, scales)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(input_ids), attention_mask)


def load_scales(path: str | Path, part: str = "text") -> dict[str, float]:
    """Per-norm prescales: ``fixtures/fp16_scales.json`` (``part`` = "text" or "vision"), or the
    older text-only ``fixtures/text_fp16_fix.json``."""
    data = json.loads(Path(path).read_text())
    return data[part] if part in data else data["scales"]


def load_weights(model: EmbeddingGemma2Text, safetensors_path: str | Path) -> EmbeddingGemma2Text:
    """Load ``google/embeddinggemma-2`` text weights; fold constant scales in fp32."""
    from safetensors import safe_open

    sd = {}
    with safe_open(str(safetensors_path), "pt") as f:
        for key in f.keys():
            if key.startswith("language_model."):
                sd[key[len("language_model."):]] = f.get_tensor(key).float()

    own = {"embed.weight": sd.pop("embed_tokens.weight") * math.sqrt(HIDDEN)}
    own["encoder.per_layer_model_projection.weight"] = sd.pop("ple.per_layer_model_projection.weight") * HIDDEN**-0.5
    own["encoder.per_layer_projection_norm.weight"] = sd.pop("ple.per_layer_projection_norm.weight")
    for key, t in sd.items():
        own["encoder." + key] = t
    params = dict(model.named_parameters())
    missing = set(params) - set(own)
    unexpected = set(own) - set(params)
    if missing or unexpected:
        raise KeyError(f"weight mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    with torch.no_grad():
        for key, p in params.items():
            t = own[key]
            if p.dim() == 4 and t.dim() == 2:
                t = t[:, :, None, None]
            elif p.dim() == 4 and t.dim() == 1:
                t = t.reshape(1, -1, 1, 1)
            p.copy_(t.reshape(p.shape))
    return model
