"""EmbeddingGemma 2 vision encoder (Gemma 4 vision tower + embed_vision), authored for the Neural Engine.

Matches ``transformers`` 5.19 ``Gemma4VisionModel`` followed by ``EmbeddingGemma2MultimodalEmbedder``
for one image with a fixed patch budget P (2520 patches = 280 soft tokens at 3x3 pooling).

Split in two exported functions, like the text model:

  ``VisionPositions`` (GPU): int32 patch x/y ids -> fp16 position embeddings (table lookup).
  The caller computes RoPE cos/sin and the key mask from the patch coordinates (``rope_and_mask``):
  angles reach ~840 rad and need fp32, which the Neural Engine does not run.
  ``VisionEncoder`` (Neural Engine, fp16): pixel patches + the tensors above -> soft tokens (1, 512, 1, P/9).

Exact rewrites:
  - RMSNorm prescale per norm (see ``embedding_gemma2_text.RMSNormBC1S``).
  - The pooler's ``x sqrt(768)`` is dropped: it feeds a weightless RMSNorm, which cancels any
    constant factor (up to eps). HF keeps it in fp32 because it can exceed the fp16 range.
  - Pooling: the caller orders patches in 3x3 blocks (``block_order``), so each soft token is the
    mean of 9 consecutive patches. Attention does not depend on patch order: positions enter only
    through the position embeddings and RoPE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.ios.embedding_gemma2_text import MASK_NEG, RMSNormBC1S, _conv

HIDDEN = 768
INTERMEDIATE = 3072
NUM_LAYERS = 16
NUM_HEADS = 12
HEAD_DIM = 64
TEXT_HIDDEN = 512
PATCH = 16
POOL = 3
POSITION_TABLE = 10240
ROPE_THETA = 100.0


def _rope_inv_freq() -> torch.Tensor:
    spatial = HEAD_DIM // 2  # 32 channels per axis
    return 1.0 / ROPE_THETA ** (torch.arange(0, spatial, 2, dtype=torch.float32) / spatial)  # (16,)


class VisionPositions(nn.Module):
    """Position-table lookup: x_ids, y_ids int32 (1, P) -> pos_embeds fp16 (1, 768, 1, P).

    Lookup only, like ``TokenEmbedding``. Padding patches may use any index (0): they are masked as
    keys and their outputs are discarded, so their position embedding never reaches a real token.
    An earlier version also computed RoPE and the mask here; that graph made the Neural Engine
    compiler fail, which moved every function in the asset to the GPU.
    """

    def __init__(self) -> None:
        super().__init__()
        # HF's (2, 10240, 768) table, split into two 2D tables (same shape as the text lookup).
        self.x_table = nn.Parameter(torch.zeros(POSITION_TABLE, HIDDEN))
        self.y_table = nn.Parameter(torch.zeros(POSITION_TABLE, HIDDEN))

    def forward(self, x_ids: torch.Tensor, y_ids: torch.Tensor) -> torch.Tensor:
        emb = F.embedding(x_ids, self.x_table) + F.embedding(y_ids, self.y_table)  # (1, P, 768)
        return emb.transpose(1, 2).unsqueeze(2)


def rope_and_mask(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Caller-side helper. positions (P, 2) int (x, y), -1 = padding ->
    rope_cos, rope_sin (1, 64, 1, P) and patch_mask (1, P) (1 = real, 0 = padding), all fp16.
    Angles in fp32. The mask is passed as (1, P), not (1, P, 1, 1): a size-1 last axis on a
    Neural Engine input was misread and broke attention."""
    padding = positions[:, 0] < 0
    pos = positions.clamp(min=0).float()
    inv = _rope_inv_freq()
    ax, ay = pos[:, 0:1] * inv, pos[:, 1:2] * inv  # (P, 16)
    ang = torch.cat([ax, ax, ay, ay], dim=1)  # (P, 64): x half, y half
    cos = ang.cos().t()[None, :, None, :].half()
    sin = ang.sin().t()[None, :, None, :].half()
    patch_mask = (~padding).float().reshape(1, -1).half()
    return cos, sin, patch_mask


def _rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (1, 64, 1, P). Rotate-half separately inside the x half and the y half of the head."""
    parts = []
    for lo in (0, HEAD_DIM // 2):
        p = x[:, lo : lo + HEAD_DIM // 2]
        q = HEAD_DIM // 4
        rot = torch.cat((-p[:, q:], p[:, :q]), dim=1)
        parts.append(p * cos[:, lo : lo + HEAD_DIM // 2] + rot * sin[:, lo : lo + HEAD_DIM // 2])
    return torch.cat(parts, dim=1)


class VisionAttention(nn.Module):
    def __init__(self, layer: int, scales: dict[str, float]) -> None:
        super().__init__()
        p = f"encoder.layers.{layer}.self_attn."
        self.q_proj = _conv(HIDDEN, HIDDEN)
        self.k_proj = _conv(HIDDEN, HIDDEN)
        self.v_proj = _conv(HIDDEN, HIDDEN)
        self.o_proj = _conv(HIDDEN, HIDDEN)
        self.q_norm = RMSNormBC1S(HEAD_DIM, scales.get(p + "q_norm", 1.0))
        self.k_norm = RMSNormBC1S(HEAD_DIM, scales.get(p + "k_norm", 1.0))
        self.v_norm = RMSNormBC1S(HEAD_DIM, scales.get(p + "v_norm", 1.0), with_weight=False)

    def forward(self, x, cos, sin, mask):
        qs = self.q_proj(x).split(HEAD_DIM, dim=1)
        ks = self.k_proj(x).split(HEAD_DIM, dim=1)
        vs = self.v_proj(x).split(HEAD_DIM, dim=1)
        outs = []
        for q, k, v in zip(qs, ks, vs):
            q = _rope_2d(self.q_norm(q), cos, sin)
            k = _rope_2d(self.k_norm(k), cos, sin).permute(0, 3, 2, 1)  # (1, P_k, 1, 64)
            v = self.v_norm(v).permute(0, 3, 2, 1)
            scores = torch.einsum("bchq,bkhc->bkhq", q, k) + mask  # scaling is 1.0
            outs.append(torch.einsum("bkhq,bkhc->bchq", scores.softmax(dim=1), v))
        return self.o_proj(torch.cat(outs, dim=1))


class VisionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _conv(HIDDEN, INTERMEDIATE)
        self.up_proj = _conv(HIDDEN, INTERMEDIATE)
        self.down_proj = _conv(INTERMEDIATE, HIDDEN)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class VisionLayer(nn.Module):
    def __init__(self, layer: int, scales: dict[str, float]) -> None:
        super().__init__()
        p = f"encoder.layers.{layer}."
        self.self_attn = VisionAttention(layer, scales)
        self.mlp = VisionMLP()
        self.input_layernorm = RMSNormBC1S(HIDDEN, scales.get(p + "input_layernorm", 1.0))
        self.post_attention_layernorm = RMSNormBC1S(HIDDEN, scales.get(p + "post_attention_layernorm", 1.0))
        self.pre_feedforward_layernorm = RMSNormBC1S(HIDDEN, scales.get(p + "pre_feedforward_layernorm", 1.0))
        self.post_feedforward_layernorm = RMSNormBC1S(HIDDEN, scales.get(p + "post_feedforward_layernorm", 1.0))

    def forward(self, x, cos, sin, mask):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class VisionEncoder(nn.Module):
    """(pixel_patches, pos_embeds, cos, sin, patch_mask) -> soft tokens (1, 512, 1, P / 9)."""

    def __init__(self, num_patches: int, scales: dict[str, float] | None = None) -> None:
        super().__init__()
        scales = scales or {}
        if num_patches % (POOL * POOL):
            raise ValueError(f"num_patches {num_patches} must be a multiple of {POOL * POOL}")
        self.num_patches = num_patches
        self.input_proj = _conv(3 * PATCH * PATCH, HIDDEN)
        self.layers = nn.ModuleList(VisionLayer(i, scales) for i in range(NUM_LAYERS))
        self.embedding_pre_projection_norm = RMSNormBC1S(
            HIDDEN, scales.get("embedding_pre_projection_norm", 1.0), with_weight=False
        )
        self.embedding_projection = _conv(HIDDEN, TEXT_HIDDEN)
        self.register_buffer("two", torch.tensor(2.0))
        self.register_buffer("one", torch.tensor(1.0))
        self.register_buffer("mask_neg", torch.tensor(MASK_NEG))

    def forward(self, pixel_patches, pos_embeds, cos, sin, patch_mask):
        key_mask = ((patch_mask - self.one) * -self.mask_neg).reshape(1, self.num_patches, 1, 1)
        x = self.input_proj(pixel_patches * self.two - self.one) + pos_embeds  # (1, 768, 1, P)
        for layer in self.layers:
            x = layer(x, cos, sin, key_mask)
        n = self.num_patches // (POOL * POOL)
        x = x.reshape(1, HIDDEN, n, POOL * POOL).mean(dim=-1, keepdim=True)  # (1, 768, n, 1): 3x3 block means
        x = x.reshape(1, HIDDEN, 1, n)
        return self.embedding_projection(self.embedding_pre_projection_norm(x))  # (1, 512, 1, n)


def load_vision_weights(positions: VisionPositions, encoder: VisionEncoder, safetensors_path) -> None:
    """Load ``vision_tower.*`` and ``embed_vision.*`` from the EmbeddingGemma 2 checkpoint."""
    from safetensors import safe_open

    with safe_open(str(safetensors_path), "pt") as f:
        sd = {k: f.get_tensor(k).float() for k in f.keys() if k.startswith(("vision_tower.", "embed_vision."))}

    with torch.no_grad():
        table = sd.pop("vision_tower.patch_embedder.position_embedding_table")  # (2, 10240, 768)
        positions.x_table.copy_(table[0])
        positions.y_table.copy_(table[1])
        own = {"input_proj.weight": sd.pop("vision_tower.patch_embedder.input_proj.weight"),
               "embedding_projection.weight": sd.pop("embed_vision.embedding_projection.weight")}
        for key, t in sd.items():
            name = key[len("vision_tower.encoder."):].replace(".linear.weight", ".weight")
            own[name] = t
        params = dict(encoder.named_parameters())
        missing, unexpected = set(params) - set(own), set(own) - set(params)
        if missing or unexpected:
            raise KeyError(f"vision weight mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
        for key, p in params.items():
            t = own[key]
            if p.dim() == 4 and t.dim() == 2:
                t = t[:, :, None, None]
            elif p.dim() == 4 and t.dim() == 1:
                t = t.reshape(1, -1, 1, 1)
            p.copy_(t.reshape(p.shape))


def block_order(grid_h: int, grid_w: int) -> torch.Tensor:
    """Patch indices (row-major in the HF patch grid) reordered so each 3x3 pooling block is contiguous.

    Block order follows HF's pooled-token order: block row-major over (grid_h/3, grid_w/3).
    """
    idx = torch.arange(grid_h * grid_w).reshape(grid_h // POOL, POOL, grid_w // POOL, POOL)
    return idx.permute(0, 2, 1, 3).reshape(-1)
