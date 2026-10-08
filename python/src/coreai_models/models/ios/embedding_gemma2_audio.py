"""EmbeddingGemma 2 audio encoder (Gemma 4 USM conformer + embed_audio), authored for the Neural Engine.

Matches ``transformers`` 5.19 ``Gemma4AudioModel`` followed by ``EmbeddingGemma2MultimodalEmbedder``
for one clip in a static bucket of T mel frames (T/4 audio tokens).

Input: log-mel features (1, 1, T, 128) fp16 and a frame mask (1, T) (1 = real frame).
Output: soft tokens (1, 512, 1, T/4); the first ceil(ceil(n_frames / 2) / 2) are real.

Rewrites (all exact up to fp16 rounding):
  - Local attention without blocks: every token attends to itself and the 11 previous tokens
    (HF sliding_window_mask_function((12, 0)): 0 <= q - k < 12). For each distance d = 0..11 the logit is
    q . (k_scale * k[t - d] + rel_d), with rel_d = relative_k_proj(sinusoid(d)) a per-layer constant
    computed at load. Then softcap 50, mask, softmax over the 12 distances, weighted sum of shifted values.
    Only shifts, elementwise products and channel sums: no T x T tensor, no gather.
  - RMSNorm / LayerNorm prescale per norm (``fp16_scales.json`` "audio").
  - HF's clamp to +-finfo.max ("gradient_clipping") becomes a clamp to the fp16 range.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.ios.embedding_gemma2_text import MASK_NEG, RMSNormBC1S, _conv

HIDDEN = 1024
HEADS = 8
HEAD_DIM = 128
LAYERS = 12
MEL = 128
SUB_CH = (128, 32)
OUT_DIMS = 1536
TEXT_HIDDEN = 512
DISTANCES = 12  # attention sees distances 0..11 (attention_context_left - 1)
CONV_K = 5
SOFTCAP = 50.0
RESIDUAL_WEIGHT = 0.5
FP16_MAX = 65504.0


def _sinusoid(pos: torch.Tensor, dim: int = HIDDEN) -> torch.Tensor:
    """Gemma4AudioRelPositionalEncoding for given positions: [sin, cos] over dim/2 timescales."""
    n = dim // 2
    inc = math.log(10000.0) / max(n - 1, 1)
    inv = torch.exp(torch.arange(n, dtype=torch.float32) * -inc)
    t = pos.float()[:, None] * inv[None, :]
    return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


def _shift(x: torch.Tensor, d: int) -> torch.Tensor:
    """x[..., t] -> x[..., t - d] along the last axis, zero-filled."""
    if d == 0:
        return x
    return torch.cat([torch.zeros_like(x[..., :d]), x[..., :-d]], dim=-1)


class LayerNormC(nn.Module):
    """LayerNorm over channels (dim 1) without bias, with an input prescale (exact: LN is scale-invariant)."""

    def __init__(self, dim: int, eps: float, scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.register_buffer("inv_scale", torch.tensor(1.0 / scale))
        self.register_buffer("eps", torch.tensor(eps / (scale * scale)))

    def forward(self, x):
        x = x * self.inv_scale
        mu = x.mean(dim=1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=1, keepdim=True)
        return xc * torch.rsqrt(var + self.eps) * self.weight


class Clipped(nn.Module):
    """Gemma4ClippableLinear as a 1x1 conv with fixed input/output clamps from the checkpoint."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.linear = _conv(cin, cout)
        for n in ("input_min", "input_max", "output_min", "output_max"):
            self.register_buffer(n, torch.tensor(0.0))

    def forward(self, x):
        x = torch.clamp(x, self.input_min, self.input_max)
        return torch.clamp(self.linear(x), self.output_min, self.output_max)


def _clip16(x):
    return torch.clamp(x, -FP16_MAX, FP16_MAX)


class FeedForward(nn.Module):
    def __init__(self, prefix: str, scales: dict[str, float]) -> None:
        super().__init__()
        self.ffw_layer_1 = Clipped(HIDDEN, 4 * HIDDEN)
        self.ffw_layer_2 = Clipped(4 * HIDDEN, HIDDEN)
        self.pre_layer_norm = RMSNormBC1S(HIDDEN, scales.get(prefix + "pre_layer_norm", 1.0))
        self.post_layer_norm = RMSNormBC1S(HIDDEN, scales.get(prefix + "post_layer_norm", 1.0))
        self.register_buffer("residual_weight", torch.tensor(RESIDUAL_WEIGHT))

    def forward(self, x):
        h = self.pre_layer_norm(_clip16(x))
        h = self.ffw_layer_2(F.silu(self.ffw_layer_1(h)))
        return x + self.post_layer_norm(_clip16(h)) * self.residual_weight


class LocalAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = Clipped(HIDDEN, HIDDEN)
        self.k_proj = Clipped(HIDDEN, HIDDEN)
        self.v_proj = Clipped(HIDDEN, HIDDEN)
        self.post = Clipped(HIDDEN, HIDDEN)
        self.register_buffer("q_scale", torch.zeros(1, HIDDEN, 1, 1))  # q_scale * softplus(per_dim_scale), per channel
        self.register_buffer("k_scale", torch.tensor(math.log(1 + math.e) / math.log(2)))
        self.register_buffer("rel", torch.zeros(DISTANCES, 1, HIDDEN, 1, 1))  # rel_d, d = 0..11
        self.register_buffer("softcap", torch.tensor(SOFTCAP))

    def forward(self, x, key_masks):
        """x (1, 1024, 1, T); key_masks (1, 1, 12, T) additive, already shifted per distance."""
        t = x.shape[-1]
        q = self.q_proj(x) * self.q_scale
        k = self.k_proj(x) * self.k_scale
        v = self.v_proj(x)
        logits, values = [], []
        for d in range(DISTANCES):
            kd = _shift(k, d) + self.rel[d]
            logits.append((q * kd).reshape(1, HEADS, HEAD_DIM, t).sum(dim=2, keepdim=True))  # (1, 8, 1, T)
            values.append(_shift(v, d).reshape(1, HEADS, HEAD_DIM, t))
        s = torch.cat(logits, dim=2)  # (1, 8, 12, T)
        s = torch.tanh(s / self.softcap) * self.softcap + key_masks
        w = s.softmax(dim=2)
        out = values[0] * w[:, :, 0:1]
        for d in range(1, DISTANCES):
            out = out + values[d] * w[:, :, d : d + 1]
        return self.post(out.reshape(1, HIDDEN, 1, t))


class LightConv(nn.Module):
    def __init__(self, prefix: str, scales: dict[str, float]) -> None:
        super().__init__()
        self.linear_start = Clipped(HIDDEN, 2 * HIDDEN)
        self.linear_end = Clipped(HIDDEN, HIDDEN)
        self.depthwise_conv1d = nn.Conv2d(HIDDEN, HIDDEN, kernel_size=(1, CONV_K), groups=HIDDEN, bias=False)
        self.pre_layer_norm = RMSNormBC1S(HIDDEN, scales.get(prefix + "pre_layer_norm", 1.0))
        self.conv_norm = RMSNormBC1S(HIDDEN, scales.get(prefix + "conv_norm", 1.0))

    def forward(self, x):
        h = self.linear_start(self.pre_layer_norm(x))
        h = h[:, :HIDDEN] * torch.sigmoid(h[:, HIDDEN:])  # GLU over channels
        h = self.depthwise_conv1d(F.pad(h, (CONV_K - 1, 0)))  # causal: left pad 4
        h = self.conv_norm(_clip16(h))
        return x + self.linear_end(F.silu(h))


class AudioLayer(nn.Module):
    def __init__(self, i: int, scales: dict[str, float]) -> None:
        super().__init__()
        p = f"layers.{i}."
        self.feed_forward1 = FeedForward(p + "feed_forward1.", scales)
        self.feed_forward2 = FeedForward(p + "feed_forward2.", scales)
        self.self_attn = LocalAttention()
        self.lconv1d = LightConv(p + "lconv1d.", scales)
        self.norm_pre_attn = RMSNormBC1S(HIDDEN, scales.get(p + "norm_pre_attn", 1.0))
        self.norm_post_attn = RMSNormBC1S(HIDDEN, scales.get(p + "norm_post_attn", 1.0))
        self.norm_out = RMSNormBC1S(HIDDEN, scales.get(p + "norm_out", 1.0))

    def forward(self, x, key_masks):
        x = self.feed_forward1(x)
        h = self.self_attn(self.norm_pre_attn(_clip16(x)), key_masks)
        x = x + self.norm_post_attn(_clip16(h))
        x = self.lconv1d(x)
        x = self.feed_forward2(x)
        return self.norm_out(_clip16(x))


class SubsampleLayer(nn.Module):
    def __init__(self, cin: int, cout: int, prefix: str, scales: dict[str, float]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = LayerNormC(cout, 1e-6, scales.get(prefix + "norm", 1.0))

    def forward(self, x, frame_mask):
        """x (1, C, T, F); frame_mask (1, 1, T, 1)."""
        return F.relu(self.norm(self.conv(x * frame_mask)))


class AudioEncoder(nn.Module):
    """(mel (1, 1, T, 128), frame_mask (1, T)) -> soft tokens (1, 512, 1, T / 4)."""

    def __init__(self, mel_frames: int, scales: dict[str, float] | None = None) -> None:
        super().__init__()
        scales = scales or {}
        if mel_frames % 4:
            raise ValueError(f"mel_frames must be a multiple of 4, got {mel_frames}")
        self.mel_frames = mel_frames
        sp = "subsample_conv_projection."
        self.layer0 = SubsampleLayer(1, SUB_CH[0], sp + "layer0.", scales)
        self.layer1 = SubsampleLayer(SUB_CH[0], SUB_CH[1], sp + "layer1.", scales)
        self.input_proj_linear = _conv((SUB_CH[0] // 4) * SUB_CH[1], HIDDEN)
        self.layers = nn.ModuleList(AudioLayer(i, scales) for i in range(LAYERS))
        self.output_proj = nn.Conv2d(HIDDEN, OUT_DIMS, kernel_size=1, bias=True)
        self.embedding_pre_projection_norm = RMSNormBC1S(
            OUT_DIMS, scales.get("embedding_pre_projection_norm", 1.0), with_weight=False)
        self.embedding_projection = _conv(OUT_DIMS, TEXT_HIDDEN)
        self.register_buffer("mask_neg", torch.tensor(MASK_NEG))
        self.register_buffer("one", torch.tensor(1.0))

    def forward(self, mel, frame_mask):
        t = self.mel_frames
        m0 = frame_mask.reshape(1, 1, t, 1)
        m1 = frame_mask.reshape(1, 1, t // 2, 2)[..., :1]  # mask[::2]
        tokens = frame_mask.reshape(1, 1, t // 4, 4)[..., 0]  # mask[::4] -> (1, 1, T/4)
        h = self.layer0(mel, m0)  # (1, 128, T/2, 64)
        h = self.layer1(h, m1)  # (1, 32, T/4, 32): (channel c, time, freq f)
        n = t // 4
        h = h.permute(0, 3, 1, 2).reshape(1, (SUB_CH[0] // 4) * SUB_CH[1], 1, n)  # feature = f * 32 + c
        x = self.input_proj_linear(h)  # (1, 1024, 1, T/4)

        tok = tokens.reshape(1, 1, 1, n)
        key_masks = torch.cat([(_shift(tok, d) - self.one) * -self.mask_neg for d in range(DISTANCES)], dim=2)
        for layer in self.layers:
            x = layer(x, key_masks)
        x = self.output_proj(x)
        return self.embedding_projection(self.embedding_pre_projection_norm(x))  # (1, 512, 1, T/4)


def load_audio_weights(encoder: AudioEncoder, safetensors_path) -> None:
    from safetensors import safe_open

    with safe_open(str(safetensors_path), "pt") as f:
        sd = {k: f.get_tensor(k).float() for k in f.keys() if k.startswith(("audio_tower.", "embed_audio."))}
    own: dict[str, torch.Tensor] = {}
    for key, t in sd.items():
        if key.startswith("embed_audio."):
            own[key[len("embed_audio."):]] = t
            continue
        name = key[len("audio_tower."):]
        name = name.replace("subsample_conv_projection.", "")
        if name.endswith("self_attn.per_dim_scale"):
            q_scale = (HEAD_DIM**-0.5) / math.log(2)
            own[name.replace("per_dim_scale", "q_scale")] = (q_scale * F.softplus(t)).repeat(HEADS)
            continue
        if name.endswith("self_attn.relative_k_proj.weight"):
            rel = F.linear(_sinusoid(torch.arange(DISTANCES)), t)  # (12, 1024): distance d -> rel_d
            own[name.replace("relative_k_proj.weight", "rel")] = rel
            continue
        if name.endswith("depthwise_conv1d.weight"):
            own[name] = t[:, :, None, :]  # (1024, 1, 5) -> (1024, 1, 1, 5)
            continue
        own[name] = t
    with torch.no_grad():
        state = encoder.state_dict()
        skip = {k for k in state if k.endswith((".inv_scale", ".eps", "k_scale", "softcap", "residual_weight",
                                                "mask_neg")) or k == "one"}
        missing = set(state) - set(own) - skip
        unexpected = set(own) - set(state)
        if missing or unexpected:
            raise KeyError(f"audio weight mismatch: missing={sorted(missing)[:6]} unexpected={sorted(unexpected)[:6]}")
        for key, t in own.items():
            dst = state[key]
            if dst.dim() == 4 and t.dim() == 2:
                t = t[:, :, None, None]
            elif dst.dim() == 4 and t.dim() == 1:
                t = t.reshape(1, -1, 1, 1) if dst.shape[0] == 1 else t
            dst.copy_(t.reshape(dst.shape))
