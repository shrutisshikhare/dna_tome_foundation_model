"""
Shared LLaMA-style Transformer block used by the Latent Encoder and Latent Decoder

Design choices follow LLaMA:
  - Pre-norm with RMSNorm
  - Rotary Position Embeddings (RoPE) applied to Q and K
  - SwiGLU activation in the feed-forward network
  - Optional Flash Attention (falls back to scaled dot-product attention)

The local-window variant (used in Local Encoder / Local Decoder) is
implemented in local_encoder.py as a subclass that overrides the attention
call with a windowed version.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False

try:
    from rotary_embedding_torch import RotaryEmbedding
    _ROTARY_AVAILABLE = True
except ImportError:
    _ROTARY_AVAILABLE = False


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation used in LLaMA"""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
## SwiGLU Feed-Forward Network
# Copilot suggestion implemented below
# https://pub.towardsai.net/llama-explained-a70e71e706e9 and https://medium.com/@s_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7

# this article breaks down SwiGLU, ReGLU, GEGLU. Implementation in LLaMA uses SwiGLU which is executed below with the help of copilot and chatgpt.
# Learned implementation from github https://github.com/viai957/llama-inference. Enhanced with copilot and chatgpt.

# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block.

    FFN(x) = SiLU(xW_gate) o (xW_up) . W_down

    The hidden dimension is conventionally set to ⌊(4 * d_model * 2/3) / mult⌋ * mult
    rounded to the nearest multiple of `multiple_of` following LLaMA conventions.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, multiple_of: int = 256):
        super().__init__()
        if d_ff is None:
            # LLaMA formula: 4 * d * 2/3, rounded up to multiple_of
            d_ff = int(2 * 4 * d_model / 3)
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
## Multi-head Self-Attention with RoPE
# Copilot suggestion implemented below specifically for RoPE
# Learned implementation from github https://github.com/viai957/llama-inference. Enhanced with copilot and chatgpt.
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings.

    Uses Flash Attention when available, otherwise falls back to
    PyTorch's scaled_dot_product_attention (efficient on modern GPUs).

    Args:
        d_model:    Model dimension (1024 in paper).
        n_heads:    Number of attention heads. d_model must be divisible by n_heads.
        dropout:    Attention dropout probability (0 during inference).
        use_flash:  Use Flash Attention if installed. Auto-disabled if not available.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads" # bug fix
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.use_flash = use_flash and _FLASH_AVAILABLE

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if _ROTARY_AVAILABLE:
            self.rotary = RotaryEmbedding(dim=self.d_head)
        else:
            self.rotary = None

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        token_sizes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:            [B, N, D]
            attn_mask:    [B, N, N] or [B, 1, N, N] additive mask (–inf for masked positions)
            token_sizes:  [B, N] sizes for proportional attention (ToMe Eq. 1).
                          If provided, log(sizes) is added to the attention logits.
        Returns:
            out:   [B, N, D]   attention output
            keys:  [B, N, D_k] key vectors (used as similarity signal for global ToMe)
        """
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x).reshape(B, N, H, Dh)
        k = self.k_proj(x).reshape(B, N, H, Dh)
        v = self.v_proj(x).reshape(B, N, H, Dh)

        # Apply RoPE
        if self.rotary is not None:
            q = self.rotary.rotate_queries_or_keys(q.transpose(1, 2)).transpose(1, 2)
            k = self.rotary.rotate_queries_or_keys(k.transpose(1, 2)).transpose(1, 2)

        # Cache keys (averaged over heads) for global ToMe similarity
        keys_for_tome = k.mean(dim=2)   # [B, N, Dh]

        if self.use_flash:
            # Flash Attention expects [B, N, H, Dh]
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(Dh),
                causal=False,
            )   # [B, N, H, Dh]
        else:
            # Standard scaled dot-product attention
            q_ = q.transpose(1, 2)   # [B, H, N, Dh]
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)

            # Proportional attention: add log(sizes) to logits (ToMe Eq. 1)
            if token_sizes is not None:
                bias = torch.log(token_sizes.clamp(min=1.0))   # [B, N]
                # Broadcast as a key-side bias: [B, 1, 1, N]
                size_bias = bias.unsqueeze(1).unsqueeze(2)
            else:
                size_bias = None

            attn_logits = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(Dh)
            if size_bias is not None:
                attn_logits = attn_logits + size_bias
            if attn_mask is not None:
                attn_logits = attn_logits + attn_mask
            attn_weights = F.softmax(attn_logits, dim=-1)
            if self.dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            out = torch.matmul(attn_weights, v_)   # [B, H, N, Dh]
            out = out.transpose(1, 2)              # [B, N, H, Dh]

        out = out.reshape(B, N, D)
        out = self.out_proj(out)
        return out, keys_for_tome


# ---------------------------------------------------------------------------
# Full Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One LLaMA-style Transformer block: pre-norm attention + pre-norm FFN.

    Used by:
      - LatentEncoder (20 layers, full attention)
      - LatentDecoder  (4 layers, full attention)

    The local-window variant used in the Local Encoder / Decoder is defined
    in local_encoder.py as a subclass that replaces self.attn with a windowed
    attention module.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of attention heads.
        d_ff:     FFN hidden dimension. Defaults to LLaMA formula.
        dropout:  Dropout probability.
        use_flash: Use Flash Attention if available.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        d_ff: int | None = None,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, use_flash)
        self.ffn  = SwiGLUFFN(d_model, d_ff)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        token_sizes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            x:    [B, N, D]   updated token embeddings
            keys: [B, N, D_k] attention keys from this block (used by global ToMe)
        """
        attn_out, keys = self.attn(self.norm1(x), attn_mask=attn_mask, token_sizes=token_sizes)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, keys
