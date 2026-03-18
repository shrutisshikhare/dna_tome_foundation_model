"""
Local Encoder  E_phi

4 stacked LocalToMeAttn blocks that jointly perform:
  1. Local-window self-attention (window_size = 16)
  2. Differentiable token merging (ToMe) to reduce N -> L ~ N/2

Each block applies attention within fixed windows, then runs one round
of local bipartite soft matching to merge similar adjacent tokens.
The accumulated source matrix  S records which original
base positions belong to each merged token — needed by the Local Decoder
for reconstruction.

Compression schedule
--------------------
At training time, the total number of merges  r_total = N - L  is drawn
from a Gaussian centred at N/2 (so L ~ N/2) and clamped to [0.4 N, 0.6 N]
The per-layer merge count is  r_l = int(r_total / n_layers).
"""

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .transformer import RMSNorm, SwiGLUFFN
from .tome.local_tome import LocalToMeMerge
from .tome.source_matrix import init_source_matrix


# ---------------------------------------------------------------------------
# Local-window Self-Attention
# ---------------------------------------------------------------------------

class LocalWindowAttention(nn.Module):
    """
    Self-attention restricted to non-overlapping local windows.

    Each token only attends to tokens in the same window of `window_size`
    bases, keeping complexity O(N · w^2) rather than O(N^2).

    RoPE is applied locally within each window (positions 0 … w-1).

    Args:
        d_model:     Model dimension.
        n_heads:     Number of attention heads.
        window_size: Number of tokens per attention window (paper: 16).
        dropout:     Attention dropout.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        window_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.dropout = dropout
        self.scale = self.d_head ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable relative position bias within the window  [n_heads, w, w]
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(n_heads, window_size, window_size)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, D]   N must be a multiple of window_size (caller pads).
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        w = self.window_size
        H, Dh = self.n_heads, self.d_head

        # Pad if needed
        pad = (w - N % w) % w
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        N_padded = x.shape[1]
        n_wins = N_padded // w

        # Reshape into windows: [B*n_wins, w, D]
        x_win = x.reshape(B * n_wins, w, D)

        q = self.q_proj(x_win).reshape(B * n_wins, w, H, Dh).transpose(1, 2)  # [B*W, H, w, Dh]
        k = self.k_proj(x_win).reshape(B * n_wins, w, H, Dh).transpose(1, 2)
        v = self.v_proj(x_win).reshape(B * n_wins, w, H, Dh).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*W, H, w, w]
        attn = attn + self.rel_pos_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)                         # [B*W, H, w, Dh]
        out = out.transpose(1, 2).reshape(B * n_wins, w, D) # [B*W, w, D]
        out = self.out_proj(out)

        out = out.reshape(B, N_padded, D)
        return out[:, :N, :]   # strip padding


# -----------------------------------------------------------------------------------------
## Local ToMe Attention Block  (attention + merge)
# ---------------------------------------------------------------------------------------

class LocalToMeAttnBlock(nn.Module):
    """
    One layer of the Local Encoder:
        pre-norm LocalWindowAttention  ->  pre-norm SwiGLUFFN  ->  LocalToMeMerge

    Args:
        d_model:     Model dimension (1024).
        n_heads:     Attention heads (16).
        window_size: Attention and merging window (16).
        d_ff:        FFN hidden dimension (LLaMA default).
        d_group:     Grouping-embedding dimension for ToMe similarity.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        window_size: int = 16,
        d_ff: int | None = None,
        d_group: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn  = LocalWindowAttention(d_model, n_heads, window_size, dropout)
        self.ffn   = SwiGLUFFN(d_model, d_ff)
        self.merge = LocalToMeMerge(d_model, window_size, d_group)

    def forward(
        self,
        x: Tensor,
        S: Tensor,
        r: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, N, D]        input token embeddings
            S: [B, N, N_orig]   current source matrix
            r: int              number of tokens to merge this layer
        Returns:
            x: [B, N-r, D]
            S: [B, N-r, N_orig]
        """
        # Attention + FFN (residual connections around both)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # Token merging
        x, S = self.merge(x, S, r)
        return x, S


# ---------------------------------------------------------------------------
# Local Encoder  (4 stacked LocalToMeAttn blocks)
# ---------------------------------------------------------------------------

class LocalEncoder(nn.Module):
    """
    Local Encoder E_phi — the learnable DNA tokenizer.

    Stacks `n_layers` (paper: 4) LocalToMeAttn blocks that progressively
    compress N base embeddings down to L ~ N/2 merged tokens.

    Args:
        d_model:        Embedding dimension (1024).
        n_heads:        Attention heads (16).
        n_layers:       Number of local-attention + merge layers (4).
        window_size:    Local attention window size (16).
        d_ff:           FFN hidden dim.
        d_group:        Grouping embedding dim for ToMe.
        dropout:        Dropout.
        target_ratio:   Target compression ratio L/N (0.5 -> L = N/2).
        ratio_std:      Std-dev of compression ratio sampling during training.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 4,
        window_size: int = 16,
        d_ff: int | None = None,
        d_group: int | None = None,
        dropout: float = 0.0,
        target_ratio: float = 0.5,
        ratio_std: float = 0.05,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.target_ratio = target_ratio
        self.ratio_std = ratio_std

        self.layers = nn.ModuleList([
            LocalToMeAttnBlock(d_model, n_heads, window_size, d_ff, d_group, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def _sample_r_per_layer(self, N: int) -> list[int]:
        """
        Sample compression ratio and distribute merges evenly across layers.

        During training: L ~ N(N*target_ratio, N*ratio_std), clamped to [0.4N, 0.6N].
        During eval:     L = round(N * target_ratio) deterministically.

        Returns a list of per-layer merge counts, r_total = N - L.
        """
        if self.training:
            mean = N * self.target_ratio
            std  = N * self.ratio_std
            L = int(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std))
                    .clamp(0.4 * N, 0.6 * N).item())
        else:
            L = round(N * self.target_ratio)

        r_total = N - L
        # Distribute evenly; give any remainder to the first layers
        r_base, remainder = divmod(r_total, self.n_layers)
        return [r_base + (1 if i < remainder else 0) for i in range(self.n_layers)]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, N, D]  embedded nucleotide tokens
        Returns:
            z:  [B, L, D]      merged token embeddings
            S:  [B, L, N]      source matrix  (maps L tokens -> N original bases)
        """
        B, N, D = x.shape
        S = init_source_matrix(B, N, device=x.device, dtype=x.dtype)

        r_schedule = self._sample_r_per_layer(N)

        for layer, r in zip(self.layers, r_schedule):
            x, S = layer(x, S, r)

        z = self.norm(x)
        return z, S
