"""
Local Decoder  E_zeta

2 stacked local window attention blocks that map the latent decoder's
L-length output back to the original N base positions (the detokenizer).

The unmerging step (Zhat_N = S^T . Zhat_L) happens in the forward pass before this module is called
The local decoder then refines the per-base embeddings with local context and projects them over the VOCAB

Architecture mirrors the local encoder:
  - 2 LocalWindowAttention + SwiGLUFFN blocks (no merging)
  - Final RMSNorm
  - Linear head projecting D → vocab_size (5: A, C, G, T, N)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

from .transformer import SwiGLUFFN, RMSNorm 
from .local_encoder import LocalWindowAttention
from .embedding import VOCAB_SIZE


class LocalAttnBlock(nn.Module):
    """
    Local attention block without merging (used in the Local Decoder).

    Identical structure to LocalToMeAttnBlock but without the ToMe merge step.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        window_size: int = 16,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn  = LocalWindowAttention(d_model, n_heads, window_size, dropout)
        self.ffn   = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LocalDecoder(nn.Module):
    """
    Local Decoder E_zeta — 2 local-window attention blocks with reconstruction head

    Receives the unmerged base-level tensor Zbar_N  [B, N, D]  (produced by
    applying  S^T · Zhat_L  in the forward pass) and outputs per-base
    outputs over the nucleotide vocabulary.

    Args:
        d_model:      Embedding dimension (1024).
        n_heads:      Attention heads (16).
        n_layers:     Number of local attention blocks (2).
        window_size:  Local window size (16).
        d_ff:         FFN hidden dimension.
        vocab_size:   Output vocabulary size (5).
        dropout:      Dropout.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 2,
        window_size: int = 16,
        d_ff: int | None = None,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalAttnBlock(d_model, n_heads, window_size, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, D]  unmerged base-level token embeddings
        Returns:
            outputs: [B, N, vocab_size]  per-base prediction outputs (pre-softmax)
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)
