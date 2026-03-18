"""
Latent Encoder  E_psi  and  Latent Decoder  E_w

Latent Encoder  (20 LLaMA-style transformer blocks, full attention)
- Takes the L merged tokens from the Local Encoder and builds rich contextual representations via full (global) self-attention over the entire sequence

- During PRE-TRAINING, the latent encoder runs a 2nd forward pass with global ToMe, compressing L tokens further down to K = L/2 tokens. This produces the source matrix S' used by:
  - The Latent Decoder for reconstruction
  - The AMTM objective to derive importance-weighted masking probabilities

Latent Decoder  (4 LLaMA-style transformer blocks, full attention)
- Symmetric to the Latent Encoder. Receives the output of the Latent Encoder (unmerged-back-to-L) and reconstructs the token space Zhat_L
- Only for PRE-TRAINING, not for sequence-level downstream tasks
"""

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn
from .transformer import TransformerBlock, RMSNorm
from .tome.global_tome import global_bipartite_merge
from .tome.source_matrix import unmerge_with_source


class LatentEncoder(nn.Module):
    """
    Latent Encoder E_psi — 20 full-attention transformer blocks.

    Two modes controlled by `use_tome`:
      - use_tome=False (default during inference): standard forward, returns Z'_L.
      - use_tome=True  (pre-training pass 2):  after the final block, applies
        one round of global bipartite matching to produce K = L/2 tokens, and returns (Z'_K, S').

    Args:
        d_model:    Embedding dimension (1024).
        n_heads:    Number of attention heads (16).
        n_layers:   Number of transformer blocks (20).
        d_ff:       FFN hidden dimension (LLaMA default if None).
        dropout:    Dropout probability.
        use_flash:  Use Flash Attention if available.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 20,
        d_ff: int | None = None,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_flash)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        z: Tensor,
        token_sizes: Tensor | None = None,
        use_tome: bool = False,
        attn_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            z:            [B, L, D]  token embeddings from Local Encoder
            token_sizes:  [B, L]    optional sizes for proportional attention
            use_tome:     bool      if True, apply global ToMe after final layer
                                    and return (Z'_K, S') instead of Z'_L
            attn_mask:    optional additive attention bias [B, L, L]

        Returns:
            Without ToMe:  Z'_L  [B, L, D]
            With    ToMe:  (Z'_K [B, K, D],  S' [B, K, L])
        """
        keys = None
        for layer in self.layers:
            z, keys = layer(z, attn_mask=attn_mask, token_sizes=token_sizes)
        z = self.norm(z)

        if not use_tome:
            return z

        # Global ToMe: compress L → K = L // 2 using final-layer keys
        B, L, D = z.shape
        r = L - L // 2   # number of merges needed
        z_K, S_prime_step = global_bipartite_merge(z, keys, r)

        # S_prime_step maps [B, K, L] — one merge step over identity(L)
        # This is the S' used by the AMTM objective and Latent Decoder
        return z_K, S_prime_step


class LatentDecoder(nn.Module):
    """
    Latent Decoder E_w — 4 full-attention transformer blocks.

    Receives the (K or L) token embeddings from the Latent Encoder and
    reconstructs the L-length token sequence Zhat_L that the Local Decoder
    can then use to reconstruct individual bases.

    Used only during pre-training; can be discarded at inference for classification tasks.

    Args:
        d_model:   Embedding dimension (1024)
        n_heads:   Attention heads (16)
        n_layers:  Number of blocks (4)
        d_ff:      FFN hidden dimension
        dropout:   Dropout
        use_flash: Flash Attention if available
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_flash)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        z: Tensor,
        S_prime: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            z:        [B, K, D]  or [B, L, D]  input from Latent Encoder.
                      When S_prime is provided, z has K tokens (Pass 2).
                      Otherwise z has L tokens (Pass 1 / Pass 3).
            S_prime:  [B, K, L]  optional source matrix from Latent Encoder ToMe.
                      When provided, we first unmerge z from K → L before
                      passing through the decoder blocks.
            attn_mask: optional attention bias.

        Returns:
            Zhat_L: [B, L, D]
        """
        if S_prime is not None:
            # Upsample K → L using the source matrix (Pass 2)
            z = unmerge_with_source(z, S_prime)   # [B, L, D]

        for layer in self.layers:
            z, _ = layer(z, attn_mask=attn_mask)
        return self.norm(z)
