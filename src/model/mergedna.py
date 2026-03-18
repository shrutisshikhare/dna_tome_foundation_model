"""
MergeDNA — full model

Assembles the four components into a single nn.Module and exposes the
three forward passes used during pre-training, plus a clean inference
path for downstream tasks.

Pre-training forward passes (Section 3 of the paper)

Pass 1 — Merged Token Reconstruction (MTR)
Pass 2 — Latent MTR
Pass 3 — Adaptive Masked Token Modeling (AMTM)

Combined loss:
  L_total = L_MTR + 0.25 * L_LatentMTR + L_AMTM

Inference:
  - Encoder-only (classification / regression):  drop both decoders,
    use LatentEncoder output + task head.
  - Full autoencoder (base-resolution tasks):    use all four components.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from .embedding import DNAEmbedding, VOCAB_SIZE
from .local_encoder import LocalEncoder
from .latent_encoder import LatentEncoder, LatentDecoder
from .local_decoder import LocalDecoder
from .tome.source_matrix import unmerge_with_source, token_sizes_from_source
from .tome.global_tome import compute_importance_mask, expand_token_mask_to_bases


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class MergeDNAConfig:
    # Dimensions
    d_model: int = 1024
    n_heads: int = 16
    vocab_size: int = VOCAB_SIZE

    # Local Encoder / Decoder
    local_enc_layers: int = 4
    local_dec_layers: int = 2
    window_size: int = 16
    d_group: int | None = None     # grouping-embed dim. None -> d_model / 8

    # Latent Encoder / Decoder
    latent_enc_layers: int = 20
    latent_dec_layers: int = 4

    # Compression ratios
    local_ratio: float = 0.5       # Local Enc: L = N * local_ratio
    latent_ratio: float = 0.5      # Latent Enc (ToMe): K = L * latent_ratio

    # FFN
    d_ff: int | None = None        # None -> LLaMA formula

    # Training
    dropout: float = 0.0
    use_flash: bool = True
    ratio_std: float = 0.05        # std of compression ratio sampling

    # Loss weights
    lambda_latent_mtr: float = 0.25


# ---------------------------------------------------------------------------
# MergeDNA
# ---------------------------------------------------------------------------

class MergeDNA(nn.Module):
    """
    MergeDNA: context-aware genome modelling with dynamic tokenisation.
    https://arxiv.org/pdf/2511.14806

    Args:
        config: MergeDNAConfig dataclass with all hyperparameters.
    """

    def __init__(self, config: MergeDNAConfig | None = None):
        super().__init__()
        if config is None:
            config = MergeDNAConfig()
        self.config = config

        self.embedding = DNAEmbedding(config.d_model, config.vocab_size)

        self.local_encoder = LocalEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,

            n_layers=config.local_enc_layers,
            window_size=config.window_size,
            d_ff=config.d_ff, # additional ffn necessary to make the similarity computation cheap
            d_group=config.d_group,
            dropout=config.dropout,
            target_ratio=config.local_ratio,
            ratio_std=config.ratio_std,
        )

        self.latent_encoder = LatentEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.latent_enc_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            use_flash=config.use_flash,
        )

        self.latent_decoder = LatentDecoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.latent_dec_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            use_flash=config.use_flash,
        )

        self.local_decoder = LocalDecoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.local_dec_layers,
            window_size=config.window_size,
            d_ff=config.d_ff,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, token_ids: Tensor) -> Tensor:
        """[B, N] -> [B, N, D]"""
        return self.embedding(token_ids)

    def _local_encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """[B, N, D] -> Z_L [B, L, D],  S [B, L, N]"""
        return self.local_encoder(x)

    def _decode_to_bases(self, z_hat_L: Tensor, S: Tensor) -> Tensor:
        """Ẑ_L [B, L, D] + S [B, L, N] -> outputs [B, N, V]"""
        z_bar_N = unmerge_with_source(z_hat_L, S)   # [B, N, D]
        return self.local_decoder(z_bar_N)            # [B, N, V]

    # ------------------------------------------------------------------
    # Pass 1: Merged Token Reconstruction (trains all θ)
    # ------------------------------------------------------------------

    def forward_mtr(self, token_ids: Tensor) -> Tensor:
        """
        Full autoencoder forward pass for the MTR objective

        X -> embed -> LocalEnc -> LatentEnc -> LatentDec -> LocalDec -> outputs

        Args:
            token_ids: [B, N] integer token IDs
        Returns:
            outputs: [B, N, V]
        """
        x = self._embed(token_ids)               # [B, N, D]
        z_L, S = self._local_encode(x)           # [B, L, D], [B, L, N]

        token_sizes = token_sizes_from_source(S)  # [B, L] for proportional attn
        z_L_prime = self.latent_encoder(z_L, token_sizes=token_sizes)   # [B, L, D]
        z_hat_L   = self.latent_decoder(z_L_prime)                       # [B, L, D]

        outputs = self._decode_to_bases(z_hat_L, S)   # [B, N, V]
        return outputs

    # ------------------------------------------------------------------
    # Pass 2: Latent MTR (trains ψ, ω, ζ — NOT φ)
    # ------------------------------------------------------------------

    def forward_latent_mtr(
        self,
        token_ids: Tensor,
        S_frozen: Tensor,
        Z_L_frozen: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Latent MTR pass: Local Encoder is frozen; Latent Encoder runs with
        global ToMe to select K salient tokens.

        Caller is responsible for detaching Z_L_frozen and S_frozen so that
        gradients do not flow through the Local Encoder.

        Args:
            token_ids:   [B, N]        original (unmasked) token IDs — for the loss target
            S_frozen:    [B, L, N]     detached source matrix from Pass 1 / a frozen run
            Z_L_frozen:  [B, L, D]     detached local encoder output

        Returns:
            outputs:      [B, N, V]     reconstruction outputs
            S_prime:     [B, K, L]     latent-level source matrix (for AMTM mask derivation)
            K:           int            number of salient tokens
        """
        token_sizes = token_sizes_from_source(S_frozen)
        # Latent Encoder with global ToMe -> Z'_K, S'
        z_K, S_prime = self.latent_encoder(
            Z_L_frozen, token_sizes=token_sizes, use_tome=True
        )   # z_K: [B, K, D],  S_prime: [B, K, L]

        K = z_K.shape[1]

        # Latent Decoder unmerges K -> L via S', then decodes
        z_hat_L = self.latent_decoder(z_K, S_prime=S_prime)   # [B, L, D]
        outputs  = self._decode_to_bases(z_hat_L, S_frozen)    # [B, N, V]

        return outputs, S_prime, K

    # ------------------------------------------------------------------
    # Pass 3: Adaptive Masked Token Modeling (trains all θ)
    # ------------------------------------------------------------------

    def forward_amtm(
        self,
        token_ids: Tensor,
        S_prime: Tensor,
        S_local: Tensor,
        K: int,
    ) -> tuple[Tensor, Tensor]:
        """
        AMTM pass: apply importance-weighted masking (derived from S') and
        predict the masked positions.

        Args:
            token_ids: [B, N]     original (unmasked) token IDs
            S_prime:   [B, K, L]  latent source matrix from Pass 2
            S_local:   [B, L, N]  local encoder source matrix (frozen copy)
            K:         int        number of tokens to mask

        Returns:
            outputs:   [B, N, V]  prediction outputs for all positions
            M_N:      [B, N]     boolean mask (True = masked base)
        """
        B, N = token_ids.shape

        # Derive importance-weighted mask
        L = S_local.shape[1]
        mask_token_idx = compute_importance_mask(S_prime, L, K)   # [B, K]
        M_N = expand_token_mask_to_bases(mask_token_idx, S_local)  # [B, N]

        # Apply mask: replace masked positions with a mask token (use vocab index 0
        # — the caller / dataset should reserve a [MASK] token, but here we use 0
        # as a simple stand-in; a dedicated MASK_ID constant can be added later)
        MASK_ID = 0
        masked_ids = token_ids.clone()
        masked_ids[M_N] = MASK_ID

        # Full forward (no ToMe in latent encoder during AMTM)
        x = self._embed(masked_ids)
        z_L, S_m = self._local_encode(x)

        token_sizes = token_sizes_from_source(S_m)
        z_L_prime = self.latent_encoder(z_L, token_sizes=token_sizes)
        z_hat_L   = self.latent_decoder(z_L_prime)
        outputs    = self._decode_to_bases(z_hat_L, S_m)

        return outputs, M_N

    # ------------------------------------------------------------------
    # Combined pre-training step
    # ------------------------------------------------------------------

    def pretrain_step(self, token_ids: Tensor) -> dict[str, Tensor]:
        """
        Execute all three forward passes and return individual losses and
        the combined weighted loss.

        This method is the single entry point called by the training loop.

        Args:
            token_ids: [B, N]  integer nucleotide token IDs (unmasked)
        Returns:
            dict with keys:
              "loss"            — combined loss (scalar)
              "loss_mtr"        — Pass 1 MTR loss
              "loss_latent_mtr" — Pass 2 Latent MTR loss
              "loss_amtm"       — Pass 3 AMTM loss
        """
        from ..training.objectives import mtr_loss, amtm_loss

        # --- Pass 1: MTR (all params) ---
        outputs_1 = self.forward_mtr(token_ids)
        loss_mtr = mtr_loss(outputs_1, token_ids)

        # --- Pass 2: Latent MTR (freeze φ) ---
        # First get Z_L and S with no_grad so Local Encoder is frozen
        with torch.no_grad():
            x_frozen = self._embed(token_ids)
            z_L_frozen, S_frozen = self._local_encode(x_frozen)

        outputs_2, S_prime, K = self.forward_latent_mtr(
            token_ids,
            S_frozen=S_frozen.detach(),
            Z_L_frozen=z_L_frozen.detach(),
        )
        loss_latent_mtr = mtr_loss(outputs_2, token_ids)

        # --- Pass 3: AMTM (all params) ---
        outputs_3, M_N = self.forward_amtm(
            token_ids,
            S_prime=S_prime.detach(),
            S_local=S_frozen.detach(),
            K=K,
        )
        loss_amtm = amtm_loss(outputs_3, token_ids, M_N)

        lam = self.config.lambda_latent_mtr
        total = loss_mtr + lam * loss_latent_mtr + loss_amtm

        return {
            "loss":             total,
            "loss_mtr":         loss_mtr.detach(),
            "loss_latent_mtr":  loss_latent_mtr.detach(),
            "loss_amtm":        loss_amtm.detach(),
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def encode(self, token_ids: Tensor) -> Tensor:
        """
        Encode a batch of DNA sequences for downstream tasks.

        Runs Local Encoder -> Latent Encoder, returns contextual token
        embeddings.  Both decoders are not used.

        Args:
            token_ids: [B, N]
        Returns:
            z: [B, L, D]  contextual latent embeddings (L ≈ N/2)
        """
        self.eval()
        with torch.no_grad():
            x = self._embed(token_ids)
            z_L, S = self._local_encode(x)
            token_sizes = token_sizes_from_source(S)
            z = self.latent_encoder(z_L, token_sizes=token_sizes)
        return z

    def encode_pooled(self, token_ids: Tensor, mode: str = "mean") -> Tensor:
        """
        Encode and pool to a single vector per sequence.

        Args:
            token_ids: [B, N]
            mode:      "mean" | "max" | "cls"  (cls uses position 0)
        Returns:
            [B, D]
        """
        z = self.encode(token_ids)   # [B, L, D]
        if mode == "mean":
            return z.mean(dim=1)
        if mode == "max":
            return z.max(dim=1).values
        if mode == "cls":
            return z[:, 0, :]
        raise ValueError(f"Unknown pooling mode: {mode!r}")

    def reconstruct(self, token_ids: Tensor) -> Tensor:
        """
        Full autoencoder reconstruction.

        Args:
            token_ids: [B, N]
        Returns:
            outputs: [B, N, V]  per-base prediction outputs
        """
        self.eval()
        with torch.no_grad():
            return self.forward_mtr(token_ids)
