"""
Pre-training loss functions for MergeDNA.

3 loss objectives:

  1. MTR  — Merged Token Reconstruction
     Cross-entropy between the full reconstructed sequence and the original.
     L_MTR = ... equation (6) from paper

  2. Latent MTR  (same formula, different forward pass)
     Same cross-entropy but the forward pass ran with the Local Encoder frozen
     and global ToMe in the Latent Encoder.  Weighted by lambda = 0.25.

  3. AMTM  — Adaptive Masked Token Modeling
     Masked cross-entropy over ONLY the K important (masked) positions.
     L_AMTM = ... equation (7) from paper

Combined:
  L_total = L_MTR + lambda * L_LatentMTR + L_AMTM
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def mtr_loss(outputs: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Merged Token Reconstruction loss (equation 6).

    Average cross-entropy over ALL positions in the sequence.

    Args:
        outputs:       [B, N, V]  per-base prediction outputs (pre-softmax)
        targets:      [B, N]     ground-truth integer token IDs
        ignore_index: positions with this target ID are excluded from the loss
                      (used for padding tokens).
    Returns:
        Scalar loss.
    """
    B, N, V = outputs.shape
    # Improvement: F.cross_entropy expects [B*N, V] and [B*N]
    loss = F.cross_entropy(
        outputs.reshape(B * N, V),
        targets.reshape(B * N),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


def amtm_loss(
    outputs: Tensor,
    targets: Tensor,
    mask: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """
    Adaptive Masked Token Modeling loss (equation 7).

    Cross-entropy computed only over the K masked positions identified by
    the importance-weighted sampling procedure in global_tome.py.

    Args:
        outputs:       [B, N, V]  per-base prediction outputs
        targets:      [B, N]     ground-truth integer token IDs (UNMASKED original)
        mask:         [B, N]     boolean tensor — True at masked positions
        ignore_index: excluded from loss (padding).
    Returns:
        Scalar loss  (returns 0 if no tokens are masked).
    """
    B, N, V = outputs.shape

    if not mask.any():
        return outputs.new_zeros(())

    # Build a target tensor where non-masked positions are set to ignore_index
    masked_targets = targets.clone()
    masked_targets[~mask] = ignore_index

    loss = F.cross_entropy(
        outputs.reshape(B * N, V),
        masked_targets.reshape(B * N),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


def combined_pretrain_loss(
    loss_mtr: Tensor,
    loss_latent_mtr: Tensor,
    loss_amtm: Tensor,
    lambda_latent: float = 0.25,
) -> Tensor:
    """
    Combine the three pre-training losses (equation 8).

    L_total = L_MTR + lambda * L_LatentMTR + L_AMTM

    Args:
        loss_mtr:         Pass 1 MTR loss
        loss_latent_mtr:  Pass 2 Latent MTR loss
        loss_amtm:        Pass 3 AMTM loss
        lambda_latent:    Dampening factor lambda (=0.25)
    Returns:
        Combined Loss
    """
    return loss_mtr + lambda_latent * loss_latent_mtr + loss_amtm
