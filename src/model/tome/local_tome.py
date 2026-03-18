"""
Application Enhanced by Copilot:
Local-window Token Merging for the MergeDNA Local Encoder.

Adapts ToMe (Bolya et al., 2023) with two key changes:
  1. Merging is restricted to local windows of size `window_size` (default 16)
     so the computational complexity remains O(N) rather than O(N²).
  2. The similarity signal comes from a lightweight *grouping embedding*
     (a small separate linear projection, following DTEM / Lee & Hong 2024)
     rather than the main attention keys. This decouples where-to-merge
     from what-to-attend-to and is jointly learned end-to-end.

Interface
---------
LocalToMeMerge is a stateless nn.Module.  It takes the current token tensor
and source matrix, applies one round of local bipartite matching, and returns
the reduced token tensor and updated source matrix.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .source_matrix import update_source_matrix


class GroupingProjection(nn.Module):
    """
    Lightweight linear projection used to compute the merging similarity signal.

    Projects from d_model -> d_group using a single linear layer (no bias).
    d_group is kept small (paper does not specify; we default to d_model // 8)
    to make the similarity computation cheap.
    """

    def __init__(self, d_model: int, d_group: int | None = None):
        super().__init__()
        if d_group is None:
            d_group = max(32, d_model // 8)
        self.proj = nn.Linear(d_model, d_group, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """[B, N, D] -> [B, N, d_group] (L2-normalised)"""
        return F.normalize(self.proj(x), dim=-1)


class LocalToMeMerge(nn.Module):
    """
    One round of local-window bipartite token merging.

    Within each non-overlapping window of `window_size` tokens:
      - The window is split into A (even) and B (odd) sub-sequences.
      - Cosine similarity is computed between every (A_i, B_j) pair using the
        grouping embedding (not attention keys).
      - The top-r_window most similar pairs are merged (A_i absorbed into B_j).
      - The source matrix S is updated to reflect the new base->token mapping.

    Args:
        d_model:     Token embedding dimension (1024 in the paper).
        window_size: Local window size (16 in the paper).
        d_group:     Dimension of the grouping embedding. Defaults to d_model//8.
    """

    def __init__(self, d_model: int = 1024, window_size: int = 16, d_group: int | None = None):
        super().__init__()
        self.window_size = window_size
        self.grouping = GroupingProjection(d_model, d_group)

    def forward(
        self,
        x: Tensor,
        S: Tensor,
        r: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, N, D]       token embeddings (N must be divisible by window_size)
            S: [B, N, N_orig]  source matrix from the previous layer
            r: int             total number of merges to perform across *all* windows.
                               This is distributed evenly: r_window = r // n_windows.

        Returns:
            x_new: [B, N - r, D]
            S_new: [B, N - r, N_orig]
        """
        B, N, D = x.shape
        w = self.window_size

        # Pad N to a multiple of window_size if necessary
        pad = (w - N % w) % w
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
            # Extend S with zero rows (padding tokens map to no original base)
            S = F.pad(S, (0, 0, 0, pad))

        N_padded = x.shape[1]
        n_windows = N_padded // w

        # Distribute r merges evenly across windows
        r_window = max(0, r // n_windows)
        if r_window == 0:
            # Nothing to merge — return unchanged (after unpadding)
            return x[:, :N - r if r > 0 else N, :], S[:, :N - r if r > 0 else N, :]

        # --- Reshape into windows ---
        # x_win: [B * n_windows, w, D]
        x_win = x.reshape(B * n_windows, w, D)
        S_win = S.reshape(B * n_windows, w, -1)   # [B*W, w, N_orig]

        # --- Compute grouping embeddings ---
        g = self.grouping(x_win)   # [B*W, w, d_group]

        # --- Bipartite matching within each window ---
        # Split into A (even) and B (odd)
        g_a = g[:, ::2, :]   # [B*W, w//2, d_group]
        g_b = g[:, 1::2, :]  # [B*W, w//2, d_group]

        scores = torch.bmm(g_a, g_b.transpose(1, 2))  # [B*W, w//2, w//2]

        # For each a_i find best b_j
        node_max, node_idx = scores.max(dim=-1)   # [B*W, w//2]

        # Sort a-tokens by their best match score
        edge_idx = node_max.argsort(dim=-1, descending=True).unsqueeze(-1)  # [B*W, w//2, 1]
        src_idx = edge_idx[:, :r_window, :]    # [B*W, r_window, 1]
        unm_idx = edge_idx[:, r_window:, :]   # [B*W, w//2 - r_window, 1]
        dst_idx = node_idx.unsqueeze(-1).gather(dim=1, index=src_idx)  # [B*W, r_window, 1]

        # Keep unmerged in original order (stable sort)
        unm_idx = unm_idx.sort(dim=1)[0]

        # --- Merge token embeddings ---
        x_new_win = _merge_tokens(x_win, src_idx, dst_idx, unm_idx)    # [B*W, w - r_window, D]

        # --- Update source matrix ---
        S_new_win = update_source_matrix(
            S_win,
            src_idx,
            dst_idx,
            unm_idx,
            n_A=w // 2,
        )  # [B*W, w - r_window, N_orig]

        # --- Reshape back ---
        new_w = w - r_window
        x_new = x_new_win.reshape(B, n_windows * new_w, D)
        S_new = S_new_win.reshape(B, n_windows * new_w, -1)

        # Strip padding tokens (they contributed no original bases)
        final_len = N - r
        x_new = x_new[:, :final_len, :]
        S_new = S_new[:, :final_len, :]

        return x_new, S_new


def _merge_tokens(
    x: Tensor,
    src_idx: Tensor,
    dst_idx: Tensor,
    unm_idx: Tensor,
) -> Tensor:
    """
    Merge token embeddings given the matching indices.

    The merged token receives the *sum* of the src and dst embeddings.
    (Soft merge — differentiable; the Local Decoder can still reconstruct
    the individual bases because the source matrix preserves position info.)

    Args:
        x:       [B, N, D]
        src_idx: [B, r, 1]    A-side tokens to be merged away
        dst_idx: [B, r, 1]    B-side tokens that absorb them
        unm_idx: [B, nA-r, 1] A-side tokens that survive unchanged
    Returns:
        [B, N - r, D]
    """
    B, N, D = x.shape
    r = src_idx.shape[1]
    n_unm = unm_idx.shape[1]

    src_side = x[:, ::2, :]   # [B, nA, D]
    dst_side = x[:, 1::2, :]  # [B, nB, D]

    unm = src_side.gather(1, unm_idx.expand(B, n_unm, D))
    src = src_side.gather(1, src_idx.expand(B, r, D))

    # Add src into dst (scatter_add merges values at the target position)
    dst_new = dst_side.scatter_add(1, dst_idx.expand(B, r, D), src)

    return torch.cat([unm, dst_new], dim=1)
