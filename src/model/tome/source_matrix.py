"""
Standard Function Enhancedby GitHub Copilot - Claude Haiku 4.5
Binary Source matrix: S (to capture size and mapping coordinates of merged tokens). 
S tracks the many-to-one mapping from original base positions (N) to
merged token positions (L) accumulated across all local encoder layers

  implementation -
  S[i, j] = 1 -> j was merged into i

Properties
----------
- Each column sums to exactly 1 (every base belongs to exactly one token).
- Row sums give token sizes (how many bases each token represents).
- The initial S is the identity: S = I_N.
- After all merges: S = S^(L-layers) ... S^(1), accumulated via matrix multiply.

We store S as a float tensor (not bool) so that the unmerge operation
    Z-dash_N = S^T . Zhat_L
is a standard matrix multiply: each base receives the embedding of the
token it was grouped into.


# Copilot suggestion for improvements:
For memory efficiency S is kept as a dense [B, L, N_orig] float32 tensor.
For very long sequences (N >> 4096) a sparse format would be preferable,
but 4096 * 4096 * 4 bytes = 64 MB per item, so we use dense here and
leave sparse optimisation as future work.
"""

from __future__ import annotations
import torch
from torch import Tensor


def init_source_matrix(B: int, N: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Return the identity source matrix for a batch of sequences of length N.

    S = I_N broadcast over batch dimension.

    Returns:
        [B, N, N] float tensor
    """
    eye = torch.eye(N, device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(B, -1, -1).clone()  # [B, N, N]


def update_source_matrix(
        # Copilot Enhanced
    S: Tensor,
    src_indices: Tensor,
    dst_indices: Tensor,
    unm_indices: Tensor,
    n_A: int,
) -> Tensor:
    """
    Compose S with one layer's merging decisions to get the new source matrix.

    After a merge step the current token sequence has length N_cur.
    Tokens were split into A (even) and B (odd) sets.  r pairs were merged:
    A[src_indices[k]] → B[dst_indices[k]].
    The remaining A tokens (unm_indices) plus all B tokens survive unchanged.

    The new token sequence is  [unmerged_A | B]  of length  N_cur - r.

    We update S so that S_new[i_new, j_orig] = 1 whenever original base j
    belongs to the new token at position i_new.

    Args:
        S:           [B, N_cur, N_orig]  current source matrix
        src_indices: [B, r, 1]   A-side indices of tokens being merged away
        dst_indices: [B, r, 1]   B-side indices of their merge targets
        unm_indices: [B, nA-r, 1] A-side indices of unmerged tokens
        n_A:         int, total number of A-side tokens in the current layer

    Returns:
        S_new: [B, N_cur - r, N_orig]
    """
    B, N_cur, N_orig = S.shape
    r = src_indices.shape[1]
    n_unm = unm_indices.shape[1]

    # Split S into A-side and B-side rows
    S_A = S[:, ::2, :]   # [B, nA, N_orig]
    S_B = S[:, 1::2, :]  # [B, nB, N_orig]

    # Gather the rows that are being merged (src) and their targets (dst)
    src_rows = S_A.gather(
        1, src_indices.expand(B, r, N_orig)
    )  # [B, r, N_orig]
    dst_rows = S_B.gather(
        1, dst_indices.expand(B, r, N_orig)
    )  # [B, r, N_orig]

    # Merge: union of base memberships. logical OR, implemented as addition
    # since values are binary 0/1 before the first merge. after merges they
    # remain less than 1 because we use clamp below
    merged_rows = (dst_rows + src_rows).clamp(max=1.0)

    # Write merged rows back into S_B at dst positions
    S_B_new = S_B.scatter(
        1,
        dst_indices.expand(B, r, N_orig),
        merged_rows,
    )

    # Unmerged A rows
    unm_rows = S_A.gather(
        1, unm_indices.expand(B, n_unm, N_orig)
    )  # [B, nA-r, N_orig]

    # New token order: [unmerged_A | B_updated]
    S_new = torch.cat([unm_rows, S_B_new], dim=1)  # [B, N_cur - r, N_orig]
    return S_new


def unmerge_with_source(Z_L: Tensor, S: Tensor) -> Tensor:
    """
    Expand L merged tokens back to N original positions using the source matrix.

    Implements  Zdash_N = S^T · Zhat_L  from the paper.

    Each base position j receives the embedding of whichever token i satisfies
    S[i, j] = 1, which is simply the j-th column of S dotted with Z_L.

    Args:
        Z_L: [B, L, D]  merged token embeddings
        S:   [B, L, N]  source matrix (each column is a one-hot vector)
    Returns:
        Z_N: [B, N, D]  base-level embeddings
    """
    # S^T: [B, N, L]
    # Zdash_N = S^T @ Z_L  →  [B, N, D]
    return torch.bmm(S.transpose(1, 2), Z_L)


def token_sizes_from_source(S: Tensor) -> Tensor:
    """
    Compute the size (number of original bases) of each merged token.

    Args:
        S: [B, L, N]  source matrix
    Returns:
        sizes: [B, L]  integer counts
    """
    return S.sum(dim=-1)  # [B, L]
