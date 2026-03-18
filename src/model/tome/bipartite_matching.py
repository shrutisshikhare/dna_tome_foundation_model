"""
Bipartite soft matching — the core ToMe algorithm
This file leverages on functions in ToMe's GitHub repo and adapts for 
https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py

Implementation follows Appendix D of the ToMe paper and then adds:
  - token-size tracking for proportional attention
  - a `merge_with_size` variant that performs size-weighted averaging
  - an `unmerge` operation that redistributes merged tokens back to original positions

The returned `merge` closure and `unmerge` closure share the index tensors
computed in `bipartite_soft_matching`, so they must be used together.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def bipartite_soft_matching(
    metric: Tensor,
    r: int,
    class_token: bool = False,
) -> tuple[callable, callable]:
    """
    Compute a bipartite matching between tokens based on `metric` similarity,
    then return (merge, unmerge) closures that can be applied to *any* tensor
    with the same token dimension.

    Algorithm :
      1. Partition tokens into sets A (even idx) and B (odd idx)
      2. For each token in A, find it's most similar token in B (using cosine similarity)
      3. Keep the r most similar pairs as merge candidates
      4. merge(): average each selected (A_i, B_j) pair into B_j, drop A_i. Removed A tokens are kept and concatenated with B.
      5. unmerge(): broadcast each merged B token back to its original positions

    Args:
        metric:       [B, N, C] — the similarity feature (e.g. normalised keys
                      or a dedicated grouping embedding). Will be L2-normalised
                      internally.
        r:            Number of token pairs to merge. 0 ≤ r ≤ N//2.
        class_token:  If True, protect index 0 from being merged (ViT CLS token
                      analogue — not used for DNA but kept for completeness).

    Returns:
        merge:   callable([B, N, C], size=None) → [B, N-r, C]
                 Applies the computed matching to any feature matrix.
                 If `size` ([B, N] token sizes) is passed the merge is
                 size-weighted (proportional attention).
        unmerge: callable([B, N-r, C]) → [B, N, C]
                 Reverses the merge by broadcasting merged tokens back.
    """
    B, N, _ = metric.shape

    if r <= 0:
        def _identity(x: Tensor, size: Tensor | None = None) -> Tensor:
            return x
        def _identity_unmerge(x: Tensor) -> Tensor:
            return x
        return _identity, _identity_unmerge

    # --- Normalise and split into A / B sets ---------------------------------
    metric = F.normalize(metric, dim=-1)
    a, b = metric[..., ::2, :], metric[..., 1::2, :]   # [B, ⌈N/2⌉, C] # matrix and equation augmented with github copilot

    # Cosine similarity matrix between every (a_i, b_j) pair
    scores = a @ b.transpose(-1, -2)                    # [B, nA, nB]

    # Protect the first token (class / BOS analogue) if requested
    if class_token:
        scores[..., 0, :] = -math.inf

    # For each a_i, find the best b_j
    node_max, node_idx = scores.max(dim=-1)             # [B, nA]

    # Sort a-side tokens by their best match score (descending)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # [B, nA, 1]

    unm_idx = edge_idx[..., r:, :]    # unmerged a tokens  [B, nA-r, 1]
    src_idx = edge_idx[..., :r, :]    # merged   a tokens  [B, r,    1]
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # [B, r, 1]

    # Keep unmerged tokens in original order (puts CLS back at idx 0)
    unm_idx = unm_idx.sort(dim=-2)[0]

    def merge(x: Tensor, size: Tensor | None = None) -> Tensor:
        """
        Args:
            x:    [B, N, C]
            size: S [B, N] optional token sizes for proportional attention weights
        Returns:
            [B, N-r, C]
        """
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        r_ = src_idx.shape[-2]

        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r_, c))
        src_ = src.gather(dim=-2, index=src_idx.expand(n, r_, c))

        if size is not None:
            # Size-weighted average: accumulate size-weighted features into dst,
            # then normalise by the new combined size.
            sz_src, sz_dst = size[..., ::2], size[..., 1::2]
            sz_src_ = sz_src.gather(dim=-1, index=src_idx[..., 0])  # [B, r]
            sz_dst_ = sz_dst.scatter_add(-1, dst_idx[..., 0], sz_src_)  # [B, nB]

            # Weight features before summing
            src_weighted = src_ * sz_src_[..., None]
            dst_weighted = dst * sz_dst[..., None]
            dst_accum = dst_weighted.scatter_add(-2, dst_idx.expand(n, r_, c), src_weighted)
            dst_out = dst_accum / sz_dst_[..., None].clamp(min=1e-6)
        else:
            dst_out = dst.scatter_add(-2, dst_idx.expand(n, r_, c), src_)

        return torch.cat([unm, dst_out], dim=-2)

    def unmerge(x: Tensor) -> Tensor:
        """
        Broadcast merged tokens back to the original N positions.

        Args:
            x: [B, N-r, C]  (output of merge)
        Returns:
            [B, N, C]
        """
        n, _, c = x.shape
        nA = unm_idx.shape[-2] + src_idx.shape[-2]  # original nA = t1
        nB = dst_idx.shape[-2] if dst_idx.ndim == 3 else (n,)

        unm_out = x[..., :unm_idx.shape[-2], :]
        dst_out = x[..., unm_idx.shape[-2]:, :]

        # Reconstruct A side
        src_out = dst_out.gather(dim=-2, index=dst_idx.expand(n, src_idx.shape[-2], c))

        # Place A and B tokens back into original even/odd positions
        out = torch.zeros(n, unm_idx.shape[-2] + src_idx.shape[-2] + dst_out.shape[-2], c,
                          device=x.device, dtype=x.dtype)
        # Re-interleave: A positions (even), B positions (odd)
        # We reconstruct the full A side first, then interleave
        a_full = torch.zeros(n, nA, c, device=x.device, dtype=x.dtype)
        a_full.scatter_(-2, unm_idx.expand(n, unm_idx.shape[-2], c), unm_out)
        a_full.scatter_(-2, src_idx.expand(n, src_idx.shape[-2], c), src_out)

        # Interleave A and B back into length-N sequence
        # A occupies even positions, B occupies odd positions
        nB_actual = dst_out.shape[-2]
        total = a_full.shape[-2] + nB_actual
        out = torch.zeros(n, total, c, device=x.device, dtype=x.dtype)
        out[..., ::2, :] = a_full
        out[..., 1::2, :] = dst_out

        return out

    return merge, unmerge


def merge_sizes(size: Tensor, merge_fn: callable) -> Tensor:
    """
    Update token-size vector after a merge operation.

    Token size tracks how many original positions each token represents —
    needed for proportional attention (Eq. 1 in ToMe paper).

    Args:
        size:     [B, N] current token sizes (all 1 at the start).
        merge_fn: the `merge` closure from bipartite_soft_matching.
    Returns:
        [B, N-r] updated sizes.
    """
    # merge_fn treats the last dimension as channels; unsqueeze then squeeze
    return merge_fn(size.unsqueeze(-1)).squeeze(-1)

# ----------------------------------------------------------------------------------------------------
## TO ME FUNCTIONS
# ----------------------------------------------------------------------------------------------------

# def kth_bipartite_soft_matching(
#     metric: torch.Tensor, k: int
# ) -> Tuple[Callable, Callable]:
#     """
#     Applies ToMe with the two sets as (every kth element, the rest).
#     If n is the number of tokens, resulting number of tokens will be n // z.

#     Input size is [batch, tokens, channels].
#     z indicates the stride for the first set.
#     z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
#     """
#     if k <= 1:
#         return do_nothing, do_nothing

#     def split(x):
#         t_rnd = (x.shape[1] // k) * k
#         x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
#         a, b = (
#             x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
#             x[:, :, (k - 1), :],
#         )
#         return a, b

#     with torch.no_grad():
#         metric = metric / metric.norm(dim=-1, keepdim=True)
#         a, b = split(metric)
#         r = a.shape[1]
#         scores = a @ b.transpose(-1, -2)

#         _, dst_idx = scores.max(dim=-1)
#         dst_idx = dst_idx[..., None]

#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         src, dst = split(x)
#         n, _, c = src.shape
#         dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

#         return dst

#     def unmerge(x: torch.Tensor) -> torch.Tensor:
#         n, _, c = x.shape
#         dst = x

#         src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

#         src = src.view(n, -1, (k - 1), c)
#         dst = dst.view(n, -1, 1, c)

#         out = torch.cat([src, dst], dim=-2)
#         out = out.contiguous().view(n, -1, c)

#         return out

#     return merge, unmerge


# def random_bipartite_soft_matching(
#     metric: torch.Tensor, r: int
# ) -> Tuple[Callable, Callable]:
#     """
#     Applies ToMe with the two sets as (r chosen randomly, the rest).
#     Input size is [batch, tokens, channels].

#     This will reduce the number of tokens by r.
#     """
#     if r <= 0:
#         return do_nothing, do_nothing

#     with torch.no_grad():
#         B, N, _ = metric.shape
#         rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

#         a_idx = rand_idx[:, :r, :]
#         b_idx = rand_idx[:, r:, :]

#         def split(x):
#             C = x.shape[-1]
#             a = x.gather(dim=1, index=a_idx.expand(B, r, C))
#             b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
#             return a, b

#         metric = metric / metric.norm(dim=-1, keepdim=True)
#         a, b = split(metric)
#         scores = a @ b.transpose(-1, -2)

#         _, dst_idx = scores.max(dim=-1)
#         dst_idx = dst_idx[..., None]

#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         src, dst = split(x)
#         C = src.shape[-1]
#         dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

#         return dst