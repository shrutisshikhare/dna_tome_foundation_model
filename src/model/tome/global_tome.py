"""
Application Enhanced by Copilot:
Global Token Merging for the MergeDNA Latent Encoder.

Used only during pre-training (2nd pass of the training loop) to select the
K most informative tokens from the L merged tokens produced by the Local
Encoder.  The global bipartite matching operates over the full sequence
(not windowed), identifying tokens that are redundant (mergeable) vs. tokens
that carry unique information (survivors).

The resultant source matrix  S' records which of the L
local tokens were grouped into each of the K latent tokens.  It is used by:
  - The Latent Decoder (via unmerge_with_source) to broadcast K tokens -> L.
  - The AMTM objective (via compute_importance_mask) to derive importance-
    weighted masking probabilities.

Implementation follows the same bipartite soft matching as the Local Encoder
but without the window constraint and using the attention K matrix as the
similarity signal (the Latent Encoder has full attention, so K is meaningful).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor

from .source_matrix import init_source_matrix, update_source_matrix, unmerge_with_source


def global_bipartite_merge(
    x: Tensor,
    keys: Tensor,
    r: int,
) -> tuple[Tensor, Tensor]:
    """
    Apply one round of global (full-sequence) bipartite soft matching.

    Args:
        x:    [B, L, D]  token embeddings to merge
        keys: [B, L, D_k] attention key vectors used as the similarity signal
        r:    int  number of token pairs to merge  (r ≤ L // 2)

    Returns:
        x_merged: [B, L - r, D]    merged token embeddings
        src_matrix_step: [B, L-r, L]  source matrix for this single merge step
                         (maps new token indices -> original L positions)
    """
    B, L, D = x.shape

    if r <= 0:
        S_identity = init_source_matrix(B, L, device=x.device, dtype=x.dtype)
        return x, S_identity

    # Normalise keys for cosine similarity
    k = F.normalize(keys, dim=-1)   # [B, L, D_k]

    # Bipartite split: A = even indices, B = odd indices
    k_a = k[:, ::2, :]    # [B, nA, D_k]
    k_b = k[:, 1::2, :]   # [B, nB, D_k]

    scores = torch.bmm(k_a, k_b.transpose(1, 2))   # [B, nA, nB]

    node_max, node_idx = scores.max(dim=-1)          # [B, nA]
    edge_idx = node_max.argsort(dim=-1, descending=True).unsqueeze(-1)  # [B, nA, 1]

    src_idx = edge_idx[:, :r, :]     # [B, r, 1]
    unm_idx = edge_idx[:, r:, :]     # [B, nA-r, 1]
    dst_idx = node_idx.unsqueeze(-1).gather(1, src_idx)  # [B, r, 1]
    unm_idx = unm_idx.sort(dim=1)[0]

    # Merge embeddings (sum — soft merge, differentiable)
    x_merged = _global_merge_tokens(x, src_idx, dst_idx, unm_idx)

    # Build the source matrix for this step starting from identity over L tokens
    S_step = init_source_matrix(B, L, device=x.device, dtype=x.dtype)
    S_step = update_source_matrix(S_step, src_idx, dst_idx, unm_idx, n_A=L // 2)

    return x_merged, S_step


def _global_merge_tokens(
    x: Tensor,
    src_idx: Tensor,
    dst_idx: Tensor,
    unm_idx: Tensor,
) -> Tensor:
    B, N, D = x.shape
    r = src_idx.shape[1]
    n_unm = unm_idx.shape[1]

    src_side = x[:, ::2, :]
    dst_side = x[:, 1::2, :]

    unm = src_side.gather(1, unm_idx.expand(B, n_unm, D))
    src = src_side.gather(1, src_idx.expand(B, r, D))
    dst_new = dst_side.scatter_add(1, dst_idx.expand(B, r, D), src)

    return torch.cat([unm, dst_new], dim=1)


def compute_importance_mask(S_prime: Tensor, L: int, K: int) -> Tensor:
    """
    Derive AMTM masking probabilities from the Latent Encoder source matrix.

    Tokens grouped into *large* latent clusters are low-information (high
    redundancy) -> low masking probability.
    Tokens that survived as singletons or small clusters are high-information
    -> high masking probability.

    Implements the weighting in Section 3.4 of the MergeDNA paper:
        g_i  = #{local tokens grouped into latent token i}
        w_i  = 1 / g_i
        P_L(j) ∝ w_i / g_i   for each j belonging to group i

    Args:
        S_prime: [B, K, L]  source matrix from the Latent Encoder ToMe step.
                             S_prime[b, i, j] = 1 means local token j was
                             assigned to latent token i in batch element b.
        L:       int  number of local tokens (columns of S_prime)
        K:       int  number of tokens to mask (also number of latent tokens)

    Returns:
        mask_indices: [B, K]  long tensor of local-token indices to mask,
                              sampled without replacement according to P_L.
    """
    B = S_prime.shape[0]

    # Group sizes:  g[b, i] = number of local tokens in latent group i
    g = S_prime.sum(dim=-1).clamp(min=1.0)   # [B, K]

    # Weight per latent group:  w[b, i] = 1 / g_i^2
    # (the paper writes w_i / g_i where w_i = 1/g_i, so overall 1/g_i^2)
    w = 1.0 / (g * g)   # [B, K]  — relative weight per group

    # Distribute group weight evenly across its member local tokens
    # P_L[b, j] = w[b, i(j)] / g[b, i(j)]  for the group i that owns token j
    # Implemented as: expand group weights to each member via S_prime^T @ (w/g)
    per_member_w = w / g   # [B, K]   weight each member of group i receives
    # Broadcast: P_raw[b, j] = Σ_i S_prime[b,i,j] * per_member_w[b,i]
    P_raw = torch.bmm(S_prime.transpose(1, 2), per_member_w.unsqueeze(-1)).squeeze(-1)  # [B, L]

    # Normalise to a proper probability distribution
    P_L = P_raw / P_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)   # [B, L]

    # Sample K indices without replacement according to P_L
    mask_indices = torch.multinomial(P_L, num_samples=K, replacement=False)   # [B, K]
    return mask_indices


def expand_token_mask_to_bases(
    mask_token_indices: Tensor,
    S_local: Tensor,
) -> Tensor:
    """
    Expand a mask over local tokens to a mask over original base positions
    
    When a local token is selected for masking, all of its constituent bases
    (tracked by the Local Encoder source matrix S_local) are also masked.

    Args:
        mask_token_indices: [B, K]  indices of local tokens to mask
        S_local:            [B, L, N]  Local Encoder source matrix
    Returns:
        M_N: [B, N]  boolean mask (True = masked)
    """
    B, L, N = S_local.shape
    K = mask_token_indices.shape[1]

    # Gather the source-matrix rows for masked tokens
    # masked_rows[b, k, :] = S_local[b, mask_token_indices[b, k], :]
    idx = mask_token_indices.unsqueeze(-1).expand(B, K, N)   # [B, K, N]
    masked_rows = S_local.gather(1, idx)   # [B, K, N]

    # Union: a base is masked if it belongs to any selected token
    M_N = masked_rows.sum(dim=1) > 0   # [B, N]
    return M_N
