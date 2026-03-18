"""
DNA input embedding

Converts raw nucleotide sequences into dense embeddings for the Local Encoder

Encoding:
  A -> 0,  C -> 1,  T -> 2,  G -> 3,  N (unknown) -> 4
Input token IDs are projected through a learned embedding table into R^D,
and a learnable positional bias is added via RoPE in each attention layer
(not added here — this module only handles the token embedding).
"""

import torch
import torch.nn as nn
from torch import Tensor

# Nucleotide vocabulary: A C T G N(unknown)
VOCAB = {"A": 0, "C": 1, "T": 2, "G": 3, "N": 4}
VOCAB_SIZE = 5  # 4 bases + N


class DNAEmbedding(nn.Module):
    """
    Maps integer-encoded nucleotide sequences to dense vectors of dimension D.

    Args:
        d_model: Embedding (and model) dimension. Paper uses 1024.
        vocab_size: Number of distinct input tokens (5: A, C, T, G, N).
        padding_idx: Token index used for padding (masked out in gradients).
    """

    def __init__(self, d_model: int = 1024, vocab_size: int = VOCAB_SIZE, padding_idx: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # Scale by sqrt(d_model) following standard transformer practice
        self._scale = d_model ** 0.5

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: integer token IDs
        Returns:
            [B - batch size, N - seq length, D - embedding dim] float embeddings
        """
        return self.embed(x) * self._scale


def encode_sequence(seq: str) -> Tensor:
    """
    Convert a DNA string to a 1-D integer tensor.

    Unknown characters (anything not in ACTG or N) treated as N.

    Args:
        seq: e.g "ACTGACTGNNAC"
    Returns:
        LongTensor of shape [len(seq)]
    """
    ids = [VOCAB.get(c.upper(), VOCAB["N"]) for c in seq]
    # ids = [VOCAB.get(c) for c in seq]
    return torch.tensor(ids, dtype=torch.long)


def encode_batch(seqs: list[str], max_len: int | None = None, pad_id: int = 0) -> tuple[Tensor, Tensor]:
    """
    Encode a list of variable-length DNA strings into a padded batch.

    Args:
        seqs:    list of DNA strings
        max_len: if not None, padding all sequences to this length, if None, max_len=longest sequence
        pad_id:  Token ID used for padding positions
    Returns:
        ids:      LongTensor [B, L] of token IDs.
        lengths:  LongTensor [B] with the original (unpadded) length of each sequence
    """
    encoded = [encode_sequence(s) for s in seqs]
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    L = max_len if max_len is not None else int(lengths.max().item())

    batch = torch.full((len(seqs), L), fill_value=pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        n = min(len(e), L)
        batch[i, :n] = e[:n]

    return batch, lengths
