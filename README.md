# MergeDNA Implementation

**A transformer-based model for efficient DNA sequence representation learning via hierarchical token merging.**

Token merging is an efficient tokenization method implemented in Transformers. Li et al., utilise the dynamic tokenisation method for DNA sequence encoding in their proposed foundation model MergeDNA. 

This repository implements MergeDNA, a novel architecture that compresses DNA sequences while maintaining interpretability through differentiable token merging. The model employs a three-stage pre-training objective combining reconstruction and masked token modeling.

## References
This is just an implementation from scratch of the awesome work by the authors below. Please refer to these papers for more information.
- **MergeDNA Paper**: Li et al., "MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging" [https://arxiv.org/abs/2511.14806]
- **Token Merging (ToMe)**: Bolya et al., "Token Merging for Fast Stable Diffusion" (ICCV 2023) [https://arxiv.org/abs/2210.09461]

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pre-training](#pre-training)
- [Inference](#inference)
- [Components](#components)
---

## Overview

**MergeDNA** addresses the computational bottleneck of processing long DNA sequences through:

1. **Hierarchical Compression**: Local encoder progressively merges similar adjacent tokens, reducing sequence length by ~50% with full traceability
2. **Three-Stage Pre-training**:
   - **MTR** (Merged Token Reconstruction): Reconstruct compressed sequences
   - **Latent MTR**: Reconstruct after global dimensionality reduction
   - **AMTM** (Adaptive Masked Token Modeling): Predict importance-weighted masked bases
3. **Interpretability**: Source matrices (`S`) track which original bases are merged into each token, enabling explainability
4. **Flexible Inference**: 
   - **Encoder-only** for efficient classification/regression
   - **Full autoencoder** for base-resolution reconstruction tasks

### Key Features

- ✅ **Token Merging (ToMe)**: Soft bipartite matching for differentiable token compression
- ✅ **Learnable Compression**: Stochastic ratio sampling during training (target: L/N = 0.5)
- ✅ **Multi-scale Processing**: Local (windowed) attention + latent (global) attention
- ✅ **Attention-based Masking**: AMTM uses attention patterns to select importance-weighted masks
- ✅ **RoPE Positional Encoding**: Rotary position embeddings for efficient attention
- ✅ **Pre-training Objectives**: Combined loss across 3 forward passes
- ✅ **Inference Utilities**: Built-in `encode()`, `encode_pooled()`, and `reconstruct()` methods

---

## Architecture

### Model Components

```
Input DNA Sequence
    ↓
[Embedding] → Dense token embeddings
    ↓
[Local Encoder] → Windowed attention + local ToMe merging
    ↓ (L tokens, source matrix S)
[Latent Encoder] → Global attention + global ToMe (optional)
    ↓ (K latent tokens)
[Latent Decoder] → Reconstruct latent tokens
    ↓
[Local Decoder] → Full resolution reconstruction
    ↓
Output logits (vocabulary distribution per base)
```

### Pre-training Passes

| Pass | Component | Objective | Loss Weight |
|------|-----------|-----------|-------------|
| **1** | Local Enc → Local Dec | Reconstruct from merged tokens | 1.0 |
| **2** | Local Enc → Latent Enc/Dec → Local Dec | Reconstruct after global ToMe | 0.25 |
| **3** | Latent Enc + mask | Predict masked bases (importance-weighted) | 1.0 |

**Combined Loss**: `L_total = L_MTR + 0.25 * L_LatentMTR + L_AMTM`

### Configuration

Default configuration (from `MergeDNAConfig`):
```python
d_model = 1024              # Embedding and hidden dimensions
n_heads = 16                # Number of attention heads
local_enc_layers = 4        # Stacked local encoders
latent_enc_layers = 20      # Latent encoder depth
latent_dec_layers = 4       # Latent decoder depth
local_dec_layers = 4        # Local decoder layers
window_size = 16            # Local attention window
d_group = 16                # Group normalization dimension
dropout = 0.1               # Dropout rate
use_flash = True            # Flash attention (if available)
```

---

## Repository Structure

```
MergeDNA/
├── README.md                          # This file
├── MergeDNA_Implementation_Plan.pdf   # Detailed design specification
│
├── notebooks/
│   └── mergedna_demo.ipynb           # End-to-end walkthrough notebook
│       ├── DNA encoding
│       ├── Forward pass through each component
│       ├── Source matrix visualization
│       ├── All 3 pre-training passes
│       ├── Mini training loop (loss convergence)
│       ├── Inference helpers (encode, reconstruct)
│       ├── Cosine similarity analysis
│       └── Compression ratio distribution
│
└── src/
    ├── __init__.py
    │
    ├── model/
    │   ├── __init__.py
    │   ├── embedding.py              # DNA input embedding (A/C/T/G/N → dense vectors)
    │   ├── local_encoder.py          # Windowed attention + local token merging (ToMe)
    │   ├── latent_encoder.py         # Global attention over merged tokens
    │   ├── latent_decoder.py         # Reconstruct latent representations
    │   ├── local_decoder.py          # Full-resolution reconstruction decoder
    │   ├── mergedna.py               # Full model assembly + pre-training objectives
    │   ├── transformer.py            # Reusable attention, FFN, normalization layers
    │   │
    │   └── tome/                     # Token Merging (ToMe) utilities
    │       ├── __init__.py
    │       ├── bipartite_matching.py # Soft bipartite matching algorithm
    │       ├── source_matrix.py      # Track merged base→token relationships (matrix S)
    │       ├── local_tome.py         # Local window-based ToMe
    │       ├── global_tome.py        # Global attention-based ToMe + AMTM masking
    │       └── ...
    │
    ├── data/
    │   ├── __init__.py
    │   └── [datasets and loaders]
    │
    ├── training/
    │   ├── __init__.py
    │   ├── objectives.py             # Pre-training loss functions
    │   └── [training loops]
    │
    └── downstream/
        ├── __init__.py
        └── [fine-tuning, evaluation]
```

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy, SciPy
- (Optional) FlashAttention for faster attention

### Setup

```bash
# Clone repository
git clone https://github.com/shrutisshikhare/MergeDNA.git
cd MergeDNA

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy scipy matplotlib jupyter

# (Optional) Install FlashAttention for faster attention
# pip install flash-attn
```

---

## Quick Start

### 1. Load and Encode DNA Sequences

```python
from src.model.embedding import encode_batch, VOCAB

sequences = [
    "ATCGATCGATCG",
    "GGCCTTAAGGCC",
    "ACGTNNACGT",
]

# Encode sequences to token IDs
token_ids, lengths = encode_batch(sequences)
print(f"Encoded shape: {token_ids.shape}")  # [B, N]
```

### 2. Initialize Model

```python
import torch
from src.model.mergedna import MergeDNA, MergeDNAConfig

# Use default config (1024-dim, 386M params) or reduced for demo
cfg = MergeDNAConfig(
    d_model=128,           # Reduced for demo
    n_heads=4,
    local_enc_layers=4,
    latent_enc_layers=4,
    local_dec_layers=2,
    dropout=0.0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MergeDNA(cfg).to(device)
model.train()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. Forward Pass

```python
# Full forward pass through all components
token_ids = token_ids.to(device)
outputs = model(token_ids)  # [B, N, vocab_size]

# Access intermediate components
z_L, S = model.local_encoder(model.embedding(token_ids))  # Merged tokens + source matrix
print(f"Merged tokens: {z_L.shape}")  # [B, L ≈ N/2, D]
print(f"Source matrix: {S.shape}")    # [B, L, N]
```

### 4. Pre-training

```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Single pre-training step (runs all 3 passes)
losses = model.pretrain_step(token_ids)

# Full training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        out = model.pretrain_step(batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % log_freq == 0:
            print(f"Loss: {out['loss'].item():.4f}")
```

### 5. Inference

```python
model.eval()

# Option 1: Per-token embeddings (for fine-tuning)
z_tokens = model.encode(token_ids)  # [B, L, D]

# Option 2: Sequence-level embeddings (for classification)
z_seq_mean = model.encode_pooled(token_ids, mode="mean")   # [B, D]
z_seq_max = model.encode_pooled(token_ids, mode="max")     # [B, D]
z_seq_cls = model.encode_pooled(token_ids, mode="cls")     # [B, D]

# Option 3: Reconstruction (full resolution)
recon_logits = model.reconstruct(token_ids)  # [B, N, vocab_size]
recon_seq = recon_logits.argmax(dim=-1)      # [B, N]
```

---

## Pre-training

### Data Preparation

Your data should be in one of these formats:

1. **FASTA files**: Standard DNA sequence format
   ```
   >seq1
   ATCGATCGATCG
   >seq2
   GGCCTTAAGGCC
   ```

2. **CSV**: Tab/comma-separated with `sequence` column

3. **HDF5**: For large datasets with metadata

### Training Script

See `notebooks/mergedna_demo.ipynb` for a complete mini training example (30 steps on 10 sequences).

For production:

```python
from torch.utils.data import DataLoader
from src.training.objectives import pretrain_step

# Create your DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, token_ids in enumerate(train_loader):
        token_ids = token_ids.to(device)
        
        # Forward pass (all 3 pre-training passes)
        losses = model.pretrain_step(token_ids)
        
        # Backward pass
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {losses['loss'].item():.4f}")
```

---

## Inference

### Representation Learning (Classification/Regression)

```python
# 1. Encode sequences to embeddings
embeddings = model.encode_pooled(token_ids, mode="mean")  # [B, D=1024]

# 2. Attach classification head
class_head = nn.Linear(cfg.d_model, num_classes)
logits = class_head(embeddings)
pred = logits.argmax(dim=-1)
```

### Base-Level Reconstruction

```python
# Reconstruct full-resolution sequences
logits = model.reconstruct(token_ids)  # [B, N, vocab_size]
predictions = logits.argmax(dim=-1)     # [B, N]

# Compute reconstruction accuracy
accuracy = (predictions == token_ids).float().mean()
```

### Interpretability: Source Matrix Visualization

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    x = model.embedding(token_ids)
    z_L, S = model.local_encoder(x)

# Visualize source matrix: which bases are merged?
plt.imshow(S[0].cpu().numpy(), aspect="auto", cmap="Blues")
plt.xlabel("Original base position (N)")
plt.ylabel("Merged token index (L)")
plt.title("Source Matrix S: Base-to-Token Mapping")
plt.show()
```

---

## Components

### 1. **Embedding** (`src/model/embedding.py`)
- Converts DNA nucleotides (A/C/T/G/N) to dense vectors
- Uses learned embedding table + position encoding (RoPE)

### 2. **Local Encoder** (`src/model/local_encoder.py`)
- Windowed self-attention (window_size=16)
- Differentiable local token merging (ToMe)
- Outputs: compressed tokens (L ≈ N/2) + source matrix S

### 3. **Latent Encoder** (`src/model/latent_encoder.py`)
- Global transformer attention over merged tokens
- Optional global ToMe for further compression (K ≈ L/2)
- Bottleneck representation for pre-training

### 4. **Latent Decoder** (`src/model/latent_decoder.py`)
- Reconstructs latent representations
- Enables "Latent MTR" pre-training objective

### 5. **Local Decoder** (`src/model/local_decoder.py`)
- Reconstructs full-resolution sequences
- Uses source matrix to upsample from L→N
- Outputs: logits for each base

### 6. **Token Merging (ToMe)** (`src/model/tome/`)
- **Bipartite Matching**: Soft assignment matrix for token merging
- **Source Matrix**: Tracks base→token relationships
- **Local ToMe**: Window-based merging in local encoder
- **Global ToMe**: Attention-based merging in latent encoder
- **AMTM**: Adaptive masked token modeling using attention patterns

### 7. **Transformer** (`src/model/transformer.py`)
- Reusable components: MultiHeadAttention, FeedForward, LayerNorm, RoPE
- Configurable for window-based and full attention

---

## Demo Notebook

The `notebooks/mergedna_demo.ipynb` provides:

✅ **Section 1**: DNA encoding with vocabulary  
✅ **Section 2**: Model initialization with reduced config  
✅ **Section 3**: Step-by-step forward pass through each component  
✅ **Section 4**: Source matrix visualization (which bases merge?)  
✅ **Section 5**: Token size distribution  
✅ **Section 6**: Local encoder + latent encoder with global ToMe  
✅ **Section 7**: All 3 pre-training passes  
✅ **Section 8**: Mini training loop (30 steps, loss convergence)  
✅ **Section 9**: Inference helpers (encode, reconstruct)  
✅ **Section 10**: Cosine similarity between sequence embeddings  
✅ **Section 11**: Compression ratio distribution analysis  

**Run the notebook**:
```bash
jupyter notebook notebooks/mergedna_demo.ipynb
```

---

## Configuration

### Model Hyperparameters

```python
from src.model.mergedna import MergeDNAConfig

# Production config (386M parameters)
cfg_large = MergeDNAConfig(
    d_model=1024,
    n_heads=16,
    local_enc_layers=4,
    latent_enc_layers=20,
    latent_dec_layers=4,
    local_dec_layers=4,
)

# Demo config (reduced, CPU-friendly)
cfg_demo = MergeDNAConfig(
    d_model=128,
    n_heads=4,
    local_enc_layers=4,
    latent_enc_layers=4,
    latent_dec_layers=2,
    local_dec_layers=2,
    use_flash=False,
    dropout=0.0,
)
```

---