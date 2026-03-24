# Spectrogram Transformer (v2 Phase 1) — Handoff

**Module:** `usv_language/models/` + `usv_language/training/`
**Author:** Claude Opus 4.6
**Date:** 2026-02-20
**Tests:** 11/11 passing (130 total suite)

---

## What Was Built

Autoregressive spectrogram transformer for next-column prediction on USV bout spectrograms. This is Phase 1 of v2 architecture — the transformer develops rich internal representations that Phase 2 will analyze via VQ-VAE.

### Files Created (6)

| File | Purpose | Lines |
|------|---------|-------|
| `usv_language/models/__init__.py` | Package init, re-exports | 13 |
| `usv_language/models/transformer.py` | TransformerConfig + TransformerBlock + SpectrogramTransformer | ~210 |
| `usv_language/training/__init__.py` | Package init | 1 |
| `usv_language/training/train_transformer.py` | CLI training script with full pipeline | ~370 |
| `usv_language/training/extract_hidden_states.py` | Hidden state extraction for Phase 2 | ~180 |
| `usv_language/tests/test_spectrogram_transformer.py` | 11 test cases | ~260 |

### Architecture

- **Pre-norm transformer** (LayerNorm before attention/FFN, not after)
- **Causal masking**: `torch.triu(ones, diagonal=1).bool()` registered as buffer, sliced to actual seq_len
- **Learned positional embeddings**: `nn.Embedding(512, 512)`
- **Input projection**: Linear(170→512) + GELU + LayerNorm
- **Output head**: LayerNorm + Linear(512→170)
- **~25.6M parameters** (target was 25-30M)

### Key Design Decisions

1. **CosineWarmupScheduler copied, not imported** from v1 trainer — avoids tight coupling between v1 VQ-VAE and v2 transformer training paths. Same algorithm, independent lifecycle.

2. **Mask convention**: Dataset returns `mask` with 1=real/0=pad. Model expects `attention_mask` with True=pad (PyTorch's `key_padding_mask` convention). Conversion happens in training script: `padding_mask = ~batch["mask"].bool()`.

3. **masked_mse_loss**: MSE weighted by mask, normalized by count of valid elements × n_freq. Returns 0.0 when no valid frames (edge case).

4. **Checkpoint format** includes config object directly (pickle-safe frozen dataclass), enabling model reconstruction from checkpoint alone.

5. **Hidden state extraction** uses overlap_ratio=0.0 (unlike training which uses 0.5) to avoid duplicate frames in the output arrays.

### Integration Points

- **Input**: `USVBoutDataset` from `usv_language/data/dataset.py` — returns `{input, target, mask, recording_id}`
- **Output of training**: `.pt` checkpoint files (best, periodic, final)
- **Output of extraction**: `hidden_states_layer{N}.npy` + `metadata.json` — consumed by Phase 2 VQ-VAE

### Test Coverage

| # | Test | What It Verifies |
|---|------|-----------------|
| 1 | Forward pass shape | (4,128,170) → (4,128,170), hidden_states=None |
| 2 | Causal mask | Modifying future position doesn't change past outputs |
| 3 | Parameter count | 20M < params < 35M |
| 4 | Single-batch overfit | 50 steps reduces loss by >50% |
| 5 | Hidden state shapes | n_layers tensors of (B, S, d_model) |
| 6 | Padding mask + causal | Padded positions produce zero gradients |
| 7 | Checkpoint save/resume | Model, optimizer, scheduler state preserved |
| 8 | Gradient clipping | Post-clip norm ≤ max_norm |
| 9-11 | Config validation | d_model%n_heads, negative n_freq, dropout range |

### Known Limitations

- **No DDP support yet**: `--distributed` flag not implemented (only `--data-parallel` works). DDP requires `torch.distributed.launch` wrapper.
- **Data loading assumes HDF5 or .npy format**: The `load_bout_spectrograms` function expects either `bout_spectrograms.h5` or `*_bout*.npy` files. The bout pipeline needs to produce these.
- **No per-frequency error visualization**: The plan mentioned per-freq-bin error logging every 10 epochs — deferred to training time when we have real data.

### Reviewer Notes

- No changes to existing files (pure addition)
- All imports use the public API from `usv_language/data/`
- No STFT parameter changes (model is downstream of spectrogram computation)
