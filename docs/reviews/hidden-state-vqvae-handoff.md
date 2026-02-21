# Implementation Handoff: Hidden State VQ-VAE (Phase 8.3)

**Module:** VQ-VAE on Hidden States
**Review Tier:** 3
**Date:** 2026-02-20
**Branch:** main

## What Changed

- Built a fresh VectorQuantizerV2 with L2-normalized codebook, EMA updates, dead code reset, and k-means++ initialization (4 anti-collapse mechanisms per ROADMAP spec)
- Created HiddenStateVQVAE model with Conv1d encoder, vector quantizer, and linear decoder (~820K params)
- Implemented training CLI with HiddenStateDataset (memory-mapped .npy, overlapping windows), sequential train/val split, codebook excluded from optimizer (EMA-updated)
- Built multi-layer comparison script (trains VQ-VAE on layers 2,4,6,8 sequentially, generates markdown report with weighted scoring)
- 21 tests covering all exit criteria: forward pass shapes, gradient flow, EMA updates, dead code reset, k-means init, single-batch overfit (<0.01), codebook utilization (>50%), checkpointing, dataset windowing

## Files Changed

- `usv_language/models/vqvae.py` (NEW) — VQVAEConfig, VectorQuantizerV2, HiddenStateVQVAE, _Transpose (~340 lines)
- `usv_language/training/train_vqvae.py` (NEW) — HiddenStateDataset, training loop, checkpointing, CLI (~370 lines)
- `usv_language/training/compare_layers.py` (NEW) — Multi-layer comparison with markdown report generation (~210 lines)
- `usv_language/tests/test_hidden_state_vqvae.py` (NEW) — 21 tests across 12 test classes (~280 lines)
- `usv_language/models/__init__.py` (MODIFIED) — Added VQVAEConfig, VectorQuantizerV2, HiddenStateVQVAE exports
- `usv_language/training/__init__.py` (MODIFIED) — Updated docstring with new module descriptions

## Key Decisions Made

1. **Fresh VectorQuantizerV2 instead of importing v1** (`src/model/quantizer.py`). v1 multiplies `commitment_weight * MSE` internally (baked-in beta), lacks L2 normalization, lacks k-means init, and lives in the v1 namespace. V2 returns raw commitment loss for cleaner separation of concerns. Referenced v1 for EMA/distance computation correctness.

2. **L2 normalization on both encoder outputs AND codebook vectors**. Makes nearest-neighbor lookup equivalent to cosine similarity (unit hypersphere). Critical: must re-normalize codebook after each EMA update because the weighted average of unit vectors is NOT a unit vector.

3. **Commitment loss NOT scaled inside quantizer**. VectorQuantizerV2.forward() returns raw MSE; HiddenStateVQVAE.forward() applies `config.commitment_weight * commit_loss`. This matches ROADMAP spec and allows the same quantizer to be reused with different beta values.

4. **Sequential val split** (first 90% train, last 10% val) instead of random split. Prevents temporal leakage — consecutive hidden-state windows from the same recording share autocorrelation, so random split would leak.

5. **Codebook excluded from AdamW optimizer**. The codebook is updated via EMA (exponential moving average) in the forward pass. Including it in the optimizer would create conflicting gradient vs EMA updates. The `build_optimizer()` function filters out `quantizer.embedding` parameters.

6. **K-means++ initialization** (not sklearn KMeans). Implemented from scratch using k-means++ seeding + 20 Lloyd iterations. Avoids sklearn dependency and runs on GPU tensors directly.

7. **`_Transpose` helper module**. Conv1d expects (B, C, S) but our data is (B, S, C). Rather than reshaping in forward(), a small nn.Module wrapper enables clean nn.Sequential composition.

8. **Memory-mapped dataset with explicit `.close()`**. On Windows, numpy mmap holds file handles that prevent temp directory cleanup. Added `close()` method and `__del__` fallback. Tests use `try/finally` to ensure cleanup.

## What I'm Unsure About

1. **Overfit test threshold vs production config**: The overfit test uses `codebook_dim=64` (matching d_model=64) with `use_conv_encoder=False` for easier convergence. The production config uses `codebook_dim=64` with `d_model=512` and Conv1d encoder — this creates a real bottleneck. The test proves the architecture CAN overfit, but the production bottleneck ratio (512->64) is untested for overfit specifically.

2. **Dead code reset threshold interpretation**: v1 uses `usage_count >= threshold` (integer count of consecutive unused passes). V2 uses `ema_cluster_size < threshold` (EMA-smoothed cluster size). These are conceptually different — v2's threshold=2.0 means a code is "dead" if its smoothed assignment count drops below 2.0, which is more sensitive. The ROADMAP says `dead_code_threshold: float = 2.0` without specifying which interpretation.

3. **K-means++ on GPU with large N**: The `initialize_from_data()` method uses `torch.cdist()` which computes an (N, K) distance matrix. With N=5000 and K=64 this is fine, but if someone passes a much larger dataset it could OOM. No guard on input size.

4. **`compare_layers.py` missing some hyperparameter forwarding**: The `train_argv` list in `compare_layers` forwards core hyperparameters (epochs, batch_size, lr, patience, d_model, codebook_size, codebook_dim) but not all training args (e.g., commitment_weight, ema_decay, stride, window_size). These fall back to `train_vqvae` defaults, which is correct for controlled comparison but means users can't override them via the comparison CLI.

## Test Results

```
.\.venv\Scripts\python.exe -m pytest usv_language/tests/test_hidden_state_vqvae.py -v
21 passed in 6.23s

.\.venv\Scripts\python.exe -m pytest usv_language/tests/ -v
151 passed in 9.86s
```

## ROADMAP Exit Criteria Status

- [x] Forward pass on dummy hidden states: correct shapes, valid indices
- [x] Gradient flow verified through straight-through estimator
- [x] Single-batch overfit: reconstruction loss < 0.01
- [x] Codebook utilization > 50% on synthetic data
- [x] Compare_layers script produces comparison table for 4 layers
- [x] All tests pass (151/151)
- [x] py_compile passes on all new files (6/6)

## Docs Written/Updated

- `docs/reviews/hidden-state-vqvae-handoff.md` — this file
- `DECISIONS.md` — no new ADRs needed (ADR-007 already covers two-phase architecture)
- `docs/architecture/patterns.md` — not updated (no new patterns; Conv1d transposition and EMA codebook are standard PyTorch patterns)
- `IMPLEMENTATION_PROGRESS.md` — should be updated to mark 8.3 as DONE
