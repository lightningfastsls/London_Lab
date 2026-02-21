# Hidden State VQ-VAE (Phase 8.3) Module Review

**Reviewed by:** Master Reviewer (Opus 4.6)
**Date:** 2026-02-20
**Module:** `usv_language/models/vqvae.py` + `usv_language/training/train_vqvae.py` + `usv_language/training/compare_layers.py`
**Tier:** 3 (Critical — ML model + training pipeline)
**Verdict:** CHANGES NEEDED

---

## WARNINGS (no blockers)

### W1. `--seed` parsed but never applied — experiments non-reproducible

**File:** `usv_language/training/train_vqvae.py:592` (argument), `train()` function (missing seed logic)
**Problem:** `parse_args()` includes `--seed` with default 42, but no call to `torch.manual_seed()` or `np.random.seed()` exists in `train()`. K-means++ seeding uses `torch.randint`, dataset windowing uses `np.random.choice`, and DataLoader shuffles randomly. Two runs with `--seed 42` produce different results, making experiments unreproducible.
**Fix:** Add at the beginning of `train()`:
```python
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
```
Also forward `--seed` in `compare_layers.py`'s `train_argv` list.

### W2. Dead code reset sets `ema_cluster_size = 0`, creating oscillation

**File:** `usv_language/models/vqvae.py:266`
**Problem:** After resetting a dead code, `_reset_dead_codes` sets `self.ema_cluster_size[dead_mask] = 0`. Since the dead check is `ema_cluster_size < dead_code_threshold` (2.0), any newly reset code is immediately "dead" again next batch. EMA update: `0.99 * 0 + 0.01 * 1 = 0.01` — still far below 2.0. This creates oscillation where the same codes get reset every batch, wasting initialization effort and preventing codebook stabilization.
**Fix:** Change to:
```python
self.ema_cluster_size[dead_mask] = self.dead_code_threshold
```
This gives newly reset codes one grace period to accumulate assignments.

### W3. K-means seeding uses D, not D² — mislabeled as "k-means++"

**File:** `usv_language/models/vqvae.py:302-305`
**Problem:** Selection probabilities use linear distance `D(x)` instead of squared distance `D(x)²`. True k-means++ (Arthur & Vassilvitskii 2007) requires D² weighting for its O(log K) competitive ratio guarantee. With linear weighting, the theoretical guarantee is lost. For K=64, N=5000 the practical impact is small, but the scientific claim is inaccurate.
**Fix:** Change:
```python
probs = dists / (dists.sum() + 1e-10)
```
to:
```python
probs = dists.pow(2) / (dists.pow(2).sum() + 1e-10)
```
Or correct the docstring if linear variant is intentional.

### W4. `compare_layers.py` has zero automated test coverage

**File:** `usv_language/tests/test_hidden_state_vqvae.py` — no `TestCompareLayersReport` exists
**Problem:** `generate_report()` and `score_layer()` are untested. The ROADMAP test plan item 8 requires "Multi-layer comparison produces report with all expected metrics." The exit criterion is verified only by manual inspection in the handoff.
**Fix:** Add `TestCompareLayers` class with:
1. `test_score_layer_ranking` — synthetic metric dicts, verify ranking correctness
2. `test_generate_report_4_layers` — verify report file contains all 4 layers, Metrics Table header, Recommendation section

### W5. Default `batch_size=64` deviates from ROADMAP spec of 256

**Files:** `usv_language/training/train_vqvae.py:579`, `usv_language/training/compare_layers.py:224`
**Problem:** ROADMAP Phase 8.3 specifies "Batch size: 256 (large batches OK — inputs are just 512-dim vectors)". Both scripts default to 64. Smaller batches mean noisier EMA codebook updates and slower training.
**Fix:** Change default to 256 in both files.

### W6. Missing config validation for `commitment_weight` and `dead_code_threshold`

**File:** `usv_language/models/vqvae.py`, `VQVAEConfig.__post_init__()` lines 66-79
**Problem:** Negative values silently accepted. `commitment_weight=-1.0` inverts the loss; `dead_code_threshold=-1.0` silently disables dead code reset. Existing validation covers `d_model`, `codebook_size`, `codebook_dim`, `ema_decay`, `conv_kernel_size` but omits these two.
**Fix:** Add:
```python
if self.commitment_weight < 0:
    raise ValueError(f"commitment_weight must be >= 0, got {self.commitment_weight}")
if self.dead_code_threshold < 0:
    raise ValueError(f"dead_code_threshold must be >= 0, got {self.dead_code_threshold}")
```

### W7. `IMPLEMENTATION_PROGRESS.md` not updated for Phase 8.3

**File:** `IMPLEMENTATION_PROGRESS.md`
**Problem:** Handoff explicitly notes this needs updating. Violates State Update Rule — stale progress files cause next session to treat completed work as pending.
**Fix:** Add dated entry marking Phase 8.3 as complete.

---

## SUGGESTIONS

| # | Issue | File | Fix |
|---|-------|------|-----|
| S1 | `docs/modules/vqvae.md` missing | N/A | Create module doc following spectrogram-transformer pattern |
| S2 | K-means++ EMA buffer init semantic inconsistency | `vqvae.py` | `ema_embedding_sum` set to unit vectors after init, not actual sums. Self-correcting via `F.normalize()` but should be documented. |
| S3 | `compare_layers.py` doesn't forward all hyperparameters | `compare_layers.py` | Missing `--commitment-weight`, `--ema-decay`, `--stride`, `--window-size`. Document limitation. |
| S4 | Overfit test doesn't test production bottleneck ratio | `test_hidden_state_vqvae.py` | Uses d_model=64, codebook_dim=64 (1:1). Production is 512:64 (8:1). Add comment documenting gap. |

---

## What Passed

| Area | Verdict | Notes |
|------|---------|-------|
| ROADMAP spec compliance | PASS | All exit criteria functionally met; architecture matches spec |
| ADR constraints | PASS | ADR-007 two-phase architecture fully respected |
| DSP / ML correctness | PASS | STE gradient flow correct, EMA updates correct, L2 normalization applied correctly |
| Integration patterns | PASS | Pattern 1 (frozen config), Pattern 3 (synthetic fixtures), Pattern 8 (import bootstrap) all followed |
| Test quality | PASS | 21/21 tests meaningful; overfit, codebook utilization, gradient flow, dead code reset all covered |
| ML rigor | PASS | Sequential val split prevents temporal leakage; codebook excluded from optimizer |

---

## Handoff Uncertainties Assessment

| Uncertainty | Verdict | Notes |
|-------------|---------|-------|
| Overfit test vs production bottleneck | ACCEPTABLE | Tests fundamental property; production bottleneck is Phase 11 |
| Dead code threshold interpretation | ISSUE | See W2 — oscillation risk from reset to 0 |
| K-means++ OOM risk | ACCEPTABLE | N=5000, K=64 → 320K entries, trivial |
| compare_layers hyperparameter forwarding | ACCEPTABLE | Deliberate controlled comparison design |

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc (`docs/modules/vqvae.md`) | MISSING | S1 — needs creation |
| `docs/architecture/patterns.md` | UP TO DATE | No new patterns; Conv1d transposition and EMA codebook are standard |
| `DECISIONS.md` | UP TO DATE | ADR-007 covers two-phase architecture |
| `IMPLEMENTATION_PROGRESS.md` | NOT UPDATED | W7 — needs Phase 8.3 entry |

---

## Summary

| Severity | Count | Items |
|----------|-------|-------|
| BLOCKER | 0 | — |
| WARNING | 7 | W1 (seed), W2 (dead code oscillation), W3 (k-means D²), W4 (compare_layers tests), W5 (batch_size), W6 (config validation), W7 (progress doc) |
| SUGGESTION | 4 | S1 (module doc), S2 (EMA init semantics), S3 (hyperparameter forwarding), S4 (overfit test gap) |

---

## Verdict

**CHANGES NEEDED** -> **ALL WARNINGS FIXED** (2026-02-20)

All 7 warnings resolved. 158 tests passing (28 module + 130 existing). See Fix Log below for details.

~~No blockers. All 151 tests pass and all ROADMAP exit criteria are functionally met. Priority order for fixes:~~

~~1. **W1** — Apply seed in `train()` for reproducibility~~
~~2. **W2** — Dead code reset to `threshold` instead of 0~~
~~3. **W3** — Fix to true k-means++ (D²) or correct docstring~~
~~4. **W4** — Add 2 unit tests for `score_layer` and `generate_report`~~
~~5. **W6** — Add bounds checks for `commitment_weight` and `dead_code_threshold`~~
~~6. **W7** — Update `IMPLEMENTATION_PROGRESS.md`~~
~~7. **W5** — Change batch_size default from 64 to 256~~

---

## Fix Log

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| W1 | FIXED | `train_vqvae.py`, `compare_layers.py` | 2026-02-20 | Added `torch.manual_seed`, `np.random.seed`, `cuda.manual_seed_all` at top of `train()`. Forwarded `--seed` in `compare_layers.py` `train_argv` and `parse_args`. |
| W2 | FIXED | `vqvae.py:266` | 2026-02-20 | Changed `ema_cluster_size[dead_mask] = 0` to `= self.dead_code_threshold`. Newly reset codes now get grace period. |
| W3 | FIXED | `vqvae.py:304` | 2026-02-20 | Changed `dists / sum` to `dists.pow(2) / sum` for true k-means++ D² weighting per Arthur & Vassilvitskii 2007. |
| W4 | FIXED | `test_hidden_state_vqvae.py` | 2026-02-20 | Added `TestCompareLayers` with 3 tests: `test_score_layer_ranking`, `test_score_layer_range`, `test_generate_report_4_layers`. |
| W5 | FIXED | `train_vqvae.py`, `compare_layers.py` | 2026-02-20 | Changed default `--batch-size` from 64 to 256 in both files per ROADMAP spec. |
| W6 | FIXED | `vqvae.py` `__post_init__` | 2026-02-20 | Added non-negative validation for `commitment_weight` and `dead_code_threshold`. Added 4 tests in `TestConfigValidationExtended`. |
| W7 | FIXED | `IMPLEMENTATION_PROGRESS.md` | 2026-02-20 | Added Phase 8.3 entry. Also updated ROADMAP.md (DONE), ops/goals.md (Completed). |
| S1 | DEFERRED | | | Module doc |
| S2 | DEFERRED | | | EMA init docstring |
| S3 | DEFERRED | | | Hyperparameter forwarding docs |
| S4 | DEFERRED | | | Overfit test comment |

**Post-fix test results:** 158 passed in 8.29s (28 module tests + 130 existing)
