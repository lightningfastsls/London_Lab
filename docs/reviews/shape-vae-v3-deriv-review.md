# Shape VAE V3 Derivative Loss Module Review

**Date:** 2026-05-27
**Reviewer:** master-reviewer
**Modules:** `scripts/experiments/train_shape_vae_v3_deriv.py`, `scripts/experiments/extract_ridge_targets_v3.py`, `scripts/eval_shape_vae_v3.py`
**Spec:** `docs/handoffs/2026-05-27_shape-vae-A-derivative-loss.md` + `PLAN_geometric_shape_clustering_vae.md §3`
**Review Tier:** Tier 3 (DSP math, differentiable loss assembly, ML training pipeline)
**Tests at review time:** 38 passed, 5 skipped (rig-only)

---

## Verdict (initial review): CHANGES NEEDED — 2 BLOCKERS

### BLOCKER-1 — Shape mismatch in `_run_epoch` crashes at the first training batch
`_run_epoch` called `total_shape_vae_loss(recon_band, x if False else x, ...)`. `recon_band` is the
band crop `(B,1,170,234)` of the decoded output; the second argument was the full padded input
`(B,1,256,256)`. Inside the loss, `F.binary_cross_entropy(xr, x)` gets `(B,1,170,234)` vs
`(B,1,256,256)` → `ValueError` on the first batch. Training never starts.
- The `x if False else x` expression always evaluated to `x` — an unresolved placeholder.
- **Why tests missed it:** `TestTotalShapeVaeLoss` builds `x_recon` and `x` with the *same* shape,
  never the realistic band-crop-vs-full-image case.

### BLOCKER-2 — `_run_epoch` forward path is untested
No test instantiated a real `ImageVAE`, ran the forward pass, cropped the band region, and called
`total_shape_vae_loss`. The shape-mismatch class of bug was structurally invisible to the suite.

---

## Correctness questions answered (all PASS)

1. **Ridge alignment** — time-column count (T-1 both sides), frequency units (kHz both sides), and
   patch index order (forward iteration in extraction; `Subset` preserves order; `RidgePatchDataset`
   indexes base and cache by the same `i`) are all aligned. No off-by-one, no transposed axis.
2. **The `x if False else x` placeholder** — confirmed BLOCKER-1.
3. **soft-argmax freq axis** — `freq_khz` length (band `f_in`) matches `deriv_img` H. No broadcast error.
4. **`load_ridge_cache` / `RidgePatchDataset`** — `__init__` asserts `len(base) == cache N`; aligned.
5. **NaN/inf at low tau** — `torch.softmax` subtracts the per-column max; numerically stable. No risk.

## Constraint compliance (all PASS)
- Frozen baseline `train_contour_vae_v2.py` untouched (git clean).
- Corpus constants imported via `_compute_band_slice` / `ridge_tracker`, never redeclared.
- `beta` defaults to 0.1 (not the dead-end's 1.0).
- Print discipline: both mains print `[params]` JSON + per-step metrics, `flush=True`.
- No augmentation (Pathway A keeps invariance to the derivative term only).

## Float32 tolerance relaxation — SOUND, not greenwashing
Two formula-identity tests use `math.isclose(rel_tol=1e-5)` instead of `abs < 1e-5`. At magnitude
~3700 a float32 ULP ≈ 4.4e-4, so an absolute 1e-5 is ~44× tighter than representable. The asserted
identity is unchanged; only the tolerance is made float32-aware. Approved by the user under the
Test-Protocol "Correct code / Fail test" rule.

## Lower-tier findings
- **SHOULD-FIX-1** — `val_idx` not saved; eval-on-held-out is fragile. Save `val_idx.npy`.
- **NIT-1** — remove the `x if False else x` placeholder (subsumed by BLOCKER-1 fix).
- **NIT-2** — eval gate 4 (jump capture) not automated; acceptable (handoff labels it qualitative).

---

## Fixes Applied (2026-05-27, implementer)

All fixes verified: **41 passed, 5 skipped, 0 failed**; all 3 scripts `py_compile`; baseline git-clean.

### BLOCKER-1 — FIXED via Option B (full recon + separate derivative input)
Chose Option B over the reviewer-suggested Option A deliberately, for **attribution cleanliness**
(the handoff's core requirement): recon stays byte-identical to the frozen baseline so `λ_d=0`
reduces to the dead-end exactly and any shape-η² movement is attributable to the derivative term alone.
- `train_shape_vae_v3_deriv.py` `total_shape_vae_loss`: added optional `deriv_img` param. Recon (BCE)
  runs on `x_recon` vs `x` (full images); the derivative runs on `deriv_img` (band crop) when given,
  else on `x_recon` (preserves all isolated-unit-test calls). Docstring updated.
- `train_shape_vae_v3_deriv.py` `_run_epoch`: now passes `x_recon` (full) as recon, `x` (full) as
  target, and `deriv_img=recon_band` (band crop). Placeholder removed (NIT-1).

### BLOCKER-2 — FIXED (new integration coverage)
- `tests/test_train_shape_vae_v3_deriv.py` `TestRunEpochForwardPath` (3 tests):
  `test_run_epoch_path_no_crash_and_finite` (real ImageVAE forward → band crop → loss, finite),
  `test_run_epoch_path_backward_flows` (gradients reach model params, finite), and
  `test_recon_target_shape_mismatch_raises` (regression guard: the old band-as-x_recon vs full-target
  call must raise).

### SHOULD-FIX-1 — FIXED
- `train_shape_vae_v3_deriv.py` `main`: saves `val_idx.npy` + `train_idx.npy` to the output dir and
  prints the split sizes, so `eval_shape_vae_v3` can score on the held-out rows.

### NIT-2 — DEFERRED (per handoff: gate 4 is qualitative; revisit when `syllable_type` is confirmed).

**Re-review of the BLOCKER fixes requested from a fresh reviewer** (self-verification insufficient for blockers).

---

## Re-Review: BLOCKER Verification (2026-05-27, independent re-reviewer)

**Test run:** `pytest tests/test_train_shape_vae_v3_deriv.py -q` → **32 passed, 0 failed, 0 skipped**
(this module = 32 tests; the "41 passed" was the combined train+eval count).

- **BLOCKER-1 — RESOLVED.** `deriv_img` param added; BCE recon on full `x_recon` vs `x`; derivative
  on `deriv_img=recon_band` with `freq_khz` length matching the band dim. **Attribution property
  verified:** with `lambda_d=lambda_c=0`, `dloss["total"]=0`, so `total = recon + beta*kl` with the
  recon formula mathematically identical to the frozen baseline — holds on the production full-image
  path, not just the matched-shape unit test.
- **BLOCKER-2 — RESOLVED.** `TestRunEpochForwardPath` exercises the real forward→band-crop→loss
  composition (finite loss + backward flows), and `test_recon_target_shape_mismatch_raises` is a
  genuine regression guard (verified to raise `ValueError`, not a tautology).
- **SHOULD-FIX-1 — RESOLVED.** `val_idx.npy`/`train_idx.npy` saved after `out_dir.mkdir`, in scope.
- **Regressions — NONE.** `deriv_img=None` default preserves all isolated unit tests; dtype/device
  inherited from `x_recon`.

### Final Verdict: APPROVED — all blockers resolved, no regressions, safe to proceed.
