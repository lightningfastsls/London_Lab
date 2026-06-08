# Module 18.4 (DANN Cage-Invariance) Review

**Date:** 2026-05-26
**Reviewer:** master-reviewer (fresh context, no implementation involvement)
**Spec:** `ROADMAP_lab_cnn_classifier.md §18.4` L580–718
**Handoff read:** `docs/handoffs/2026-05-26_module-18.4-dann-resume.md`
**Review Tier:** 3 (adversarial training + encoder collapse risk) — tier claim confirmed correct

---

## Findings summary

| ID | Severity | Status |
|---|---|---|
| — | BLOCKER | none found |
| MAJOR-1 | `per_dimension_max_cohens_d` false-positive at small n | FIXED |
| MINOR-1 | `IMPLEMENTATION_PROGRESS.md` missing 18.4 entry | FIXED |
| MINOR-2 | no tests for `run_vae_diagnostic_on_encoder.py` | DEFERRED → test-hardener |
| MINOR-3 | cage probe used val split (early-stop selection bias) | FIXED |
| NIT-1 | `per_recording` silently used num_domains=2 | FIXED |
| NIT-2 | checkpoint vs metrics co-location | no action (consistent with v1) |

**Verdict: APPROVED** — no blockers. Math traces (GRL backward, λ schedule endpoints, warm-start key mapping with 0 missing keys, cage-probe freeze, DiagnosticResult direct construction) all verified by direct execution. All five 18.3 NOT-to-touch files provably unmodified; `__init__.py` additive-only. Test expectations not weakened vs the ROADMAP §18.4 test plan (no anti-greenwashing).

---

## MAJOR-1 — per-dimension max Cohen's d false-positive at small n

`scripts/run_vae_diagnostic_on_encoder.py::per_dimension_max_cohens_d` takes the max |d|
over all 512 feature dims; the 0.30 threshold (inherited from `_THRESHOLD_PER_BAND_COHENS_D`,
calibrated for 18.1's ~10 frequency bands) suffers multiple-comparisons inflation. Under the
null (cage-invariant encoder), expected max|d| ≈ 0.34 at n=200 (≈86% chance of exceeding 0.30,
a false FAIL) vs ≈ 0.15 at n=1000. The CLI default `--max-per-cohort 1000` is already safe; the
risk is a spurious VAE-gate FAIL if a user reduces sample count below ~500. Not a code bug; an
operational/documentation gap. Fix before the rig eval to avoid a confusing false-fail.

## MINOR-1 — IMPLEMENTATION_PROGRESS.md missing 18.4 entry
Completion sequence requires a dated entry per module; last entry is 18.3 (2026-05-25).

## MINOR-2 — no tests for run_vae_diagnostic_on_encoder.py
Three novel functions (`pc1_cohens_d_features`, `per_dimension_max_cohens_d`,
`notch_migration_in_feature_space`) are not in the ROADMAP test plan and untested. Verified
correct by manual trace, but regressions would go undetected. Not a spec violation → test-hardener
should add a smoke test covering criterion-3 pass/fail.

## MINOR-3 — cage probe on val split (early-stop selection bias)
`train_lab_classifier_v2.py` built the cage probe from `src_val_ds`, the same split used for
early stopping → mild optimism bias. Use the independent test split instead.

## NIT-1 — per_recording silent fallback
`num_domains = 2 if "2cage" else 2` silently trained a wrong 2-domain model under a
`per_recording` flag (only a warning). Raise instead.

## NIT-2 — checkpoint/metrics co-location
`best.pt` stores state_dict + history + warm_start + lambda_schedule; `metrics.json` separate.
Consistent with v1. No action.

---

## Math / logic trace (verified by direct execution)

| Formula | Expected | Actual |
|---|---|---|
| `GRL.backward(λ=0.5, upstream=[1,2,3])` | `[-0.5,-1.0,-1.5]` | match ✓ |
| `LambdaSchedule.lambda_at(0)` | `0.0` | `0.0` ✓ |
| `LambdaSchedule.lambda_at(50)` (total=50) | `~0.9999` | `0.9999092` ✓ |
| `LambdaSchedule.lambda_at(25)` | `~0.987` | `0.9866` ✓ |
| Warm-start v1 `fc.*` → class_head shape | `(12,512)` | `(12,512)` ✓ |
| Warm-start encoder missing keys | 0 | 0 ✓ |
| `knn_same_cohort_rate` on 512-dim features | DiagnosticResult | works ✓ |
| `per_dimension_max_cohens_d` (n=50, shift=1) | ~1.5 | `1.562` ✓ |
| `pc1_cohens_d_features` (n=50, shift=1) | >0 | `0.331` ✓ |

Additivity confirmed (`git diff --stat`/`git status`): 18.3 files unmodified, `__init__.py`
additive block only, 5 new files untracked. Anti-greenwashing: GRL backward exact, λ endpoints
`<1e-9`/`<1e-3`, cage-probe positive `>0.90` / negative `[0.45,0.55]`, F1 band `≤0.10` — none weakened.

---

## Fixes Applied (2026-05-26, implementor self-verification)

Per the reviewer's instruction (no BLOCKERs → self-verification sufficient, no re-review required):

1. **MAJOR-1 fixed** — `scripts/run_vae_diagnostic_on_encoder.py`:
   - `--max-per-cohort` help string now states the ≥500 requirement and the n=200 vs n=1000 d-inflation numbers.
   - Added a runtime `WARNING` in `main()` when the smaller cohort has < 500 patches.
   - `docs/modules/lab-classifier-v2-dann.md`: added a "Sample-size caveat" blockquote under the VAE criteria table.
   - Verified: the smoke run now emits the warning at n=14.
2. **NIT-1 fixed** — `scripts/train_lab_classifier_v2.py`: `--domain-granularity per_recording` now raises `NotImplementedError` (with a TODO) instead of silently training a 2-domain model. Verified: the flag raises.
3. **MINOR-3 fixed** — `scripts/train_lab_classifier_v2.py`: the linear cage probe now uses `src_test_ds` (independent of early-stopping selection) instead of `src_val_ds`, with an explanatory comment.
4. **MINOR-1 fixed** — appended a dated Module 18.4 entry to `IMPLEMENTATION_PROGRESS.md`.
5. **MINOR-2 deferred** — handed to `test-hardener` to add a `run_diagnostic()` smoke test (criterion-3 pass/fail) alongside the adversarial test pass.

**Re-verification after fixes:** `py_compile` clean on both changed scripts; VAE smoke path
intact (warning fires; sensitivity FAILs unchanged); `per_recording` raises; `tests/classifier/test_dann.py`
+ `test_cage_probe.py` = **23 passed**. No 18.3 file touched by any fix.
