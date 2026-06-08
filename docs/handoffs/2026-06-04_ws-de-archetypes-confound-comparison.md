# Handoff — WS-D (soft archetypes) + WS-E (confound-robust cohort comparison)

**Date:** 2026-06-04  **Program:** `PLAN_continuum_repertoire_program.md` §2 (WS-D) + §2 (WS-E), Phase 3
**Predecessors (binding):**
- `docs/handoffs/2026-06-04_ws-a-elastic-fpca-implementation.md` — WS-A DONE (GATE A = HYBRID).
- `docs/handoffs/2026-06-04_ws-bc-continuous-grammar-manifold.md` — WS-B/WS-C DONE (this session's predecessor).
**Status:** READY. WS-D and WS-E both consume the settled elastic-FPCA scores; they are independent → run in parallel.

---

## 0. What Phase 2 settled (read before starting)

- **Representation is settled** (WS-A HYBRID): primary coordinate = `models/shape_fpca/elastic_fpca_scores.parquet`
  (67,337 rows; `wav_stem, call_id, cohort, amp_pc1..amp_pc5, phase_pc1..phase_pc3`). Carry soft-DTW
  (`models/shape_kmeans/k20_softdtw.*`) as the complex-family labeled-set metric.
- **WS-C (manifold):** the shape space is **one filled ~5-D blob** — single connected component, intrinsic
  dim ≈ 5 (flat PCA spectrum 0.20×5), zero persistent H1, no detached pocket. **Implication for WS-D:**
  there are *no natural kinds* to recover — archetypes are a **resolution knob over a continuum**, not
  cluster centers. Report robustness across K; a feature that appears at only one K is an artifact.
- **WS-B (grammar):** **timing dominates**; shape-order sequential structure is weak (lab z=2.6 borderline,
  5970 ~1.7× timing). **Implication for WS-E:** when comparing cohorts, the interesting axis is the
  *distribution over shape coordinates*, not sequence grammar.

---

## 1. CRITICAL data gotchas (inherited — do not relearn the hard way)

- **Join:** FPCA scores `(wav_stem, call_id − 1)` == classified_detections `(wav_stem, det_index)`. The −1
  offset is verified (det_index 0-based). Do **NOT** join on `id` (1-based DeepSqueak call number, max 92 ≠
  det_index max 32 — naive merge gives a spurious many-to-one 100%).
- **FPCA `(wav_stem, call_id)` is NON-UNIQUE** within a cohort (5970: 12,098 rows / 7,064 unique) — multiple
  ridge fragments per call. **Dedupe to one row per call before any analysis** (WS-B kept largest |amp_pc1|).
  The scores parquet is row-for-row aligned with the soft-DTW letters parquet → positional attach is safe.
- **Cohort confound:** 3452/9252 are tiny (334/506 rows) and cage-confounded; the +12σ figure in earlier
  briefs is a *raw mean* difference — relative to pooled spread (σ≈16) it is ~+0.7σ. Treat wild-vs-wild as a
  noise floor (cage ≈ biological-unit collinearity → largely unidentifiable).
- **pitch = `principal_freq_hz`; duration = `call_length_s`** (NOT `det_duration_ms`). `mean_power_db`/`tonality`
  are **cage artifacts** — never report as biology without cross-cage calibration.

---

## 2. WS-D — Soft membership: kernel archetypal analysis

**Goal.** Replace the hard K=20 letters with **graded membership**: each call = a convex mixture of a few
extreme archetypes (sharp step, deep valley, flat ramp). Cohorts differ in *where on the simplex* they sit.

**Method.** Kernel archetypal analysis in a GAK / elastic-kernel space (interior archetypes, not just convex
hull). Choose K as a **resolution knob** — triangulate (a) explained-variance elbow, (b) resampling
stability (instability as you tile a smooth region = past useful resolution), (c) interpretability,
(d) downstream invariance. **Report robustness across K.** Given WS-C (filled 5-D blob), expect *no*
privileged K — that is the honest finding.

**Reuses.** WS-A elastic distances → GAK kernel; SEACells kernel-AA code pattern (Persad 2023). GAK from
`tslearn` (HAVE).
**Builds.** `scripts/experiments/shape_archetypes.py`.
**Packages.** `py_pcha` (MISSING; kernel mode needs a short patch) or `spams`/SEACells pattern. Smoke-test first.
**Gate D (report).** N archetypes with stable extremes; cohort simplex positions with CIs across K.
**Cost.** ~1 day.

---

## 3. WS-E — Confound-robust cohort comparison (ComBat + OT/MMD)

**Goal.** Compare continuous repertoire *distributions* across cohorts/individuals while controlling the
dominant cage axis.

**Method.** Harmonize FPCA scores with **ComBat / neuroHarmonize** (per-feature location/scale, empirical
Bayes, protect named biological covariates); **CORAL** closed-form cross-check. Compare with **optimal
transport** (Wasserstein/Sinkhorn, `POT`) and/or **MMD** with a GAK elastic kernel.

**Validation (the part people skip — the lab partner-swap matrix makes it free):**
- *Negative control:* classifier predicting cage from corrected scores → accuracy collapses toward 25%
  (4-class chance); MMD/Wasserstein between cages drops sharply.
- *Positive control:* the **17-way lab partner-swap matrix is constant-cage** → run identical correction,
  confirm partner-identity decodability / partner OT-distance is essentially unchanged. If erased →
  over-corrected.
- *Identifiability:* where a wild pair appears in only one environment, the contrast is **unidentifiable** —
  restrict to within-stratum or flag. **Wild-vs-wild stays a noise floor; ComBat mainly buys lab-internal
  and (under assumptions) lab-vs-wild.**
- *Spurious-removal:* permute cage labels, re-run, confirm nothing systematic removed.

**Reuses.** Cohort strata definitions; existing JSD/repertoire comparison scripts as baselines.
**Builds.** `scripts/experiments/harmonize_and_compare.py`.
**Packages.** `neuroCombat`/`neuroHarmonize` (MISSING), `POT` (`ot`, MISSING), `geomloss` optional,
`statsmodels` (HAVE). Smoke-test each.
**Gate E (report).** "Within matched environments, distribution X differs from Y by Wasserstein W (perm
p<…), exceeding the cross-cage noise floor."
**Cost.** ~1.5 days (validation controls dominate).

---

## 4. Files: touch / NOT touch

- **Consume (read-only):** `models/shape_fpca/elastic_fpca_scores.parquet`, `models/shape_fpca/elastic_fpca.joblib`,
  `models/shape_kmeans/k20_softdtw.*`, `classified_detections_{full,3452,9252,lab_131204_clean}.csv`.
- **DO NOT TOUCH:** WS-A artifacts (no re-fit); incumbent `models/shape_kmeans/k20.joblib`; the locked
  functions + tests in `tests/experiments/test_eval_shape_human_anchored.py` (29) and the new
  `tests/experiments/test_ksg_te.py` (18); `src/usv_spectrogram/corpus.py`; `ExtractionConfig`; production
  detection pipeline. **No CNN work. VAE family stays closed.**
- **New code:** `scripts/experiments/{shape_archetypes,harmonize_and_compare}.py`; new tests under
  `tests/experiments/` (test-architect Step 0 if a new locked-spec public function lands).

## 5. Conventions
- Print every parameter / threshold / K / row count (`feedback_analysis_print_params`).
- User-facing outputs = HTML with a `file://wsl.localhost/Ubuntu/...` URL in the closing message.
- Report per-cohort, naming the stratum (`feedback_cross_animal_population_strata`).
- Run on the **box**; rig copy of ridges is read-only canonical.

## 6. Open / blocked
- **WS-F (link shape axes to behavior)** stays blocked on (1) emitter assignment (male-vs-female attribution)
  and (2) the LMT behavioral `.sqlite` DB (not located). Keep in the program; do not start.
