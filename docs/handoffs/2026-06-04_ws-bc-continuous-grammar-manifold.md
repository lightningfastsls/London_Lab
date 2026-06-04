# Handoff — WS-B (KSG conditional transfer entropy) + WS-C (manifold), on the settled elastic-FPCA representation

**Date:** 2026-06-04  **Program:** `PLAN_continuum_repertoire_program.md` §2 (WS-B) + §3 (WS-C)
**Predecessor (binding):** `docs/handoffs/2026-06-04_ws-a-elastic-fpca-implementation.md` — DONE (GATE A = HYBRID).
**Status:** READY. The representation is settled; WS-B and WS-C run **in parallel** (they both consume WS-A scores).

---

## 0. What WS-A settled (read before starting)

GATE A = **HYBRID / qualified-ADOPT** (user decision, 2026-06-04). The shape representation for all
downstream work is:

- **Primary coordinate system:** elastic-FPCA λ=0.05 scores — `models/shape_fpca/elastic_fpca_scores.parquet`
  (67,337 rows; columns `wav_stem, call_id, cohort, amp_pc1..amp_pc5, phase_pc1..phase_pc3`). This is the
  **only O(N) scalable, navigable coordinate producer** (soft-DTW is O(N²) and not a coordinate system).
- **Carry-along metric for complex-sensitive analyses:** soft-DTW distances. Elastic-FPCA *loses* to
  soft-DTW on the `complex` family (non-overlapping CIs), so for any complex-focused question use the
  soft-DTW labeled-set distances (`models/shape_kmeans/k20_softdtw.*`) rather than the FPCA scores.
- Model bases: `models/shape_fpca/elastic_fpca.joblib` (Karcher mean, `amp_components` 5×50,
  `phase_components` 3×50, λ=0.05).
- **Dimensionality guidance:** amp_pc1/pc2 carry most amplitude variance; the program's WS-B budget caps
  shape at **≤ 2 FPCA dims** in any conditional (KSG bias control). Start with `amp_pc1, amp_pc2`.

---

## 1. WS-B — Continuous grammar: KSG conditional transfer entropy

**Goal.** Adjudicate: *is there shape-sequence structure beyond pitch/duration/timing autocorrelation?*
The honest claim to earn is a **dissociation** — shape-TE | (pitch, timing) collapses to the null while
timing-TE | shape stays significant (or vice-versa).

**⚠ VERIFY FIRST (load-bearing):** Perrodin, Verzat & Bendor 2023 (eLife) — "rhythm not order" claim.
The program's grammar interpretation leans on it. Confirm the actual finding (web search + fetch) before
writing the WS-B framing; do not cite from memory.

**Method.**
- KSG (Kraskov–Stögbauer–Grassberger) kNN MI/TE estimator on **continuous** coordinates (no alphabet).
- Nested comparison on the same within-bout call stream: timing-TE, pitch-TE, shape-TE, then conditionals
  `shape→shape | (pitch, duration)` vs `timing→timing | shape`.
- **Dimensionality budget:** joint ≤ 4–6 dims; **≤ 2 shape FPCA dims** in a conditional; k = 4–6.
- **Calibrate the estimator zero** on shuffled data (bias floor) — KSG is biased; report the null offset.
- **Surrogate = bout-wise CIRCULAR shift** (preserves each series' marginal autocorrelation, destroys
  cross-temporal coupling). NOT within-bout shuffle / IAAFT.
- Packages: `idtxl` or `jidt` (MISSING — install + smoke-test; existing `usv_language/analysis/sequence_analysis.py`
  MI is **discrete-only**, do not reuse for the continuous estimator — reuse only its bout/stream primitives).

**Join recipe (the gotcha that bites).**
- Ridge/FPCA composite id ↔ per-call features: ridge id = `f"{wav_stem}__det{call_id - 1}"` (the **−1 offset**
  is verified). Join the FPCA scores parquet to the per-call feature table on this.
- **Pitch = `principal_freq_hz`; duration = `call_length_s`** (NOT `det_duration_ms`). `mean_power_db`/`tonality`
  are cage artifacts — never use as biology.
- Within-bout adjacency: build `(amp_pc1, amp_pc2, pitch, duration, gap, bout_id, cohort)` per adjacent pair.
  Bout threshold: 0.6 s (corpus_facts) or 0.25 s (Stream-5 plateau) — report both if sensitive.

**Decision gate (WS-B).**

| Outcome | Action |
|---|---|
| shape-TE \| (pitch,timing) > null AND survives circular-shift surrogate | Real shape grammar — report effect size + per-cohort |
| shape-TE collapses to null but timing-TE \| shape survives | Grammar is timing, not shape order — reconcile with Chabout 2015 / Hertz 2020 / Perrodin 2023 |
| both collapse | No sequence structure beyond autocorrelation — the continuum has no grammar; report honestly |

---

## 2. WS-C — Manifold characterization (parallel to WS-B)

**Goal.** Characterize the shape manifold's topology on the **elastic coordinates** (so it is *shape*
topology, not pitch/time): connected? branches? density ridges? holes?

**Method.** Persistent homology (`ripser`), subspace-constrained mean-shift (SCMS) for density ridges,
and/or PHATE (`phate`) for visualization — all on `amp_pc1..amp_pc5` (+ optionally phase axes). Compare
to Track D's UMAP finding (navigable continuum, one connected manifold + small detached noise pocket).
Packages MISSING: `ripser`, `phate` — install per phase.

**Decision gate (WS-C).**

| Outcome | Action |
|---|---|
| One connected component, no significant H1 | Continuum confirmed on principled coordinates — strengthens Track D |
| Multiple components / persistent holes | Re-examine "continuum" claim; identify what separates the pieces (cohort? noise?) |

**⚠ Cohort confound:** tiny wild dyads 3452/9252 sit far out on `amp_pc1` (+12/+11 vs lab/5970 ≈ 0) — a
cage artifact (`feedback_cross_animal_population_strata`). Either restrict manifold work to lab+5970
(adequately powered) or defer cross-cohort topology to **after WS-E** (ComBat/OT harmonization). Do NOT
read 3452/9252 separation as biology.

---

## 3. Files: touch / NOT touch

- **Consume (read-only):** `models/shape_fpca/elastic_fpca_scores.parquet`, `models/shape_fpca/elastic_fpca.joblib`,
  `models/shape_kmeans/k20_softdtw.*` (complex carry), per-call feature tables.
- **DO NOT TOUCH:** the WS-A artifacts above (do not re-fit/overwrite); incumbent `models/shape_kmeans/k20.joblib`;
  the 5 locked functions + tests in `tests/experiments/test_eval_shape_human_anchored.py`;
  `src/usv_spectrogram/corpus.py`; production detection pipeline. No CNN work.
- **New code:** WS-B/WS-C scripts under `scripts/experiments/`; new tests under `tests/experiments/`
  (test-architect Step 0 if a new locked-spec module lands).

---

## 4. Relevant constraints (flattened — for an agent without vault access)

- pitch = `principal_freq_hz`; duration = `call_length_s` (NOT `det_duration_ms`); `mean_power_db`/`tonality`
  are cage artifacts.
- Ridge↔feature join: `f"{wav_stem}__det{call_id - 1}"` (−1 offset).
- All human shape labels are **lab cohort 131204**; wild (5970/3452/9252) are UNLABELED → any human-anchored
  validation is lab-only. 3452 (n=334) / 9252 (n=506) are underpowered.
- Print every parameter/threshold/row count. User-facing outputs = HTML with a `file://wsl.localhost/...` URL.
- Run on the **box** unless a step is GPU-bound; rig copy of ridges is read-only canonical.
