# Handoff — WS-A Phase 1: Elastic FPCA (SRVF + warp alignment) into the standing gate

**Date:** 2026-06-04  **Program:** `PLAN_continuum_repertoire_program.md` (WS-A, the entry point)
**Predecessors (binding):** `PLAN_elastic_shape_clustering.md`; `docs/handoffs/2026-06-04_elastic-shape-5970-topup.md`;
memory `project_shape_registration_clustering`.
**Status:** READY TO IMPLEMENT. This is the cheapest decisive next move and it **gates WS-B/C/D/E**
(they all consume the representation this workstream settles).

---

## 0. One-paragraph why

We proved (Track D) that USV shape is a navigable *continuum* and that per-pair **warp alignment** (soft-DTW)
is the only lever that beat registration — but only on `jump`. The principled generalization is the
**Fisher–Rao elastic metric via SRVF with warp alignment** (elastic FPCA). Crucially, our earlier SRVF test
*lost* (jump 0.206 vs registration 0.373) **because it was the pointwise q-transform with NO alignment** —
the `min over γ` registration step is the active ingredient, and SRVF aligns in *velocity* space (a jump = a
spike in f′), so it is predicted to be *especially* strong on jump/step calls. This handoff adds elastic FPCA
as a first-class method in the standing human-anchored gate and builds a full-corpus score producer.

---

## 1. Scope (do exactly this, nothing more)

1. **Extend the gate harness** `scripts/experiments/eval_shape_human_anchored.py` with a new method
   `elastic_fpca(SRVF-WARP)` (a precomputed elastic *amplitude* distance matrix over the labeled rows),
   compared head-to-head against the incumbent and soft-DTW on the same kNN-purity metric with bootstrap CIs.
2. **Build a full-corpus producer** `scripts/experiments/build_elastic_fpca.py` that aligns all 67,337
   registered ridges to the elastic Karcher mean and emits FPCA scores (amplitude axes) + warp/phase scores,
   as a **parallel artifact** for WS-B/C/D/E to consume.
3. **Tune the elasticity penalty λ** on the standing gate's jump-purity and report GATE A.

Out of scope: WS-B/C/D/E; any CNN work; touching the incumbent alphabet; the production detection pipeline.

---

## 2. Implementation — harness extension (precise insertion points)

All line numbers from the current `scripts/experiments/eval_shape_human_anchored.py` (verified 2026-06-04).

- **CLI flag (~line 250):** add `--no-elasticfpca` and `--fpca-lambda` (default per §4), mirroring the
  existing `--no-softdtw` / `--softdtw-gamma`.
- **Method block — insert immediately AFTER the soft-DTW block (~line 323):** follow the *distance-matrix*
  pattern exactly (soft-DTW is the template):
  ```python
  if not args.no_elasticfpca:
      print("\n  [elasticFPCA] computing SRVF elastic (amplitude) distance matrix on labeled ridges...")
      D_efpca = _elastic_amplitude_distance_matrix(X_reg, lam=args.fpca_lambda)   # (n_labeled, n_labeled)
      ef_row = {f: bootstrap_purity_ci_from_distance(D_efpca, yf, f,
                                                     k=args.k, n_boot=args.n_boot, seed=args.seed)
                for f in FAMILIES}
      results["elastic_fpca(SRVF-WARP)"] = ef_row
  ```
  - `X_reg = Sh[rows]` is the registered-ridge matrix over labeled rows only (~440–550 rows now → a
    ~550×550 matrix; O(N²) elastic alignment here is trivial, sub-minute).
  - `_elastic_amplitude_distance_matrix` is a **private helper** (no new locked-spec public function → no
    test-architect burden on the harness itself; see §3 for the producer's tests).
- **It auto-appears** in the JSON scorecard and HTML table — `results` is iterated, not hardcoded
  (HTML row builder ~line 446). No reporting code changes needed.
- **GATE-1 verdict text (~lines 398–432) is hardcoded to registration + soft-DTW** — leave it; elastic_fpca
  will show in the table/JSON but won't alter the printed GATE-1 wording. Read GATE A (§5) off the table.

**`_elastic_amplitude_distance_matrix` — intent (confirm exact `fdasrsf` API by smoke test, do not assume):**
for each pair (i,j) compute the elastic **amplitude** distance between the two registered ridges under the
Fisher–Rao/SRVF metric *with* warp optimization and elasticity penalty `lam`. The likely entry point is
`fdasrsf.elastic_distance(f1, f2, time)` which returns `(Dy_amplitude, Dx_phase)` — use the amplitude
component. **Verify the signature/return order on 10 curves before the full matrix.** Symmetrize and zero the
diagonal (as the soft-DTW path does). Time grid = `np.linspace(0, 1, 50)`.

---

## 3. Implementation — full-corpus producer + tests (test-architect Step 0)

`scripts/experiments/build_elastic_fpca.py`:
- **Input:** the staged ridge npz (`true_registered_ridges_meta.npz`, key `shapes` = (67337,50) + identity
  keys). Align-to-mean is **O(N)**, not O(N²) — full corpus is feasible on the box.
- **Steps:** (a) build the elastic Karcher mean (`fdasrsf` `fdawarp`/`srsf_align` — confirm); (b) warp-align
  all 67,337 ridges to it; (c) vertical/amplitude FPCA on aligned SRVFs → shape-axis scores; (d) horizontal
  FPCA on the warps γ → phase scores.
- **Output (parallel artifacts — do NOT overwrite incumbents):**
  `models/shape_fpca/elastic_fpca.joblib` (mean, bases, λ, seed) + a per-call parquet
  (`wav_stem, call_id, cohort, amp_pc1..ampPCk, phase_pc1..phasePCk`).

**Pre-implementation tests (per `/implement` Step 0 — write BEFORE the producer code; spawn `test-architect`):**
the elastic distance helper is the only genuinely new logic, so lock it:
- elastic amplitude distance is **non-negative** and **symmetric**;
- it is **≈ 0 for a curve vs a monotone time-warp of itself** (the warp-invariance property — the whole point);
- it is **> 0 for two genuinely different shapes** (e.g., a flat vs a chevron synthetic);
- FPCA reconstruction error decreases monotonically as #components grows (sanity on the basis).
Do NOT modify the 5 locked functions or the 29 existing tests in
`tests/experiments/test_eval_shape_human_anchored.py`.

---

## 4. λ (elasticity) tuning

SRVF + DP can "pinch" (over-warp) if unregularized. Sweep `--fpca-lambda` over a small grid (suggest
`{0.0, 0.01, 0.05, 0.1, 0.3, 1.0}` — confirm the parameter's scale against the `fdasrsf` docs) and pick the λ
that **maximizes jump kNN-purity on the standing gate without regressing chevron/flat**. Report the full
sweep (per `feedback_analysis_print_params`). Default the chosen λ in both scripts afterward.

---

## 5. GATE A (the decision)

Run the extended harness on `data/manual_shape_labels.csv` (758 labels) and read the per-family table:

| Outcome (human-anchored kNN purity, bootstrap CIs) | Action |
|---|---|
| elastic_fpca jump (and ideally complex) **≥ soft-DTW** and **> registration** with non-overlapping CIs, no chevron/flat regression | **ADOPT elastic-FPCA scores** as the representation for WS-B/C/D/E. Update `PLAN_continuum_repertoire_program.md` + memory. |
| ties registration on every family | **Retune λ once.** If still tied, **FALL BACK**: keep soft-DTW distances/kernel as the representation for downstream; skip FPCA scores. The program does not stall. |
| regresses chevron/flat | Keep registration/soft-DTW; document the regression. |

Also report the elastic-FPCA result **per cohort** (lab / 5970 / 3452 / 9252), not just pooled — per
`feedback_cross_animal_population_strata`; 3452/9252 are tiny (402/605 calls) → exploratory only.

---

## 6. Files: touch / NOT touch

- **Touch:** `scripts/experiments/eval_shape_human_anchored.py` (add method block + 2 flags + 1 private
  helper); new `scripts/experiments/build_elastic_fpca.py`; new
  `tests/experiments/test_build_elastic_fpca.py` (test-architect); `models/shape_fpca/elastic_fpca.joblib`
  (+ scores parquet); `results/shape_retrospective/human_anchored_eval_elasticfpca.{json,html}`.
- **DO NOT TOUCH:** the 5 locked functions + 29 tests in `test_eval_shape_human_anchored.py` (spec);
  incumbent `models/shape_kmeans/k20.joblib`; `models/shape_kmeans/k20_softdtw.*`;
  `src/usv_spectrogram/corpus.py`; `ExtractionConfig`; production detection pipeline
  (`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`, `postprocessing/`). No CNN retrain.

---

## 7. Relevant constraints (flattened — for an agent without vault access)

- The gate's 5 functions (`group_family`, `build_join`, `loo_knn_purity`, `knn_purity_from_distance`,
  `bootstrap_purity_ci`) are tested spec — do not change signatures. `bootstrap_purity_ci_from_distance`
  is NOT locked and is the one you call for the distance-matrix path.
- Label join: ridge composite id = `f"{wav_stem}__det{call_id - 1}"` (the −1 offset is verified; `build_join`
  handles it). 200/204 → now ~440–550 of 758 labels join.
- Per-call feature columns elsewhere (for later WS, not this one): pitch = `principal_freq_hz`,
  duration = `call_length_s` (NOT `det_duration_ms`). `mean_power_db`/`tonality` are cage artifacts.
- Print every parameter/threshold/row count. User-facing outputs = HTML with a `file://wsl.localhost/...`
  URL in the closing message.
- Run on the **box** (labeled-set matrix is tiny; full-corpus align is O(N)). No rig launch needed. The rig
  copy of the ridges is read-only canonical.

---

## 8. Data & environment

- **Ridges (staged, present 2026-06-04):**
  `/home/shachar/.claude/jobs/57976676/tmp/shape_data/true_registered_ridges{,_meta}.npz`.
  If that job dir is gone, re-stage from rig
  `/data/shachar/contour_vae/results/latent_transitions/shape_alphabet/true_registered_ridges{,_meta}.npz`.
- **Labels:** `data/manual_shape_labels.csv` (758 rows).
- **Install (MISSING — do this first, smoke-test before the full run):** `fdasrsf` into `.venv`. It can need
  a build toolchain; pin a known-good version and verify `import fdasrsf` + a 10-curve `elastic_distance`
  smoke test BEFORE writing the matrix loop. `tslearn` (soft-DTW baseline) is already installed.

---

## 9. How to run (after implementation)

```bash
cd /home/shachar/projects/mickey_london_lab
# 1. tests first
.venv/bin/python -m pytest tests/experiments/test_build_elastic_fpca.py -x -q
.venv/bin/python -m pytest tests/experiments/test_eval_shape_human_anchored.py -q   # 29 still green

# 2. gate with elastic FPCA (λ sweep handled inside or via repeated --fpca-lambda)
T=/home/shachar/.claude/jobs/57976676/tmp/shape_data
.venv/bin/python scripts/experiments/eval_shape_human_anchored.py \
    --meta $T/true_registered_ridges_meta.npz \
    --lab  $T/true_registered_ridges.npz \
    --human data/manual_shape_labels.csv \
    --fpca-lambda <chosen> \
    --out-json results/shape_retrospective/human_anchored_eval_elasticfpca.json \
    --out-html results/shape_retrospective/human_anchored_eval_elasticfpca.html

# 3. full-corpus producer (parallel artifact)
.venv/bin/python scripts/experiments/build_elastic_fpca.py --meta $T/true_registered_ridges_meta.npz
```

---

## 10. Definition of done
- 29 existing gate tests still green; new producer tests green.
- The gate HTML/JSON shows `elastic_fpca(SRVF-WARP)` per family with bootstrap CIs, pooled **and per cohort**.
- λ sweep reported; GATE A read and recorded (adopt / fall-back / regress).
- `models/shape_fpca/elastic_fpca.joblib` + scores parquet exist (incumbents untouched).
- Memory `project_shape_registration_clustering` + `PLAN_continuum_repertoire_program.md` updated with the
  GATE A outcome.

## 11. Successor
On **adopt** or **fall-back**, the representation is settled → start **WS-B** (continuous KSG conditional
transfer entropy) and **WS-C** (manifold characterization) in parallel, per
`PLAN_continuum_repertoire_program.md` §2–3. ⚠ Before WS-B writing, **verify Perrodin, Verzat & Bendor 2023
(eLife)** — it is load-bearing.
```
