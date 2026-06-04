# PLAN — Continuous USV Shape-Space: Repertoire & Grammar Program

**Date:** 2026-06-04  **Owner thread:** shape-vs-pitch clustering → continuum (Shachar + Mickey)
**Status:** PLAN / capture. No production code until each workstream's GATE is read.
**Predecessors (all binding for their scope):**
- `PLAN_elastic_shape_clustering.md` (Phase 1 GATE 1 = PROCEED; Phase 3 = KEEP registration pending top-up)
- `docs/handoffs/2026-06-03_elastic-shape-phase2-labels.md` (Phase 2/3 spec + decision table)
- `docs/handoffs/2026-06-04_elastic-shape-5970-topup.md` (per-cohort gate, 5970 top-up)
- Track D artifacts: `results/shape_retrospective/trackD_shape_map.{html,json}` + 3 PNGs (built 2026-06-04)
- Memory: `project_shape_registration_clustering` (the canonical shape-clustering record)
- External research dossier: `compass_artifact_wf-…_text_markdown.md` (web-Claude survey + clarifications, 2026-06-04)

**This plan SUPERSEDES** the open Tracks of `PLAN_shape_representation_v2.md` and does **not** re-open the
falsified learned-encoder/VAE family (`docs/handoffs/2026-06-02_shape-vae-family-CLOSED.md`, 7/7 kills).

---

## 0. The reframe (what we now believe — the premise of everything below)

1. **Shape is a navigable CONTINUUM, not discrete syllable types.** Track D: one connected manifold,
   smooth morph (valley/chevron → flat/ramp → step/jump), K=20 letters are a *coarse index*, not natural
   kinds. Externally corroborated by **Goffinet et al. 2021 (eLife)** — mouse USVs "form a broad continuum,"
   do not cluster as commonly believed.
2. **The lever is preprocessing + elastic warp-alignment, NOT representation learning.** 7/7 learned
   encoders failed to beat hand-crafted registration. Per-pair soft-DTW (warp alignment) is the only method
   that *beat* registration, and only on `jump`. Learned warp-invariant encoders (DTAN etc.) win on
   amortized inference, **not fidelity** → irrelevant on a fixed 67k corpus.
3. **The "grammar" likely lives in timing/pitch/duration, not shape-order.** Our shape-alphabet transition
   MI collapsed (−72% vs the latent alphabet). Reconciled with the literature: **Chabout 2015's "syntax"
   used a 4-symbol jump-code (s/u/d/m)** = pitch/duration/timing by another name; **Hertz et al. 2020
   (our London lab)** parsed by inter-syllable interval (timing-intertwined); **Perrodin, Verzat & Bendor
   2023 (eLife)** reportedly found females respond to *rhythm* but are invariant to *syllable order* and
   *individual-syllable structure* — near-direct behavioral support. ⚠ **Perrodin 2023 NOT yet verified —
   verify before citing.**
4. **Cage (recording environment) dominates the geometry.** Cross-WILD-cohort comparisons are largely
   *unidentifiable* (cage ≈ biological-unit collinearity); the 17-way lab partner-swap matrix (constant
   cage) is the clean within-cage biology. `mean_power_db`/`tonality` are cage artifacts — never biology.

**Net:** stop optimizing discrete alphabets. Build a continuous-shape representation, test whether
sequence structure survives conditioning on timing/pitch, characterize the manifold, express the
repertoire as soft membership, compare cohorts confound-honestly, and (when unblocked) link shape axes
to behavior.

---

## 1. The standing decision metric (LOCKED — applies to every representation choice)

`scripts/experiments/eval_shape_human_anchored.py` — human-anchored leave-one-out kNN retrieval purity
(k=10) vs the human shape labels in `data/manual_shape_labels.csv` (now **758 labels**, 440 family rows,
all cohorts), with 1000× bootstrap 95% CIs. **Decisions are made on non-overlapping CIs**, never point
estimates. Controls reported alongside every method: random base rate + registration-Euclidean (incumbent).
`shape η²` is retired (circular). The 5 core functions + 29 tests are SPEC — do not change signatures.

---

## 2. Workstreams

Each workstream: **goal · method · reuses · builds · packages · gate/kill · cost**. Ordered by
value-per-effort. WS-A is the prerequisite for B/C/D/E (they all consume its scores).

### WS-A — Representation: elastic FPCA (SRVF + warp alignment) — TOP, cheapest decisive move
- **Goal.** Replace/augment registration-Euclidean with the *principled* version of the soft-DTW win:
  Fisher–Rao elastic metric. Produce interpretable warp-invariant **shape axes** + an explicit
  amplitude(shape)/phase(warp) variance split.
- **Why it resolves the SRVF tension.** Our Phase-3 SRVF *lost* (jump 0.206 vs reg 0.373) because it was
  the *pointwise* q-transform with **no alignment**. The active ingredient is `min over γ` warp-registration
  to the Karcher mean (DP step). SRVF aligns in *velocity* space → a jump = a spike in f′ → predicted to be
  *especially* good on our jump/step calls. Tune the elasticity penalty λ (anti-"pinching"/over-warp) on
  the standing gate.
- **Method.** `fdasrsf`: curves→SRVF (q), Karcher mean under elastic metric, warp-align all curves (DP),
  FPCA on aligned SRVFs (amplitude axes) + separate PCA on warps γ (phase). Also emit the pairwise elastic
  distance matrix for the gate.
- **Reuses.** The standing gate. Insert a new method block **after the soft-DTW block (~line 323)** of
  `eval_shape_human_anchored.py`, distance-matrix path via `bootstrap_purity_ci_from_distance(...)`, computed
  on the ~440–550 labeled rows only (trivial). Add `--no-elasticfpca` flag mirroring `--no-softdtw`. The
  method auto-appears in JSON/HTML (results dict is iterated, not hardcoded).
- **Builds.** Elastic distance + FPCA-score producer over all 67,337 ridges (alignment-to-mean is **O(N)**,
  not the O(N²) that forced Track D's 2k subsample → scales to full corpus). New script
  `scripts/experiments/build_elastic_fpca.py` → `models/shape_fpca/elastic_fpca.joblib` +
  per-call scores parquet (parallel artifact; do NOT overwrite `k20.joblib`/`k20_softdtw.*`).
  If a NEW public function lands in the harness, write its test FIRST (test-architect), per /implement Step 0.
- **Packages.** `fdasrsf` (MISSING — build can need a Fortran/C toolchain; pin a known-good version, smoke-test
  on 10 curves first).
- **GATE A (proceed to use FPCA scores downstream if):** elastic-FPCA's human-anchored jump (and ideally
  complex) purity is **≥ soft-DTW** and **> registration** with non-overlapping CIs, and chevron/flat do not
  regress. **KILL/contingency:** if it ties registration on every family, the warp-alignment or λ is mis-set
  → retune once; if still tied, keep soft-DTW distances as the representation and skip FPCA scores (fall back
  to soft-DTW kernel for B/C/D/E).
- **Cost.** ~0.5–1 day (harness add ~2h; full-corpus producer + λ tuning ~half day).
- **GATE A RESULT — 2026-06-04 → HYBRID (qualified ADOPT).** Implemented per
  `docs/handoffs/2026-06-04_ws-a-elastic-fpca-implementation.md` (`fdasrsf` 2.6.9, prebuilt wheel — no
  toolchain needed). Standing gate, 611 labels (754/758 joined), λ swept {0,.01,.05,.1,.3,1.0}; **λ=0.05**
  chosen (jump-purity peak, collapses by λ=1.0 = over-rigid). Human-anchored kNN purity @λ=0.05:
  | family | registration | elastic-FPCA | soft-DTW |
  |---|---|---|---|
  | jump (33%) | 0.415 [.377,.453] | **0.506 [.467,.548]** | 0.522 [.480,.570] |
  | chevron | 0.186 [.143,.232] | **0.275 [.227,.320]** | 0.214 [.168,.261] |
  | flat | 0.419 [.384,.456] | 0.400 [.362,.438] | 0.396 [.362,.433] |
  | complex (11%) | 0.194 [.148,.236] | 0.160 [.131,.190] | 0.243 [.199,.284] |
  Elastic-FPCA **beats the incumbent (registration) on jump with non-overlapping CIs** (.467>.453) +
  improves chevron, holds flat, no regression. It **ties soft-DTW on jump** but soft-DTW **significantly
  beats it on complex** (non-overlapping: .199>.190) → fails the strict "≥ soft-DTW" ADOPT bar.
  **DECISION (user, 2026-06-04): HYBRID.** Adopt elastic-FPCA λ=0.05 scores as the **full-corpus
  coordinate system** for WS-B/C/D/E (it is the only O(N) scalable coordinate producer; soft-DTW is
  O(N²) / not a coordinate system), AND **carry soft-DTW distances as the labeled-set metric for
  complex-sensitive analyses**. Artifacts: `models/shape_fpca/elastic_fpca.joblib` + `elastic_fpca_scores.parquet`
  (5 amp + 3 phase axes, all 67,337 ridges); gate scorecard
  `results/shape_retrospective/human_anchored_eval_elasticfpca.{json,html}` (+ per-λ `sweep_lam*.json`).
  Incumbents (`k20.joblib`/`k20_softdtw.*`) untouched. Adversarially verified (3 independent reviewers, 0 blockers).
  ⚠ Gate purity is **lab-cohort-only** (all human labels are lab 131204; wild 5970/3452/9252 unlabeled).

### WS-B — Continuous grammar: KSG conditional transfer entropy — highest scientific value
- **Goal.** Quantitatively adjudicate Realization 3: *is there shape-sequence structure beyond pitch/
  duration/timing autocorrelation?* The honest claim to earn = a **dissociation**: shape-TE | (pitch,timing)
  collapses to the null while timing-TE | shape stays significant.
- **Method.** KSG (Kraskov–Stögbauer–Grassberger) kNN estimator of mutual information / transfer entropy on
  **continuous** coordinates (no alphabet). Nested comparison on the same call stream: timing-TE, pitch-TE,
  shape-TE, then conditionals shape→shape | (pitch,duration) vs timing→timing | shape. **Dimensionality
  budget: joint ≤ 4–6, ≤ 2 shape FPCA dims in a conditional, k = 4–6**; calibrate the estimator zero on
  shuffled data (bias floor). **Surrogate = bout-wise CIRCULAR shift** (preserves each series' marginal
  autocorrelation, destroys cross-temporal coupling) — the only one of {within-bout shuffle, IAAFT,
  circular} that isolates confound-vs-signal. Shift *within* bouts; pool shifts across bouts (short bouts
  admit few shifts). Belt-and-suspenders: condition AND surrogate.
- **Reuses.** `usv_language/analysis/sequence_analysis.py::segment_into_bouts`, the ordering logic from
  `scripts/analyze_sequential_structure.py` (sort by `file` + absolute time, exclude cross-file pairs), the
  bootstrap-CI pattern from `scripts/bout_threshold_sensitivity.py`. Per-call features from
  `classified_detections_{full,3452,9252,lab_131204_clean}.csv` (**pitch=`principal_freq_hz`,
  duration=`call_length_s` NOT `det_duration_ms`**, timing from `begin_time_s`/`end_time_s`+file datetime).
- **Builds.** (1) Continuous TE core (the existing MI primitives are *discrete-alphabet* — not reusable for
  KSG). Use `IDTxl` or `JIDT`(jpype) rather than hand-rolling. (2) `scripts/experiments/export_bout_pairs.py`
  — within-bout adjacent pairs with (shape FPCA scores, pitch, duration, gap, bout_id, cohort). (3) Join
  reconciliation between ridge-derived FPCA scores (`wav_stem`+`call_id`, −1 offset) and
  classified_detections (`file`+`id`) — **flagged build task**.
- **Packages.** `idtxl` (MISSING; needs jpype+Java/JIDT jar — or use JIDT directly). `dcor` optional.
- **GATE B (report, not pass/fail):** "After conditioning on pitch+duration and against bout-matched circular
  surrogates, shape→shape TE is [indistinguishable from / exceeds] the autocorrelation null." Either outcome
  is publishable; the dissociation figure is the deliverable.
- **Power caveat (from infra map):** within-bout adjacent pairs: 5970 ≈ **6,350** (fine), lab huge; **3452
  (402 calls) and 9252 (605 calls) are likely UNDERPOWERED** for conditional TE — report 5970 + lab as
  primary, wild-small as exploratory. Report across the bout-threshold plateau ([0.143,1.0] s; A2 canonical
  0.6 s vs Stream-5 MI-plateau 0.25 s) — the plateau itself is a result.
- **Cost.** ~1.5–2 days (TE core + surrogate design + join reconciliation + nested analysis).

### WS-C — Manifold characterization (what shape *is*)
- **Goal.** Convert the Track-D qualitative "one blob + detached noise pocket" into a defensible topological
  claim: filled blob vs 1-D curve/loop vs branching tree; intrinsic dimension; is the oscillatory pocket a
  distinct H₀ component?
- **Method.** Persistent homology (on kNN/spectral graph — naive PH is noise-sensitive in high-D),
  subspace-constrained mean shift (SCMS) density ridges/principal curves, intrinsic-dimension estimation,
  PHATE/diffusion maps (preserve continuous gradients; UMAP/t-SNE shatter trajectories). **Run on the
  elastic coordinates (WS-A)** so the topology is *shape* topology, not pitch/time.
- **Reuses.** Track D embedding scaffolding (`build_shape_map_trackD.py`); subsample/landmark pattern.
- **Builds.** `scripts/experiments/characterize_shape_manifold.py`.
- **Packages.** `ripser`/`giotto-tda` (MISSING), `phate` (MISSING), SCMS (impl or scikit add-on), an
  intrinsic-dim estimator (`scikit-dimension`).
- **GATE C (report):** "Shape manifold is statistically [blob/curve/branching]; intrinsic dim ≈ d; the
  oscillatory pocket is a persistent H₀ component." Cross-validate against subsampling + noise models.
- **Cost.** ~1 day.

### WS-D — Soft membership: kernel archetypal analysis (the continuum-honest "alphabet")
- **Goal.** Replace hard K=20 letters with graded membership: each call = convex mixture of a few extreme
  archetypes (sharp step, deep valley, flat ramp); cohorts differ in *where on the simplex* they sit.
- **Method.** Kernel archetypal analysis in the GAK/elastic kernel space (interior archetypes, not just
  convex hull). Choose K as a **resolution knob**: triangulate explained-variance elbow + resampling
  stability (instability as you tile a smooth region = you've passed useful resolution) + interpretability +
  downstream invariance. **Report robustness across K** — no K is privileged; a result that appears at only
  one K is an artifact.
- **Reuses.** WS-A elastic distances → GAK kernel; SEACells' kernel-AA code pattern (Nat Biotech 2023).
- **Builds.** `scripts/experiments/shape_archetypes.py`.
- **Packages.** `py_pcha` (MISSING; kernel mode needs a short patch) or `spams`/SEACells code; GAK from
  `tslearn` (HAVE).
- **GATE D (report):** N archetypes with stable extremes; cohort simplex positions with CIs.
- **Cost.** ~1 day.

### WS-E — Confound-robust cohort comparison (ComBat + OT/MMD)
- **Goal.** Compare continuous repertoire *distributions* across cohorts/individuals while controlling the
  dominant cage axis.
- **Method.** Harmonize FPCA scores with **ComBat/neuroHarmonize** (per-feature location/scale, empirical
  Bayes, protecting named biological covariates); **CORAL** as a closed-form cross-check. Then compare with
  **optimal transport (Wasserstein/Sinkhorn, POT)** and/or **MMD with a GAK elastic kernel**.
- **Validation (the part people skip — and our partner-swap matrix makes free):**
  - *Negative control:* classifier predicting cage from corrected scores → accuracy collapses toward 25%
    (4-class chance); MMD/Wasserstein between cages drops sharply.
  - *Positive control:* the **17-way lab partner-swap matrix is constant-cage** → run identical correction,
    confirm partner-identity decodability / partner OT-distance is essentially unchanged. If erased →
    over-corrected.
  - *Identifiability:* where a wild pair appears in only one environment, the contrast is **unidentifiable**
    — restrict to within-stratum or flag (per Realization 5). **Implication: wild-vs-wild stays a noise
    floor; ComBat mainly buys lab-internal and (under assumptions) lab-vs-wild.**
  - *Spurious-removal:* permute cage labels, re-run, confirm nothing systematic removed.
- **Reuses.** Cohort strata definitions; existing JSD/repertoire comparison scripts as baselines.
- **Builds.** `scripts/experiments/harmonize_and_compare.py`.
- **Packages.** `neuroCombat`/`neuroHarmonize` (MISSING), `POT` (`ot`, MISSING), `geomloss` optional,
  `statsmodels` (HAVE) for GLMM.
- **GATE E (report):** "Within matched environments, distribution X differs from Y by Wasserstein W (perm
  p<…), exceeding the cross-cage noise floor."
- **Cost.** ~1.5 days (correction design + validation controls dominate).

### WS-F — Link shape axes to behavior (BLOCKED — capture only)
- **Goal.** The biology payoff: "position along shape axis k varies continuously with arousal/context
  (effect size β), controlling for cage and animal" — impossible with discrete types.
- **Method.** db-RDA/PERMANOVA + distance correlation (dCor) directly on the elastic distance matrix
  (stays in geometry); GLMM predicting FPCA scores from behavioral covariates with random effects for
  animal+cage; KSG MI shape-coord↔behavior with circular-shift surrogates; rSLDS (`ssm`) with behavioral
  inputs for state structure.
- **BLOCKERS (must resolve first):** (1) **emitter assignment** — cohorts are male+female pairs; we *assume*
  the male vocalizes; behavioral correlation needs per-animal attribution (SLIM-type tooling). (2) the LMT
  behavioral `.sqlite` DB is **not yet located** (open reminder, Phase C of `docs/analysis-roadmap.md`).
- **Packages.** `dcor`, `ssm` (MISSING), `statsmodels` (HAVE), `scikit-bio`/`skbio` for PERMANOVA.
- **Cost.** TBD after unblocking.

---

## 3. Phasing & dependencies

```
Phase 1 (settle the representation)         WS-A  ── GATE A ──┐
                                                              ▼
Phase 2 (run on FPCA scores, parallel)      WS-B (grammar)  WS-C (manifold)
                                                              │
Phase 3 (express + compare)                 WS-D (archetypes)  WS-E (compare)
                                                              │
Phase 4 (biology, blocked)                  WS-F  ── needs emitter-ID + LMT .sqlite
```
- **WS-A gates everything** — do not start B/C/D/E until GATE A is read (they consume FPCA scores; if A
  falls back to soft-DTW distances, B/C/D/E use the soft-DTW kernel instead — still fine).
- WS-B and WS-C are independent → can run in parallel after Phase 1.
- WS-F is parked until blockers clear; keep it in the plan so it isn't lost.

---

## 4. Environment / dependencies (install per phase, NOT all at once)

| Package | Status | Workstream | Risk |
|---|---|---|---|
| `tslearn`, `statsmodels`, `sklearn`, `umap` | INSTALLED | A,B,D,E | — |
| `fdasrsf` | MISSING | A | build toolchain; smoke-test on 10 curves first |
| `idtxl` (+ `jpype`/JIDT jar) | MISSING | B | Java dependency; alternatively call JIDT directly |
| `dcor` | MISSING | B,F | light |
| `ripser`/`giotto-tda`, `phate`, `scikit-dimension` | MISSING | C | giotto can be heavy |
| `py_pcha`/`spams` (+ SEACells kernel-AA pattern) | MISSING | D | py_pcha kernel mode needs a patch |
| `neuroCombat`/`neuroHarmonize`, `POT` (`ot`), `geomloss` | MISSING | E | POT mature; geomloss GPU |
| `ssm`, `skbio` | MISSING | F | ssm = Cython build |

Install in a throwaway step per phase; record exact versions in the workstream's handoff (reproducibility).

---

## 5. Files: REUSE vs TOUCH vs DO-NOT-TOUCH

**Reuse (read / extend, parallel artifacts only):**
- Gate: `scripts/experiments/eval_shape_human_anchored.py` (extend; preserve the 5 locked functions + 29 tests)
- Sequence primitives: `usv_language/analysis/{sequence_analysis,information_theory}.py`
- Ordering/bout reference: `scripts/analyze_sequential_structure.py`, `scripts/bout_threshold_sensitivity.py`
- Embedding scaffold: `scripts/experiments/build_shape_map_trackD.py`
- Features: `classified_detections_{full,3452,9252,lab_131204_clean}.csv`
- Ridges: `…/jobs/57976676/tmp/shape_data/true_registered_ridges{,_meta}.npz` (canonical on rig
  `/data/shachar/contour_vae/results/latent_transitions/shape_alphabet/`)
- Labels: `data/manual_shape_labels.csv` (758 rows)

**New parallel artifacts (touch):** `scripts/experiments/{build_elastic_fpca,export_bout_pairs,characterize_shape_manifold,shape_archetypes,harmonize_and_compare}.py`;
`models/shape_fpca/elastic_fpca.joblib` (+ scores parquet); `results/shape_*/` outputs;
`data/manual_shape_labels.csv` (append-only if more labels).

**DO NOT TOUCH (binding):** incumbent `models/shape_kmeans/k20.joblib`; `models/shape_kmeans/k20_softdtw.*`;
the 5 tested functions in the gate harness; `src/usv_spectrogram/corpus.py`; `ExtractionConfig`; the
production detection pipeline (`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`,
`postprocessing/`). **No CNN retrain anywhere in this program.** The VAE family stays closed.

---

## 6. Data gotchas (carry into every workstream)
- **Two duration columns:** `call_length_s` (acoustic — USE for features) vs `det_duration_ms` (hysteresis
  event — only for visual-verdict filters). Differ up to 10×.
- **pitch = `principal_freq_hz`**; `mean_power_db`/`tonality` are **cage artifacts** — never condition on as
  biology, never report as biology without cross-cage calibration.
- **Bout threshold:** 0.6 s canonical (corpus_facts, 3×median IOI) vs 0.25 s (Stream-5 MI-plateau rec).
  Use file-aware; report across the [0.143, 1.0] s plateau.
- **Join keys differ:** ridge npz uses `wav_stem`+`call_id` (−1 offset, verified); classified_detections
  uses `file`+`id`. Reconcile before joining FPCA scores to sequence features.
- **Cohort sizes:** 5970=7,921; lab_131204=40,787; 3452=402; 9252=605. 3452/9252 are small → underpowered
  for conditional TE and per-cohort gates. 9252 is a confidently-wrong-CNN cohort (22.6% human-noise at
  high p) — needs an FP-filter detection pass before trusting its shapes.
- **Lab is a 17-pairing partner-swap matrix** (6 matched + 11 cross), constant cage — the positive-control
  goldmine for WS-E.

---

## 7. Open verifications & risks (do not let these get lost)
1. **VERIFY Perrodin, Verzat & Bendor 2023 (eLife)** — the rhythm-not-order behavioral result is load-bearing
   for WS-B's framing. Confirm it exists and says what's claimed before citing.
2. **Wild-vs-wild may be permanently unidentifiable** (cage≈unit). Frame WS-E claims accordingly.
3. **fdasrsf / idtxl / ssm install fragility** — smoke-test each before committing a phase.
4. **KSG underpower on small wild cohorts** — pre-count within-bout pairs; don't over-claim 3452/9252.
5. **GATE A fallback path** — if elastic-FPCA ties registration, downstream uses soft-DTW kernel; the program
   does not stall.

---

## 8. One-line definition of done (program)
A human-anchored elastic-FPCA representation is settled via the standing gate; the shape continuum is
characterized (manifold topology + soft archetypes); the grammar question is answered as a conditional-TE
dissociation with circular-shift surrogates; cohort comparisons are made confound-honestly with the
partner-swap positive control; and the behavior-linking workstream is fully specified, parked only on the
emitter-ID + LMT-`.sqlite` blockers — with every loser documented, not silently dropped, and the VAE family
left closed.

---

## 9. Key external references (from the web-Claude dossier — full list in the compass artifact)
- Goffinet, Brudner, Mooney & Pearson 2021, *eLife* (USV continuum).
- Srivastava & Klassen 2016 (*Functional and Shape Data Analysis*); Tucker, Wu & Srivastava 2013
  (amplitude/phase separation); `fdasrsf`.
- Cuturi & Blondel 2017 (soft-DTW); Cuturi 2011 (GAK).
- Kraskov, Stögbauer & Grassberger 2004 (KSG MI); Schreiber 2000 (transfer entropy); JIDT/IDTxl.
- Chabout et al. 2015; Hertz et al. 2020 (London lab); **Perrodin, Verzat & Bendor 2023 (VERIFY)**.
- Johnson, Li & Rabinovic 2007 + Fortin et al. 2017/2018 (ComBat); Peyré & Cuturi 2019 (OT); Gretton et al.
  2012 (MMD).
- Persad et al. 2023 (SEACells kernel-AA); Mørup & Hansen 2012 (archetypal analysis).
- Moon et al. 2019 (PHATE); Ozertem & Erdogmus 2011 (SCMS).
- Linderman et al. 2017 (rSLDS); Weinreb et al. 2024 (keypoint-MoSeq).
