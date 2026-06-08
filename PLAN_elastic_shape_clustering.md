# PLAN — Elastic-metric shape clustering (soft-DTW / SRVF) + human-anchored eval

**Date:** 2026-06-03  **Owner thread:** shape-vs-pitch clustering (Shachar + Mickey)
**Status:** PLAN / APPROVAL_PENDING — no production code until the Phase-1 gate is read.
**Predecessor evidence:** `results/shape_retrospective/shape_clustering_retrospective.html`
(retrospective + the 35-agent verification workflow + human-anchored re-scoring).
**Supersedes for shape clustering:** the open Track A/B of `PLAN_shape_representation_v2.md`
and the (CLOSED) `PLAN_geometric_shape_clustering_vae.md`. The VAE family stays closed.

---

## Why this plan exists (what the retrospective proved)

1. **The substrate is settled.** Shape lives in the 1-D registered ridge, not the 2-D pixel
   grid. The whole learned-encoder family is falsified (`2026-06-02_shape-vae-family-CLOSED.md`).
   This plan does **not** re-open it.
2. **The eval was the real bug.** The bake-off optimized `shape η²` against the `chevron_valley`
   heuristic — a metric computed on the *same* ridge KMeans optimizes (doubly circular), while a
   204-row human gold set sat unused. Scored against humans, registration only separates
   chevron/flat (~2.6×/2×) and leaves **jump and complex at chance**.
3. **One classical lever is genuinely untried AND already shows a win.** Per-pair elastic
   warp-alignment (DTW/soft-DTW) — the internal-landmark alignment registration provably lacks —
   beat registration's Euclidean metric on the 182 labeled ridges on every family, most on
   **jump (0.327 → 0.45)** and complex (0.117 → 0.183). Mechanism: a step/jump is a discontinuity
   at a *variable internal position*; warping aligns it, Euclidean cannot.

**The bet this plan tests (falsifiable):** a full-corpus elastic-metric alphabet beats the
incumbent registration alphabet (`models/shape_kmeans/k20.joblib`) on a **human-anchored** metric,
by a margin larger than label-noise, *especially on jump/complex* — without losing chevron/flat.
If it does, it becomes the production shape preprocessor. If it matches/loses, registration stays
and the deliverable is the honest eval harness.

**What this plan operates on (and why it's cheap):** the registered ridges already exist —
`/data/shachar/contour_vae/results/latent_transitions/shape_alphabet/true_registered_ridges{,_meta}.npz`
(`shapes` (67337,50), plus `wav_stem`/`call_id` for the human join). No re-extraction, no STFT, no
CNN-frozen config is touched.

---

## The decision metric (locked — replaces shape η²)

Every comparison below is reported on the **human anchor**, never on `shape η²`:

- **Primary:** per-family leave-one-out **kNN retrieval purity** (k=10) vs human labels — "do a
  call's nearest neighbours share its human shape family?" — for families
  {chevron, jump, flat, complex (+ FM, Noise reported for context)}.
- **Secondary:** NMI / adjusted-Rand of the K=20 alphabet vs human labels.
- **Mandatory controls reported alongside every method:** (a) random label assignment (= base
  rate), (b) **identity = registration's Euclidean ridge** (the incumbent), so every number is read
  as a delta over what we already ship.
- **Uncertainty:** 1000× bootstrap 95% CIs on every purity number. Decisions are made on
  **non-overlapping CIs**, not point estimates — the retrospective showed a single join-offset choice
  swung chevron purity 0.16↔0.36, so point decimals are not trustworthy at current N.
- `shape η²` is reported in a footnote only, explicitly labelled "circular — informational."

---

## Track 0 — PREREQUISITE: expand the human gold set (critical path, human work)

Current anchor: `data/manual_shape_labels.csv` — 204 rows, **lab cohort 131204 only**, only **4 true
"Chevron"** (25 chevron-family). Every conclusion is throttled by this. Phase 1 can run on the
existing 204 for a *preliminary* read, but the **production decision (Phase 3) requires** the expanded set.

- **Tool (exists):** `scripts/labeling/hand_label_200.py` (12-class vocabulary already defined).
- **Targets (gate to proceed to Phase 3):**
  - ≥ 20 examples in **each** of {chevron, jump, flat, complex, FM} — chevron is the binding constraint.
  - Coverage of **≥ 2 wild cohorts** (5970 / 3452 / 9252), which currently have **zero** labels —
    needed so the result isn't a lab-cage artifact (see `feedback_rig_artifact_mean_power_db`).
  - Suggested total ≈ 600–800 labels, **stratified by cohort × family** (oversample rare chevron/jump
    via the `chevron_valley` heuristic as a *sampling* aid only — never as a label).
- **Output:** append to `data/manual_shape_labels.csv` (same schema: `call_id,cohort,shape_label,labeled_at_index`).
- **Why this is the bottleneck, not the code:** the clustering is hours of compute; trustworthy
  evaluation is gated on labels only a human can produce.

---

## Track A — Full-corpus elastic alphabet (the new method, compute on the rig)

Goal: a K=20 (and a K-sweep) shape alphabet over all 67,337 registered ridges using an
**elastic metric**, as a parallel artifact to the registration alphabet.

- **Scale strategy (OOM-safe):** a full 67k×67k DTW matrix is ~36 GB — do **not** build it. Instead:
  1. fit **soft-DTW k-means** (`tslearn.clustering.TimeSeriesKMeans(metric="softdtw", gamma=1.0)`)
     on a **stratified subsample (~8–10k, cohort-balanced)** to learn barycenter centroids;
  2. **assign all 67,337** to the nearest centroid by soft-DTW distance (streamed in batches).
  - `tslearn` is already installed in `.venv`. Heavy fit runs on the **rig** (GPU0 / many cores;
    box is 11 GiB and has OOM'd — keep heavy work off it). **Rig compute launches are gated** —
    request per-session OK before launching (per `feedback_rig_claude_mediation`).
- **Second method (confirmatory, optional):** `fdasrsf` SRVF + `kmeans_align` (true Fisher-Rao
  geodesic with warp alignment) on the same subsample. If `pip install fdasrsf` is heavy/fails on the
  rig, soft-DTW alone is sufficient to decide — they test the same mechanism (per-pair warping).
- **K-sweep:** K ∈ {8, 12, 20, 30} — but interpret via the continuum caveat (Track D), not a single K.
- **Outputs (parallel — do NOT overwrite the incumbent):**
  - `models/shape_kmeans/k20_softdtw.joblib` (+ per-call alphabet letters as a parquet).
  - Rig: `/data/shachar/contour_vae/results/latent_transitions/shape_alphabet_softdtw/`.

---

## Track B — Human-anchored eval harness (the standing gate; productionize the probe)

Promote the throwaway probe into the permanent evaluator the project should have had.

- **Script:** `scripts/experiments/eval_shape_human_anchored.py` — built from
  `results/shape_retrospective/{human_anchored_eval.py, a1_elastic_test.py}`.
- **Inputs:** `true_registered_ridges_meta.npz` + `data/manual_shape_labels.csv`
  (join: `wav_stem + "__det" + (call_id - 1)`; the −1 offset is verified — det is 0-indexed,
  call_id 1-indexed; 200/204 match).
- **Compares:** {registration-Euclidean (incumbent), soft-DTW alphabet, SRVF, derivative} on the
  decision metric above, with bootstrap CIs and the random/identity controls.
- **Also emits:** the `chevron_valley`-heuristic-vs-human confusion (so the proxy's quality is always
  visible — today: precision 0.30 / recall 0.56).
- **Pre-implementation tests (per `/implement` Step 0):** spawn `test-architect` for the join logic
  (offset, dedup), the kNN-purity definition, and the bootstrap-CI computation, BEFORE writing the
  evaluator. These are spec; do not edit their expectations during implementation.

---

## Track D — Continuum vs discrete (decide before over-investing in K=20)

UMAP→HDBSCAN found a **continuum**, not crisp clusters, on registered ridges. The honest deliverable
may be a **navigable 2-D shape-map** (chevron region → jump region) under the elastic metric, with soft
membership, rather than N hard letters. Build the map (UMAP/MDS on the soft-DTW distance) and report
neighbour purity on it. The map figure was specified in `2026-05-25_shape-map-and-alphabet-decision.md`
but never rendered — render it here.

---

## Phases, gates, and kill criteria

**Phase 1 — Prototype at scale on the EXISTING 204 anchor (cheap, no new labels).**
- Build Track B harness; run Track A soft-DTW on the full corpus; score on the current 204.
- **GATE 1 (proceed if):** soft-DTW's human-anchored kNN purity on **jump OR complex** beats the
  registration-identity control with **non-overlapping bootstrap CIs**, and chevron/flat do **not**
  regress beyond CI overlap. (The 182-call pilot already suggests this for jump.)
- **KILL (stop here, keep registration) if:** soft-DTW is within CI of registration on *every*
  family at full scale — i.e. the 182-call jump win does not replicate. Deliverable then = the eval
  harness + a documented "elastic validated, not better at scale" memo.

**Phase 2 — Expand the human gold set (Track 0).** Human-gated; blocks Phase 3.
- **GATE 2:** Track-0 targets met (≥20/family, ≥2 wild cohorts).

**Phase 3 — Production decision on the expanded anchor.**
- Re-run Track B on the expanded labels (with cohort as a reported stratum, per
  `feedback_cross_animal_population_strata`).
- **SHIP elastic (new production preprocessor) if:** it beats registration on jump/complex with
  non-overlapping CIs **and holds on ≥1 wild cohort** (not a lab-cage artifact) **and** does not
  regress chevron/flat. Then re-point the transition/idiom alphabet at `k20_softdtw` and update
  `docs/DATA_LOCATIONS.md` + `project_shape_registration_clustering` memory.
- **KEEP registration otherwise.** Either outcome closes the thread honestly; neither re-runs a
  falsified attempt.

---

## Files to touch / NOT touch

- **Touch (new, parallel artifacts):** `scripts/experiments/eval_shape_human_anchored.py`,
  `scripts/experiments/build_softdtw_alphabet.py`, `models/shape_kmeans/k20_softdtw.joblib`,
  rig `/data/shachar/contour_vae/results/latent_transitions/shape_alphabet_softdtw/`,
  `data/manual_shape_labels.csv` (append only).
- **DO NOT touch (binding):** `src/usv_spectrogram/corpus.py`, `ExtractionConfig`, the production
  detection pipeline (`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`,
  `postprocessing/`), the incumbent `models/shape_kmeans/k20.joblib`, `train_contour_vae_v2.py`,
  `probe_shape_existing_encoder.py`. None of this work requires a CNN retrain.

## Compute & safety

- Heavy fits on the **rig** (gated launch); the box only for the eval harness on the ~hundreds of
  labeled rows and for streamed assignment in batches. Never full-scan `patches.npz` (11 GiB, OOM'd once).
- All rig inspection read-only until a compute launch is OK'd for the session.
- Print all parameters/thresholds/row counts in every run (per `feedback_analysis_print_params`).
- User-facing outputs default to HTML with a `file://wsl.localhost/...` URL (per `feedback_html_user_facing_default`).

## Effort estimate

- Track B harness: ~0.5 day (+ test-architect). Track A soft-DTW full corpus: ~0.5 day rig compute.
- Track 0 labeling: human-bound (the real critical path). Track D map: ~0.5 day.
- Phase 1 (prototype + Gate 1) is reachable **without any new labels** — it's the cheapest next move.

## One-line "definition of done"

A human-anchored eval harness is the standing gate, a full-corpus elastic alphabet exists as a
parallel artifact, and the Phase-3 decision (ship elastic vs keep registration) is made on
non-overlapping bootstrap CIs over an expanded, cohort-covering human gold set — with the loser
documented, not silently dropped.
