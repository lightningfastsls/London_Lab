# Handoff — Elastic shape clustering, Phase 2/3 (human label expansion → production decision)

**Date:** 2026-06-03  **Predecessor:** `PLAN_elastic_shape_clustering.md` (Phase 1 DONE)
**Phase-1 result:** GATE 1 = **PROCEED**. soft-DTW beats registration on the human-anchored
**jump** family (0.452 [0.394,0.516] vs 0.327 [0.278,0.381], non-overlapping CIs). Report:
`results/shape_retrospective/human_anchored_eval_v2.html`.

This handoff covers Phase 2 (expand the gold set — the critical path) and Phase 3 (the
production ship/keep decision). **Phase 3 is BLOCKED on Phase 2.**

---

## Why this is the bottleneck (read first)

The Phase-1 win is real but **throttled by the gold set**: 182 labels, **all lab cohort
131204**, with only **12 complex** and **25 chevron**. Consequences proven in Phase 1:
- complex shows the right direction (0.117 → 0.183) but **cannot clear CI overlap at N=12**.
- The result could still be a **lab-cage artifact** — wild cohorts (5970/3452/9252) have **zero**
  labels, so we cannot yet say the elastic win generalizes beyond one recording environment
  (per `feedback_cross_animal_population_strata` / `feedback_rig_artifact_mean_power_db`).

The clustering is hours of compute that already exists. **Trustworthy evaluation is gated on
labels only a human can produce.** That is the entire Phase-2 task.

---

## Phase 2 — Input expected

Expand `data/manual_shape_labels.csv` (append-only; schema
`call_id,cohort,shape_label,labeled_at_index`). Targets that gate Phase 3:
- **≥ 20 examples in EACH of {chevron, jump, flat, complex, FM}** — chevron is the binding constraint.
- **Coverage of ≥ 2 wild cohorts** (5970 / 3452 / 9252), which currently have **zero** labels.
- Suggested total ≈ 600–800 labels, **stratified by cohort × family**. Oversample rare
  chevron/jump using the `chevron_valley` heuristic as a **sampling aid only — never as a label**
  (its precision vs human is only 0.30).

**Tool (exists):** `scripts/labeling/hand_label_200.py` (12-class vocabulary already defined).

---

## Phase 3 — Analysis (the exact incantation)

Re-run the standing gate on the expanded labels. The harness is already built and tested
(33/33), and already compares registration-Euclidean vs soft-DTW vs SRVF vs derivative with
bootstrap CIs and the random/identity controls:

```bash
.venv/bin/python scripts/experiments/eval_shape_human_anchored.py \
    --meta /home/shachar/.claude/jobs/<JOB>/tmp/shape_data/true_registered_ridges_meta.npz \
    --lab  /home/shachar/.claude/jobs/<JOB>/tmp/shape_data/true_registered_ridges.npz \
    --human data/manual_shape_labels.csv \
    --out-json results/shape_retrospective/human_anchored_eval_phase3.json \
    --out-html results/shape_retrospective/human_anchored_eval_phase3.html
```

> **Stage the ridge npz first.** The registered ridges are NOT in git. Canonical source is the
> rig: `/data/shachar/contour_vae/results/latent_transitions/shape_alphabet/true_registered_ridges{,_meta}.npz`.
> A box copy was used in Phase 1 (staged into a job tmp dir). `_meta.npz` carries `shapes`
> (67337,50) + `wav_stem`/`call_id`/`cohort`; the `_lab` npz carries the incumbent `lab_shape`
> K=20 labels + `chevron_valley`. Join offset is **−1** (det 0-indexed, call_id 1-indexed; 200/204 match).

For the **secondary** NMI on the full-corpus elastic alphabet, join the parquet to the new labels:
`models/shape_kmeans/k20_softdtw_letters.parquet` (cols `wav_stem,call_id,cohort,softdtw_letter`)
→ NMI(softdtw_letter, human_family). Phase-1 value: 0.264 vs incumbent 0.241.

**Report cohort as a stratum**, not a pooled mean.

---

## Phase 3 — Decision gate

| Outcome on the expanded, wild-covering anchor | Action |
|---|---|
| soft-DTW beats registration on **jump/complex** with non-overlapping CIs **AND** holds on **≥1 wild cohort** **AND** no chevron/flat regression | **SHIP elastic.** Re-point the transition/idiom alphabet at `k20_softdtw`; update `docs/DATA_LOCATIONS.md` + `project_shape_registration_clustering` memory. |
| Elastic wins on lab but **fails on every wild cohort** | **KEEP registration.** Document as a lab-cage-specific effect (the artifact risk the strata caveat warned about). |
| Elastic within CI of registration on jump & complex at higher N | **KEEP registration.** The Phase-1 jump win did not survive label expansion; deliverable = the honest "validated, not better at scale" memo. |
| chevron/flat **regress** under elastic | **KEEP registration** (or ship only if a per-family routing is justified — discuss). |

Either way the loser is documented, not silently dropped. Neither branch re-opens the
falsified learned-encoder/VAE family (`2026-06-02_shape-vae-family-CLOSED.md`).

---

## Side-probe (2026-06-03): "easy-first, hard-bin" two-stage idea — TESTED, does NOT help recovery

A parallel session proposed a two-stage scheme: Stage 1 = supervised/heuristic classification
of the **easy** families {Short, Flat, Chevron, Up-FM, Down-FM}; Stage 2 = route **everything
else** to a "hard bin" and cluster it unsupervised. Question raised: would peeling the easy
classes help the unsupervised step? **Probed empirically on the 182 labels** —
`scripts/experiments/probe_peel_easy_classes.py` (run with `PYTHONPATH=.`); report
`results/shape_retrospective/peel_easy_classes_probe.{html,json}`.

**Result: peeling does NOT help; it slightly hurts.** Headline = **base-rate-corrected**
adjusted purity `(purity-base)/(1-base)` (raw purity rises trivially when the pool shrinks —
do not be fooled by it). Load-bearing, artifact-free signal = **KMeans/elastic recovery of the
hard classes (multi-class NMI)**: FULL set K=7 NMI 0.104 (Euc) / 0.110 (soft-DTW) → RESIDUAL
{jump,complex,chevron} K=3 NMI **0.020 / 0.050** — recovery **halved**. Adjusted-purity agrees:
jump goes *negative* in the residual (complex points sit INSIDE the jump cloud — they don't
separate). Mechanism: peeling removes the crisp easy modes and leaves the **structureless middle
of the continuum**; unlike registration (which removes a continuous nuisance axis confounded with
shape), class-peeling removes whole subpopulations and cannot manufacture a jump/complex boundary
that was never there.

**Two things the probe CONFIRMED (carry into any pipeline design):**
1. **Contamination is real and large** — adding Noise to the bin shifted jump's adjusted purity
   by **+0.57** (soft-DTW). A hard bin's metrics are dominated by whatever junk lands in it →
   an explicit noise/FP class is **mandatory** (BootSnap pattern), not optional.
2. **The elastic metric helps hard classes regardless of peeling** — complex adjusted purity is
   positive and soft-DTW > Euclidean in every condition (full +0.126 vs +0.054; residual +0.147
   vs +0.087). **The lever is the metric, not the partitioning.**

**Implication for this plan:** do NOT build the two-stage scheme expecting cleaner hard-type
clusters. If pursued at all, its value is narrower — peel to **cut annotation burden** (auto-label
the easy ~60-70% to accelerate Phase-2 labeling), bin the rest **with an explicit noise class**,
and cluster under **soft-DTW as a soft/navigable map** (Track D), expecting a continuum, not new
crisp syllables. **Caveats:** N=12 complex, lab-131204 only — re-run the probe once Phase-2 adds
wild cohorts + more complex/chevron; and the probe tests recovery of *known* families, so it cannot
see *undiscovered* sub-structure (low prior given the continuum, but not excluded). Quick follow-up
not yet run: a `chevron`-stays-in-hard-bin variant.

---

## Files to touch / NOT touch

- **Touch:** `data/manual_shape_labels.csv` (append only), `results/shape_retrospective/human_anchored_eval_phase3.*`.
- **Do NOT touch:** `scripts/experiments/eval_shape_human_anchored.py` logic or
  `tests/experiments/test_eval_shape_human_anchored.py` expectations (the standing gate — spec);
  the incumbent `models/shape_kmeans/k20.joblib`; `models/shape_kmeans/k20_softdtw.*` (the Phase-1
  artifact — re-fit only if the ridge set changes); `src/usv_spectrogram/corpus.py`,
  `ExtractionConfig`, the detection pipeline. No CNN retrain is involved anywhere in this thread.

## Optional — Track D (not gate-critical)
Render the navigable 2-D shape-map (UMAP/MDS on the soft-DTW distance, neighbour purity on the
map). ~0.5 day; does not block Phase 2/3.
