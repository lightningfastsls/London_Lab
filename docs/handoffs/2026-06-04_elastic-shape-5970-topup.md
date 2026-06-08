# Handoff — Elastic shape Phase 3: settle the 5970 near-miss

**Date:** 2026-06-04  **Predecessors:** `2026-06-03_elastic-shape-phase2-labels.md` (spec, binding),
`2026-06-03_elastic-shape-phase2-labels-READY.md` (pool prep)
**Status:** Phase 3 ran. Decision = **KEEP registration for now**; settle one near-miss with a top-up.

---

## What Phase 3 found (per-cohort, the decisive view)

The standing gate reports a **pooled** purity only. Re-ran it on cohort-filtered label
subsets (the handoff requires per-cohort + "holds on ≥1 wild cohort"). jump family,
LOO kNN purity (k=10, 1000-boot 95% CI), elastic soft-DTW vs registration-Euclidean:

| Cohort | jump N | elastic | registration | verdict |
|---|---|---|---|---|
| lab 131204 | 63 | 0.452 [.394,.516] | 0.327 [.278,.381] | **BEATS** (non-overlap) |
| 5970 (wild) | 76 | 0.659 [.576,.741] | 0.511 [.443,.580] | **near-miss** — overlap **0.004** |
| 3452 (wild) | 21 | 0.238 [.176,.300] | 0.238 [.181,.290] | dead tie |
| 9252 (wild) | 14 | 0.121 [.079,.171] | 0.136 [.079,.193] | no advantage (45% noise) |

- **Not a lab-cage artifact:** 5970 (wild) replicates the lab direction with an even **larger** raw
  effect (+0.148 vs lab's +0.125). It misses significance only on CI width (a power issue).
- **The pooled win (jump 0.463 vs 0.373, non-overlap) is carried by lab+5970 combined** — neither
  clears the bar alone. Do not cite the pooled number as a wild win.
- complex: lab 0.183 vs 0.117, 5970 0.396 vs 0.323 — elastic point-ahead everywhere but CIs overlap.
- chevron/flat: no regression in any cohort (the KILL row does not fire).
- Secondary: NMI(incumbent K=20 `lab_shape` alphabet vs human family) = 0.170 (pooled).

## Decision (user, 2026-06-04)

**KEEP registration (`models/shape_kmeans/k20.joblib`) as production for now.** No production swap.
Settle the 5970 near-miss with a top-up labeling batch before a final ship/keep call.

---

## Phase 2.5 — Input expected (HUMAN): label the 5970 top-up

80 fresh, integrity-gated, bracketed 5970 patches (pure random — no jump-enrichment, since no
proxy beats 5970's ~53% natural jump rate and selecting on a shape metric would bias the eval).
Zero overlap with the 678 already-labeled call_ids.

```bash
cd /home/shachar/projects/mickey_london_lab
PYTHONPATH=src .venv/bin/python scripts/labeling/hand_label_200.py \
    --manifest data/alpha3_human_patches_5970_topup/manifest.csv --per-cohort 200
```

Resume-safe; appends to `data/manual_shape_labels.csv`. Label every shape (jumps are the target,
~42 expected → 5970 jump N 76 → ~118). Keys: C/R chevron · U/D FM · 1/2/3/4 step-jump · F flat ·
X complex · S short · N noise · ? unclear. **Rule:** label only what is between the red bracket;
two separate calls inside the bracket (silent gap) = label the dominant one or `?` — a gap is NOT
a jump (jump = pitch step *within one continuous call*).

## Phase 3 (rerun) — Analysis

After labeling, rebuild the 5970 label subset and re-run the gate on it:

```bash
T=/home/shachar/.claude/jobs/57976676/tmp   # if this job is gone, re-stage npz from the rig
PYTHONPATH=src .venv/bin/python - <<'PY'
import pandas as pd
df=pd.read_csv("data/manual_shape_labels.csv"); df['cohort']=df['cohort'].astype(str)
df[df['cohort']=='5970'].to_csv("/tmp/labels_5970.csv", index=False)
PY
.venv/bin/python scripts/experiments/eval_shape_human_anchored.py \
    --meta $T/shape_data/true_registered_ridges_meta.npz \
    --lab  $T/shape_data/true_registered_ridges.npz \
    --human /tmp/labels_5970.csv \
    --out-json results/shape_retrospective/eval_5970_topup.json \
    --out-html results/shape_retrospective/eval_5970_topup.html
```

Read the `jump` row: compare `soft_dtw(ELASTIC)` vs `registration_euclidean(IDENTITY)` CIs.

## Decision gate

| Outcome on 5970 jump after top-up | Action |
|---|---|
| elastic CI lower bound > registration CI upper bound (non-overlap) | **SHIP elastic.** Re-point the transition/idiom alphabet to `k20_softdtw`; update `docs/DATA_LOCATIONS.md` + `project_shape_registration_clustering` memory. A wild cohort now clears the bar. |
| CIs still overlap (effect persists but underpowered) | **KEEP registration.** Deliverable = honest "validated direction, not decisively better at scale" memo. Do not keep chasing N indefinitely — one top-up is the agreed test. |
| elastic point estimate drops toward registration | **KEEP registration** — the Phase-1/3 signal was N-fragile. Document and close. |

Either way the loser is documented, not silently dropped. Do NOT re-open the falsified
learned-encoder/VAE family (`2026-06-02_shape-vae-family-CLOSED.md`).

---

## Side findings (not blocking)

- **9252 is a confidently-wrong-CNN cohort.** 22.6% human-noise at det_prob_max>0.88; tiny ridge
  set (506 calls); short detections (median 55 ms). Cage/domain-generalization gap — the detector
  over-accepts short broadband transients in 9252's recording environment. Noise labels are
  excluded from shape families so they don't hurt the eval, but 9252 yields few usable shapes and
  is not load-bearing for the gate. A future 9252 detection pass should add an FP-filter step.
- **chevron sampler is broken** (slope/`cv` proxy precision 0.163). Wild chevron coverage is 11/5/1.
  A chevron-specific test would need a better candidate finder (the precomputed `jump` score and a
  registered-contour max-step metric both conflate jump/complex/FM — neither enriches).

## Files to touch / NOT touch

- **Touch:** `data/manual_shape_labels.csv` (append via GUI only),
  `results/shape_retrospective/eval_5970_topup.*`.
- **Do NOT touch:** `scripts/experiments/eval_shape_human_anchored.py` + its tests (standing gate);
  `models/shape_kmeans/k20.joblib` (incumbent — KEEP); `k20_softdtw.*`; the 516-pool
  `data/alpha3_human_patches_wild/` and the 80-patch `data/alpha3_human_patches_5970_topup/` PNGs;
  `corpus.py`, `ExtractionConfig`, detection pipeline.
- Untracked artifacts (516 + 80 PNGs, manifests): stage **by exact path** if keeping in git —
  never `git add -A`/`.` (`feedback_no_bulk_stage_in_parallel_chats`).

## Reproducibility

- Top-up builder: `/home/shachar/.claude/jobs/57976676/tmp/build_5970_topup.py` (seed 20260604).
- Per-cohort eval inputs: `/home/shachar/.claude/jobs/57976676/tmp/labels_{5970,3452,9252,lab}.csv`.
- Ridge npz staged: `/home/shachar/.claude/jobs/57976676/tmp/shape_data/` (canonical = rig
  `/data/shachar/contour_vae/.../true_registered_ridges{,_meta}.npz`).
