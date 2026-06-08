# Handoff — Elastic shape Phase 2/3: WILD LABELING POOL READY

**Date:** 2026-06-03  **Predecessor:** `2026-06-03_elastic-shape-phase2-labels.md` (the spec; still binding)
**Status:** Prep complete & integrity-verified. **Blocked on HUMAN labeling only.**

This is a thin pointer that makes the predecessor immediately actionable. The decision
gate, constraints, and side-probe in the predecessor are unchanged — read it for the
"why". This file gives the ready artifacts + exact commands.

---

## What is already done (this session)

- **516-patch WILD labeling pool rendered** → `data/alpha3_human_patches_wild/` (+ `manifest.csv`).
  Covers all three wild cohorts that previously had **zero** labels: 5970 (239), 9252 (159), 3452 (118).
- **Integrity verified end-to-end.** Every manifest `call_id` joins to the correct ridge
  row in the correct cohort via the eval's own `build_join` (516/516). Pool was filtered to
  rows where `det_index = call_id-1` agrees with time-closest matching (96–99% per cohort;
  lab is 99.2%), so the PNG shown == the ridge labeled.
- **Render substrate matches the existing LAB labels** — `scripts/experiments/render_human_view_patches.py`
  (percentile-baseline VocalMat cleaning, tight per-call crop), the SAME tool that produced
  `data/alpha3_human_patches/` (the normal grayscale spectrograms the human actually labeled).
  ⚠️ A first attempt used `render_vocalmat_style_patches.py` (CONTOUR-MASKED VAE render) — it is
  ILLEGIBLE for shape labeling (washes the contour into vertical bands) and was discarded. Do
  NOT use the contour-masked render for human labeling.
- **3 suspect labels to revisit:** the human labeled 3 × 3452 calls as `Noise`
  (`labeled_at_index` 0–2 in `data/manual_shape_labels.csv`) on the BAD masked render before it
  was fixed. Recommend dropping + re-labeling them on the clean render
  (`2024-10-04_10-30-53_0000317__det0`, `..._11-04-53_0000479__det1`, `..._11-50-44_0000558__det1`).
- **Standing-gate harness confirmed**: `tests/experiments/test_eval_shape_human_anchored.py` 33/33;
  dry-ran the Phase-3 incantation on lab-only labels (reproduced Phase-1: jump elastic
  0.452 [0.394,0.516] beats registration 0.327 [0.278,0.381]).
- **Ridge npz staged** for a stable Phase-3 path: `/home/shachar/.claude/jobs/57976676/tmp/shape_data/`
  (canonical source remains the rig: `/data/shachar/contour_vae/.../true_registered_ridges{,_meta}.npz`).
- Reproducible pool builder (seed 20260603): `/home/shachar/.claude/jobs/57976676/tmp/build_wild_pool.py`.

---

## Phase 2 — Input expected (HUMAN)

Label the wild pool with the GUI until the Phase-3 targets are met:
- **≥ 20 per family** in {chevron, jump, flat, complex, FM}, and **≥ 2 wild cohorts** covered.
- `chevron_valley` is a sampling aid only (precision 0.30) — already used to oversample; **never a label**.

```bash
cd /home/shachar/projects/mickey_london_lab
PYTHONPATH=src .venv/bin/python scripts/labeling/hand_label_200.py \
    --manifest data/alpha3_human_patches_wild/manifest.csv --per-cohort 200
```

Resume-safe (skips already-labeled `call_id`s); appends to `data/manual_shape_labels.csv`.
Keypresses: C/R chevron · U/D FM · 1/2/3/4 step-jump family · F flat · X complex · S short · N noise · ? unclear.

**Also (optional but recommended):** +8 lab `complex` to clear the lab 12→20 shortfall —
re-run the same tool on the lab manifest `data/alpha3_patches/manifest.csv`.

---

## Phase 3 — Analysis (exact incantation; npz already staged)

```bash
cd /home/shachar/projects/mickey_london_lab
.venv/bin/python scripts/experiments/eval_shape_human_anchored.py \
    --meta /home/shachar/.claude/jobs/57976676/tmp/shape_data/true_registered_ridges_meta.npz \
    --lab  /home/shachar/.claude/jobs/57976676/tmp/shape_data/true_registered_ridges.npz \
    --human data/manual_shape_labels.csv \
    --out-json results/shape_retrospective/human_anchored_eval_phase3.json \
    --out-html results/shape_retrospective/human_anchored_eval_phase3.html
```

> If this job (`57976676`) has been deleted, re-stage the npz pair from the rig (canonical
> source above) or any prior job tmp, then point `--meta/--lab` at it.

Secondary NMI on the full-corpus elastic alphabet: join `models/shape_kmeans/k20_softdtw_letters.parquet`
(`wav_stem,call_id,cohort,softdtw_letter`) to the new labels → NMI(softdtw_letter, human_family).
**Report cohort as a stratum, not a pooled mean.**

---

## Phase 3 — Decision gate (verbatim from predecessor)

| Outcome on the expanded, wild-covering anchor | Action |
|---|---|
| soft-DTW beats registration on **jump/complex** with non-overlapping CIs **AND** holds on **≥1 wild cohort** **AND** no chevron/flat regression | **SHIP elastic.** Re-point the transition/idiom alphabet at `k20_softdtw`; update `docs/DATA_LOCATIONS.md` + `project_shape_registration_clustering` memory. |
| Elastic wins on lab but **fails on every wild cohort** | **KEEP registration** (lab-cage-specific effect — the artifact risk realized). |
| Elastic within CI of registration on jump & complex at higher N | **KEEP registration**; deliverable = honest "validated, not better at scale" memo. |
| chevron/flat **regress** under elastic | **KEEP registration** (or per-family routing — discuss). |

Either way the loser is documented, not silently dropped. Do not re-open the falsified
learned-encoder/VAE family (`2026-06-02_shape-vae-family-CLOSED.md`).

---

## Files to touch / NOT touch

- **Touch:** `data/manual_shape_labels.csv` (append only, via the GUI),
  `results/shape_retrospective/human_anchored_eval_phase3.*`.
- **Do NOT touch:** `scripts/experiments/eval_shape_human_anchored.py` logic or its tests
  (the standing gate — spec); `models/shape_kmeans/k20.joblib` (incumbent); `k20_softdtw.*`;
  `data/alpha3_human_patches_wild/` PNGs (the labeling substrate — re-render via the seeded
  builder + `render_human_view_patches.py` if the ridge set changes); `corpus.py`,
  `ExtractionConfig`, detection pipeline.
- The 516 wild PNGs + manifest are untracked. If keeping them in git, stage **by exact path**
  (`git add data/alpha3_human_patches_wild/`) — never `git add -A`/`.` (`feedback_no_bulk_stage_in_parallel_chats`).
