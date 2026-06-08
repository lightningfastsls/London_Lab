# Handoff — WS-F (link shape axes to behavior) — BLOCKED on inputs

**Date:** 2026-06-05  **Program:** `PLAN_continuum_repertoire_program.md` §2 (WS-F), Phase 4
**Status:** BLOCKED — do not start until both inputs below exist. WS-D + WS-E (Phase 3) are DONE.

## What Phase 3 settled (read before starting)
- **Representation is settled (WS-A HYBRID):** primary coordinate = `models/shape_fpca/elastic_fpca_scores.parquet` (67,337 rows). Use the shared, tested merge helper `scripts/experiments/_fpca_merge.py` — do NOT re-derive the join (`call_id−1==det_index`, NOT `id`) or the largest-`|amp_pc1|` dedup.
- **WS-D (Gate D):** archetypes are a resolution knob — no privileged K, continuum confirmed. There are no natural kinds; any behavior link must be to *continuous shape coordinates / simplex position*, not to discrete archetype labels.
- **WS-E (Gate E):** cage dominates. lab-vs-wild and wild-vs-wild are unidentifiable under cohort-ComBat. Within-lab partner pairs differ statistically but **sub-noise-floor**. Any WS-F behavior contrast must run *within the lab cage* (the only identifiable stratum) and harmonize cage before claiming a shape↔behavior link is biological.

## Blocking inputs (BOTH required)
1. **Emitter assignment** — male-vs-female attribution per call. Without it, shape↔behavior coupling conflates the two animals. Source: not yet produced. Decision needed from user on method (mic-array localization? amplitude? manual?).
2. **LMT behavioral `.sqlite` DB** — the per-frame behavior/position table for lab_131204. Not located as of 2026-06-05 (also blocks the OVERDUE Phase-C PETH analysis). Ask the user where it lives or whether it exists.

## When unblocked — analysis sketch
- Join FPCA shape coords (via `_fpca_merge.load_merged_fpca`) to behavior events by timestamp (reuse `event_triggered.py` PETH machinery; mind the 75 ms DeepSqueak tolerance).
- Within the lab cage only: regress/condition continuous shape coordinates (amp_pc1..5, phase_pc1..3) or WS-D simplex membership on behavioral state; report per-stratum, harmonize cage with the WS-E ComBat pipeline first.
- Decision gate:

  | Outcome | Action |
  |---|---|
  | Shape coord couples to a behavior within-cage, survives ComBat + perm test | Report as candidate biological signal; replicate in a 2nd lab pairing |
  | Couples only before ComBat | Cage artifact — do not report |
  | Emitter unknown / DB absent | Stay blocked; do not force a pooled-emitter analysis |

## Files: touch / NOT touch
- **Consume (read-only):** `models/shape_fpca/*`, `models/shape_kmeans/k20_softdtw.*`, `classified_detections_*.csv`, the new `results/ws_d_archetypes/*` and `results/ws_e_harmonize/*`.
- **DO NOT TOUCH:** WS-A artifacts; `models/shape_kmeans/k20.joblib`; locked tests `test_eval_shape_human_anchored.py`, `test_ksg_te.py`, `test_fpca_merge.py`, `test_dist_stats.py`, `test_shape_archetypes.py`; `src/usv_spectrogram/corpus.py`; `ExtractionConfig`; production detection pipeline. No CNN work. VAE family stays closed.
- **New code:** `scripts/experiments/shape_behavior_link.py` + tests (test-architect Step 0 if a new locked-spec public function lands).

## Conventions
- Print every parameter / threshold / row count. User-facing outputs = HTML with a `file://wsl.localhost/Ubuntu/...` URL. Name the stratum in every cross-group claim. Run on the box.
