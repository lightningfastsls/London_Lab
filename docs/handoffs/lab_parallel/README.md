# Lab-Wild Comparison — Parallel Chat Handoffs

**Date:** 2026-04-24
**Context:** Lab mouse WAV data being copied into repo (cohort `131204`, 6 male/female pairs × 2 timepoints). The wild-trained CNN will be applied unchanged. Before lab detection runs (Stream A, GPU-bound), these 5 streams can run **in parallel** to:
1. Bring 3452 and 9252 wild datasets to the same analysis state as 5970
2. Audit 5970 detection quality (Phase A4)
3. Build the cross-population comparison module
4. Lock in the bout threshold question that blocks all sequential analyses

Each handoff in this directory is **self-contained** — paste it into a fresh Claude chat with no other context.

## Streams

| # | Handoff | What it does | Owner | Blocks |
|---|---------|--------------|-------|--------|
| 1 | `01_3452_audit_and_a3.md` | Verify existing 3452 classification, run audit_corpus, run A3 acoustic deep-dive | parallel chat | C1 cross-pop |
| 2 | `02_9252_classify_and_a3.md` | Run classification on 9252 (271 events), audit_corpus, A3 | parallel chat | C1 cross-pop |
| 3 | `03_5970_detection_audit_a4.md` | Phase A4: CNN confidence by type, HDBSCAN noise inspection, ultra-short cluster | parallel chat | nothing — quality check |
| 4 | `04_cross_population_module.md` | Build `src/usv_spectrogram/classification/cross_population.py` | parallel chat | D-stream lab analysis |
| 5 | `05_bout_threshold_sensitivity.md` | Bout threshold sweep + file-aware logic prototype (Q1 from Mickey) | parallel chat | sequential analyses |

## Coordination rules for all chats

1. **Do NOT touch lab data.** Stream A (lab batch detection) is owned by the main chat. Lab WAVs are still being copied. Stay on existing wild datasets only.
2. **Use the canonical pipeline only.** Model: `models/hard_neg_retrain/best_model.pt`. Bout threshold: 0.6 s (canonical, frozen until Q1 resolves). STFT/sample-rate constants come from `src/usv_spectrogram/corpus.py` — do NOT redeclare.
3. **No test-expectation editing.** If a test fails, surface it; do not modify expected values to pass.
4. **Print parameters.** Every analysis script run must print thresholds, sort keys, and filter row counts to stdout (project convention — `feedback_analysis_print_params`).
5. **Commit your work.** When done, write a commit with a clear message and update the handoff with the commit SHA + result summary.
6. **Surface decisions back.** If you hit a real architectural decision, do NOT auto-decide — append a "Decision needed" section to your handoff and ping the user.

## Resource budget

- Streams 1, 2, 3, 5 are CPU/pandas-light — safe to run all four concurrently.
- Stream 4 is code-writing only (no compute) — runs anywhere.
- All five together stay well under the GPU budget that Stream A (lab detection) will consume later.

## When all 5 finish

Main chat will:
- Move lab WAVs to `USV_lab_131204/` (organized by timepoint)
- Run Stream A (lab batch detection)
- Apply classification to lab data using the same scripts each parallel chat used
- Run Stream 4's `cross_population.py` on (5970+3452+9252_pooled, lab)
- Write `data/corpus_facts/lab_131204.json` and the comparison report
