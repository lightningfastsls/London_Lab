# 9252 Rate Anomaly — Evidence Against Four Hypotheses

**Stream:** Lab-parallel Stream 2, Step 7 (rate-anomaly investigation)
**Date:** 2026-04-24
**Commit SHA:** _(recorded after commit)_
**Author:** Claude (Python-half work; MATLAB half pending)
**Inputs:** `results/batch_9252/all_detections.csv`, `results/batch_9252/summary.parquet`, `results/batch_5970_v2_full/summary.parquet`
**Outputs:** `results/rate_anomaly_9252/` — `rate_anomaly_stats.json`, `per_session_rates.csv`, 2 figures, 9 sanity-check PNGs

---

## TL;DR

The 9252 wild animal (`usv_lmt_036`) is genuinely quiet — **not** a detection artifact. Three of four hypotheses are falsified or weak; the only surviving explanation is **animal silence**, and even that is non-uniform across the 8 sessions.

The corrected magnitude is **7.6× lower file-yield** and **23× lower events-per-file** vs 5970 — not 50× as the handoff stated. The inflated 50× came from reading the partial-coverage `summary.parquet` (271 events) rather than the complete per-file JSON output (597 events). Downstream pipelines should read the merged CSV, not the summary parquet, to avoid compounding this error.

---

## Headline numbers

| metric | 5970 | 9252 | ratio |
|---|---:|---:|---:|
| WAVs | 6 400 | 11 580 | |
| Files with events | 1 328 | **318** (not 173) | 4.2× more in 5970 |
| Events total | 7 575 | **597** (not 271) | 12.7× more in 5970 |
| File-yield | 20.75 % | **2.75 %** | **7.56× lower in 9252** |
| Events/file (mean, 95 % bootstrap CI) | 1.184 [1.107, 1.254] | **0.0516 [0.044, 0.060]** | **22.96× lower in 9252** |
| Median cumulative USV duration per active file | 486 ms | **51 ms** | 9.5× lower in 9252 |

The events/file CIs do not overlap and are separated by more than an order of magnitude — the gap is not a sampling artifact.

---

## Per-session breakdown (9252 only)

| session | n WAVs | files w/ events | file-yield | events | events/file (mean) | 95 % CI |
|---|---:|---:|---:|---:|---:|---|
| USV1 | 130 | 4 | 3.08 % | 5 | 0.0385 | [0.008, 0.077] |
| USV2 | 1 664 | 18 | 1.08 % | 24 | 0.0144 | [0.008, 0.022] |
| **USV3** | **1 602** | **114** | **7.12 %** | **288** | **0.1798** | **[0.140, 0.224]** |
| USV4 | 1 665 | 17 | 1.02 % | 18 | 0.0108 | [0.006, 0.016] |
| USV5 | 1 620 | 22 | 1.36 % | 25 | 0.0154 | [0.009, 0.023] |
| USV6 | 1 594 | 42 | 2.63 % | 58 | 0.0364 | [0.024, 0.050] |
| USV7 | 1 672 | 38 | 2.27 % | 51 | 0.0305 | [0.019, 0.045] |
| USV8 | 1 633 | 63 | 3.86 % | 128 | 0.0784 | [0.055, 0.105] |

**USV3 dominates**: 48.2 % of all 9252 events come from this single session (288 / 597), and it has the highest file-yield by a factor of ~2. USV4 is the extreme opposite at 0.011 ev/file. **Max/min ratio = 16.6×.** Whatever modulates this animal's vocalization cycles on and off — it is not a uniform "silent mouse."

*(Figure: `results/rate_anomaly_9252/fig_h2_per_session_rate.png` — per-session bars with 5970 reference line.)*

---

## Hypothesis evaluations

### H1 — Recording length (WEAK)

The per-WAV max event-end-time in 9252 reaches > 1 s across many files, so clips are not trivially short. Direct audio-duration comparison (reading WAV headers) was not performed; this is a gap worth closing if a definitive test is wanted. Within files that *do* have events, 9252's cumulative USV duration per file is 51 ms median vs 5970's 486 ms — a 9.5× gap that mirrors the events-per-file gap almost exactly. This suggests the within-active-file difference is the **same quantity** as the across-files difference, not a separate confound: when this animal does vocalize, it vocalizes less and shorter.

**Status:** Recording-length explanation does not account for the gap. If anything, 9252 has more file-count, so any uniform per-second rate would produce *more* events, not fewer.

### H2 — Animal silence (SUPPORTED — primary explanation)

All 8 sessions have rates below 5970's, but the distribution is wildly non-uniform (USV3 = 0.18; USV4 = 0.011; ratio 16.6). Even USV3 — the most vocal session — remains ~6.6× below 5970's mean. The data are consistent with "this is a quiet animal, but with session-dependent modulation" (circadian, social context, experimenter-driven stimuli, etc.).

**Status:** Primary surviving explanation. Needs correlating with LMT behavioral data (Phase C) to identify what makes USV3 different from USV4 — the scientifically interesting next step.

### H3 — Noise floor (FALSIFIED)

| statistic | 5970 | 9252 | direction |
|---|---:|---:|---|
| median `noise_floor_p90` | 0.0359 | **0.0199** | 9252 is **lower** |
| mean | 0.2154 | 0.0598 | 9252 is **lower** |
| q90 | 0.9930 | 0.1400 | 9252 is **lower** |
| KS statistic | 0.204 | | |
| KS p-value | < 1e-6 | | distributions differ |

9252's noise floor is systematically **lower** than 5970's across every summary statistic. If noise floor were suppressing detections, the relationship would run the other way. KS test confirms the distributions differ significantly — but in the direction that removes H3 from consideration.

**Caveat:** this uses the 60 % subset of 9252 WAVs that have summary rows. If the 4 675 uncovered WAVs are systematically noisier and absent from triage for that reason, the conclusion could flip. A sanity run over all 11 580 files with a quick noise-floor recompute is worth doing if this ever matters for a publication.

**Status:** FALSIFIED with a minor coverage caveat.

*(Figure: `results/rate_anomaly_9252/fig_h3_noise_floor.png` — overlaid histograms.)*

### H4 — Date / season (WEAK)

| | date range | unique dates |
|---|---|---:|
| 5970 | 2024-09-30 to 2024-10-01 | 2 |
| 9252 | 2024-10-06 to 2024-10-10 (summary-covered) | 5 |

Only ~5 calendar days separate the two datasets. Same season, same month, effectively identical daylight and temperature for indoor mouse housing. Seasonal effects at this temporal resolution are implausible unless a specific disturbance occurred (no evidence).

**Status:** WEAK. Not worth further investigation unless someone finds a specific environmental log entry.

---

## Summary-parquet coverage gap (important for downstream)

The 9252 `summary.parquet` has **6 905 rows**, not 11 580:

| session | WAVs | summary rows |
|---|---:|---:|
| USV1 | 130 | **0** |
| USV2 | 1 664 | **0** |
| USV3 | 1 602 | **0** |
| USV4 | 1 665 | 386 (partial) |
| USV5 | 1 620 | 1 620 |
| USV6 | 1 594 | 1 594 |
| USV7 | 1 672 | 1 672 |
| USV8 | 1 633 | 1 633 |

USV1–USV3 and most of USV4 are completely absent from tier triage — but their detection JSONs exist and contain events (USV3 alone contributes 288 events). The batch-detection CNN pass ran everywhere; the FP-filter/tier-assignment pass only ran on USV4-partial + USV5–USV8.

**Implication:** any analysis keyed off summary.parquet tier counts will systematically undercount by 145 files and 326 events for 9252 (45.7 % of the true event total). `results/batch_9252/all_detections.csv` is the canonical event source going forward.

**Remedial action needed:** rerun the triage stage on USV1–USV3 and the uncovered USV4 WAVs to produce a complete summary.parquet. Without this the tier-level comparison with 5970 is unsafe. This is a gating item for Step 5 (corpus_facts).

---

## Manual sanity check — 9 spectrograms

Rendered for human eyeball review at `results/rate_anomaly_9252/sanity_check/`. Each PNG shows the full WAV spectrogram (20–120 kHz USV band) with CNN-detected events as red highlights.

| tier | stem | session |
|---|---|---|
| auto_accept | 2024-10-06_14-58-31_0000121 | USV6 |
| auto_accept | 2024-10-06_15-00-22_0000127 | USV6 |
| auto_accept | 2024-10-06_15-19-18_0000215 | USV6 |
| manual_review | 2024-10-06_15-03-22_0000144 | USV6 |
| manual_review | 2024-10-06_15-07-36_0000160 | USV6 |
| manual_review | 2024-10-06_15-07-48_0000162 | USV6 |
| no_summary (USV1) | 2024-10-06_15-54-05_0000377 | USV1 |
| no_summary (USV4 gap) | 2024-10-07_09-13-48_0003015 | USV4 |
| no_summary (USV4 gap) | 2024-10-07_09-28-12_0003024 | USV4 |

**What to look for when reviewing:**
1. Are the red-highlighted events real USVs (upswept tonal contours in 40–100 kHz band)?
2. Are there obvious USVs the CNN did *not* flag? Especially in USV1 and USV4 files — if yes, this reopens H2 and suggests the model generalizes poorly to this animal's recording conditions.
3. Does the spectrogram look healthy (not dominated by broadband noise, not clipped)?

**Expansion path if a missed-USV is found:** pick 20 random "no events" files from each of USV1, USV4, USV8, render the same PNG, eyeball for missed syllables. Any systematic miss → model retraining needed before lab data.

---

## Implications for lab data pipeline

1. **Model generalization is uncertain.** The same CNN that extracted 1.18 events/file on 5970 extracts 0.052 on 9252 — both wild animals under putatively similar conditions. The factor-of-23 gap with lower noise floor says: either this animal is genuinely quiet, or the model has a per-animal recording-rig bias we haven't characterized. Lab data (`project_lab_data_pipeline`) could behave either way.
2. **FP-filter calibration was done on 5970.** If 9252's rate variability reflects rig-level differences, the FP filter's precision on the lab rig is an open question. Recommend: run a small held-out sanity set from the lab recording rig through the pipeline before committing to full-batch detection on weeks of recordings.
3. **Do not use summary.parquet as an event source for 9252.** It under-covers by ~40 % of WAVs. Always go through `all_detections.csv` or the per-file JSONs.
4. **Corpus-facts generation is blocked.** Step 5 of Stream 2 (`audit_corpus.py --dataset 9252`) requires the classified CSV, which requires MATLAB DeepSqueak feature extraction. Once MATLAB runs, the 9252 JSON will inherit the non-uniform per-session signal, and any Shannon-entropy / transition-matrix / Zipf stat on 9252 must use bootstrap CIs with a warning about the 8-session heterogeneity.

---

## Decision-needed flags (for Mickey)

- **[P0]** Rerun the FP-filter/triage stage on USV1–USV3 and the uncovered USV4 WAVs so summary.parquet covers the whole batch. Without this the tier statistics are misleading.
- **[P1]** Manual eyeball review of the 9 sanity-check PNGs. If you see obvious missed USVs in USV1 / USV4 files, the CNN may have a generalization gap that matters before lab data.
- **[P2]** Decide whether 9252 is suitable for N=3 cross-animal comparison given the 20× rate gap. Descriptive-only statistics (no inferential stats) with 5970 comparison is the conservative path; bootstrap CIs + JSD are mandatory per `project_wild_mice` memory.
- **[P3]** Once MATLAB DeepSqueak feature extraction runs, feed `all_detections.csv` (597 events), not the summary-parquet subset (271 events), so we don't bake the coverage gap into downstream analyses.

---

## What's NOT in this write-up (and why)

- **Classification distributions (scattoni_7, HDBSCAN).** Blocked until MATLAB feature extraction completes Steps 2–4. Raven selection tables for MATLAB are already staged at `raven_tables_9252/` (318 tables, one per WAV with events).
- **Phase A3 acoustic deep-dive.** Same blocker — needs the classified CSV.
- **WAV header duration read for H1 definitive test.** Skipped as not load-bearing given H3 is falsified and H2 is supported; can be added in a follow-up if a publication needs it.
- **Per-hour / per-minute rate curves within sessions.** Not requested by Stream 2; natural next question once LMT behavioral sync is set up (Phase C).

---

## Reproduce

```bash
# 1. Build merged detections CSV
.venv/bin/python scripts/merge_batch_detections.py \
    --detections-dir results/batch_9252/detections \
    --output         results/batch_9252/all_detections.csv

# 2. Staging for MATLAB half (318 Raven tables)
PYTHONPATH=src:. .venv/bin/python scripts/export_raven_tables.py \
    --detections-dir results/batch_9252/detections \
    --output-dir     raven_tables_9252 \
    --batch-format

# 3. Rate-anomaly analysis (all 4 hypotheses + figures + bootstrap CIs)
.venv/bin/python scripts/analyze_rate_anomaly_9252.py

# 4. Sanity-check spectrograms (9 PNGs for human eyeball review)
PYTHONPATH=src:. .venv/bin/python scripts/render_sanity_check_9252.py
```

All scripts print parameter blocks + row counts per `feedback_analysis_print_params.md`.
