---
title: Lab 131204 batch — post-hoc duration filter for tonal-driven false positives
date_captured: 2026-05-13
source_type: methodology-decision
status: applied
---

## Decision

For the lab 131204 batch (`results/batch_lab_full_softnotch_20260513_1538/`), all
downstream analysis consumes `events_clean.parquet`, which is
`all_events_with_unmatched_flag.parquet` filtered to `duration_ms < 300`.

- 41,563 raw events → 41,061 kept (98.8%)
- 502 dropped (455 auto_accept + 47 manual_review)
- Filter is **per-event**, not per-chunk

## Why

The single-entry `data/lab_tonal_lines/lab_131204.json` library (calibrated at
50.7 kHz, `min_detection_rate=0.5`) did not catch ~4–6 additional persistent rig
tonals: prominently 61.6 kHz (+23.6 dB above local PSD median), plus 65.6 / 63.4
/ 30.8 / 66.4 / 50.6 kHz. These tonals appear in 7–22% of chunks per couple
across **all** major couples (m1fm1 through m6fm6) — lab-wide rig artifacts, not
couple-specific. Each fell below the calibration `min_detection_rate=0.5` filter
and was rejected.

In audit mode, unmatched tonals are **logged but not subtracted** from the
audio. So the CNN saw spectrograms with these tonals still present and produced
confidently-positive detections (max_prob ≈ 0.99–1.00) on flat horizontal energy
that the operator visually confirmed as noise.

## Evidence

| Test | Result |
|---|---|
| Triage rate inside vs outside unmatched-tonal chunks | 46.3% vs 16.8% (2.75× inflation) |
| Long-duration (>300 ms) event rate inside vs outside | 0.044 vs 0.0067 per chunk (6.5× enriched) |
| % of >500 ms events in unmatched-tonal chunks | 26 of 27 = 96.3% |
| Visual confirmation of top-5 longest events | 5/5 confirmed noise |
| % of detections >300 ms in the full lab batch | 1.2% |

## Why post-hoc filter, not recalibration

1. Pipeline change is wrong scope — wild data (5970, 3452, 9252) uses the
   default-off invariant in `notch.py`; modifying the CNN or notch for one
   batch's rig quirk would corrupt that guarantee.
2. Cost-benefit — 1.2% of events to drop vs 1.5 hr to recalibrate + rerun batch.
3. The contamination is concentrated in a duration tail that's already
   biologically implausible (real USV medians are 17–60 ms across our datasets;
   anything >300 ms is 5× the longest median).

## Caveats / what this does NOT fix

- Short events (<300 ms) in tonal-contaminated chunks may still contain some
  CNN false positives at tonal frequencies. The duration filter cannot
  distinguish a 50 ms real USV at 70 kHz from a 50 ms CNN response to a 65.6 kHz
  tonal. If Phase 2 classification finds a cluster that's pure 60–67 kHz flat
  energy with zero frequency modulation, that's the next thing to flag.
- All wild-vs-lab comparisons should treat the lab event count as having a
  ~1–2% noise floor that the wild-data event counts do not have.

## Related artifacts

- `scripts/diagnose_lab_unmatched_tonals.py` — diagnostic plot generator
- `scripts/plot_long_event_spectrograms.py` — top-5 long-event spectrogram render
- `results/batch_lab_full_softnotch_20260513_1538/diagnose_unmatched_tonals.txt` — full verdict
- `results/batch_lab_full_softnotch_20260513_1538/long_event_inspection/*.png` — visual confirmation
- `data/lab_tonal_lines/lab_131204.json` — the single-entry library that
  triggered the 34.3% unmatched warning
- `docs/modules/corpus-constants.md` §"Layer-2 fact: data/lab_tonal_lines" — soft-notch
  library design rationale
