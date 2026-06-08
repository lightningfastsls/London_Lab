---
title: Lab 131204 batch — 200-event manual labeling verdict
date_captured: 2026-05-14
source_type: methodology-decision
status: applied
supersedes: inbox/lab_131204_duration_filter_decision.md
---

## TL;DR

After hand-labeling 200 randomly-sampled events in the 200–299 ms duration band
of the lab 131204 batch, the population noise rate is **11.0%** [7.0%, 16.2%
95% CI]. The earlier 27-event hand-picked sample's 56% noise rate was a
selection-bias artifact (deliberately seeded with +23.6 dB 61.6 kHz tonal cases).

**Decision: KEEP the `events_clean.parquet` `<300 ms` filter as-is.** Layer in
tier-aware and couple-aware handling at Phase 2 instead of tightening duration.

## Findings

- **Per-stratum noise rates** (200 events, 50 per stratum × 4 batches):
  - 200–249 ms: 8.0% [3.5%, 15.2%]
  - 250–299 ms: 14.0% [7.9%, 22.4%]
  - Combined: 11.0% [7.0%, 16.2%]

- **Per-batch stability**: 10%, 12%, 6%, 16% — clustered around 11%, no batch
  was an outlier. Estimate is stable.

- **Tier signal is strong**: auto_accept = 8.8% noise; manual_review = 24.1%
  noise. The CNN's confidence calibration carries 2.7× discrimination power.

- **Noise is couple-specific, not batch-wide.** 18 of 22 noise events came
  from 4 couples (m1fm1, m1fm2, m1fm4, m3fm3 — 18–50% noise rates each). 12
  other couples had ZERO noise in our 200-event sample. The earlier
  "lab-wide contamination" framing was wrong.

- **61.6 kHz tonal is dangerous when present but rare.** WITH: 44% noise
  (4/9). WITHOUT: 9.4% (18/191). Only 9 of 200 events had this tonal.

- **Audit-tonal count is counter-predictive.** Chunks with 0 unmatched tonals
  had 14.1% noise; chunks with 3+ tonals had 4.9% noise. Many-tonal chunks
  are *active* periods rich in real biology; "clean" chunks are where the
  CNN slips on broadband or matched-library artifacts the audit can't see.

- **Calibration coverage helps but isn't decisive.** IN calibration: 6.1%
  noise. OUT of calibration: 14.4%. Modest 2.4× lift.

- **Window-extension is a third failure mode.** 4 of 200 events labeled
  `unsure`: real USV inside the CNN window, but the window also extends
  through surrounding noise, inflating the duration measurement.

## Filter cost-benefit (extrapolated to full 41,563 events)

```
Cut        Events kept   Est. noise kept   Retained rate   Real USVs lost
no filter      41,563         ~680             1.6%             0
<500 ms        41,536          654             1.6%             1
<300 ms ★      41,061          293             0.7%           115
<250 ms        40,298          186             0.5%           771
<200 ms        37,971           ~0             0.0%         2,912
```

`<300 ms` (current `events_clean.parquet`) is optimal: 0.7% retained noise rate
is clean enough for Phase 2 clustering to surface residual noise as its own
cluster, at a cost of only 115 real USVs vs. no-filter baseline.

## Phase 2 implementation notes

1. **Use `events_clean.parquet`** (the existing <300 ms filter) — NOT
   `events_super_clean.parquet` with `<250 ms` or `<200 ms`.

2. **Treat manual_review as a separate tier.** Either exclude from primary
   repertoire stats (cleanest) or report as a flagged subset (preserves
   manual-review biology). 24% noise rate means these *will* bias UMAP
   centroids and entropy calculations if mixed with auto_accept.

3. **Flag the 4 noise-prone couples** (m1fm1, m1fm2, m1fm4, m3fm3) for
   sensitivity analysis. Run the wild-vs-lab comparison both ways:
   - With all 17 couples
   - Excluding the 4 noise-prone couples
   Report whether conclusions change.

4. **Expect a residual-noise UMAP cluster.** ~0.7% of events are noise that
   slipped the duration filter. They'll likely cluster together. Manual
   inspection of the resulting "weird" cluster catches them post-hoc.

5. **m3fm3 paradox to investigate later.** This couple has 50% noise rate
   *despite* being in the soft-notch calibration sample. The noise events
   come from chunks with 0 unmatched tonals — meaning something else (matched
   library tonal stronger here? unique broadband signature?) is fooling the
   CNN. Worth one focused-investigation session if Phase 2 cluster surfaces
   m3fm3-heavy noise.

## Labeling artifacts (audit trail)

- `eyeball_labeling/labels.csv` — 200 per-event labels
- `eyeball_labeling/done/batch_{01..04}/` — archived spectrograms
- `eyeball_labeling/batch_{01..04}_picks.parquet` — per-batch sample manifests
- `eyeball_labeling/batch_{01..04}_position_map.csv` — position-to-stem maps
- `scripts/labeling_batch.py` — picker + recorder

## Supersedes

This memo supersedes the original `lab_131204_duration_filter_decision.md`
verdict, which committed to `<300 ms` based on the 27-event biased sample.
The conclusion is unchanged (keep `<300 ms`) but the rationale and the
Phase-2 guidance are substantially upgraded.
