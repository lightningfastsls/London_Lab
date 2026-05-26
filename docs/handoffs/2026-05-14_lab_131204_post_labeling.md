# Handoff — Lab 131204 batch, post-labeling verdict

**Date:** 2026-05-14
**Status:** Phase 1 (detection + filtering) COMPLETE. Phase 2 (classification) UNBLOCKED.
**Previous handoff:** `docs/handoffs/HANDOFF_05_LAB_DATA_PIPELINE.md` (lab data pipeline plan, pre-arrival)

---

## TL;DR

First lab batch (131204, 25,770 chunks × 2 s ≈ 14 h, 17 couples) was detected
with the production CNN + soft-notch. The soft-notch fired a stale-library
warning (34.3% unmatched chunks). Diagnosis + 200-event hand-labeling
established the true noise rate in the contested 200–299 ms duration band is
**11% [7–16% 95% CI]**, NOT the 56% an earlier hand-picked micro-sample
suggested.

**Decision: `events_clean.parquet` (post-hoc `<300 ms` duration filter,
41,061 events) is the canonical Phase 2 input.** Do NOT tighten further.
Phase 2 should layer in **tier-aware** (manual_review = 24% noise vs
auto_accept's 9%) and **couple-aware** (4 couples produce 82% of noise)
handling instead of cutting more events.

---

## What was done this session

1. **Diagnosed the 34.3% stale-library warning** (`scripts/diagnose_lab_unmatched_tonals.py`)
   → Verdict: undercalibrated library, not drift. Library has 1 entry at
   50.7 kHz; audit caught ~6 missing tonals across the band (notably +23.6 dB
   at 61.6 kHz).
2. **Confirmed long-duration events are CNN errors on tonals** — visual
   inspection of top-5 longest events (>500 ms) → 5/5 noise.
3. **Applied `<300 ms` post-hoc duration filter** → `events_clean.parquet`
   (41,563 → 41,061 events).
4. **Hand-labeled 200 events** in the contested 200–299 ms band across 4
   batches of 50 with cross-couple stratification
   (`results/batch_lab_full_softnotch_20260513_1538/eyeball_labeling/labels.csv`).
5. **Verdict memos written** —
   `inbox/lab_131204_labeling_verdict.md` (awaits `/reduce`); auto-memory
   updated.

---

## The four findings that drive Phase 2 design

| Finding | Phase 2 implication |
|---|---|
| **Pop. noise rate in 200–299 ms = 11%** | `<300 ms` filter retains 0.7% noise rate — clean enough |
| **Tier signal: manual_review = 24% noise vs auto_accept = 9%** | Exclude or separately-report manual_review in primary stats |
| **Couple concentration: 4 couples produce 18 of 22 noise events** | Run wild-vs-lab comparison with AND without m1fm1, m1fm2, m1fm4, m3fm3 |
| **m3fm3 paradox: 50% noise rate despite being in calibration** | Investigate if Phase 2 cluster surfaces m3fm3-heavy noise |

---

## Canonical artifacts (Phase 2 inputs)

| Path | Purpose | Status |
|---|---|---|
| `results/batch_lab_full_softnotch_20260513_1538/events_clean.parquet` | **PRIMARY: 41,061 events, `<300 ms` filtered** | use this |
| `results/batch_lab_full_softnotch_20260513_1538/all_events_with_unmatched_flag.parquet` | Raw 41,563 events with per-event audit-tonal flag | audit/re-filter only |
| `results/batch_lab_full_softnotch_20260513_1538/summary.parquet` | Per-chunk summary (n_events, tier, max_confidence) | per-chunk views |
| `results/batch_lab_full_softnotch_20260513_1538/soft_notch_applied.parquet` | Per-tonal-application records (library + audit) | tonal-context lookups |
| `results/batch_lab_full_softnotch_20260513_1538/eyeball_labeling/labels.csv` | **200 hand labels — ground truth for the verdict** | empirical reference |
| `data/lab_tonal_lines/lab_131204.json` | Soft-notch library (1 entry @ 50.7 kHz) | DO NOT recalibrate (see "What NOT to do") |

---

## Phase 2 implementation guards

When running classification on the lab batch:

1. **Load `events_clean.parquet` exclusively.** Not `all_events_*` (raw,
   contains 502 confirmed-noise events) and not any tightened variant.
2. **Filter or tag `tier == 'manual_review'`** in repertoire-stats outputs.
   24% noise rate will bias Shannon entropy, JSD, and UMAP centroids if
   mixed with auto_accept.
3. **Two-pass sensitivity analysis** on the 4 noise-prone couples. Generate
   the wild-vs-lab comparison once with all 17 couples, once excluding
   m1fm1, m1fm2, m1fm4, m3fm3. Report whether conclusions change.
4. **Expect and characterize the residual-noise UMAP cluster.** ~0.7% of
   events (≈290) are noise that slipped the duration filter. Manual
   inspection of any tight cluster with no FM curvature is a Phase-2 step.
5. **Apply the same three classification approaches** as wild datasets:
   Scattoni-7 rule-based taxonomy, DeepSqueak k-means bridge, UMAP+HDBSCAN.

---

## What NOT to do (without explicit user approval)

- **DO NOT recalibrate `data/lab_tonal_lines/lab_131204.json`.** We decided
  against recalibration after labeling showed the audit-tonal flag is
  counter-predictive (audit-clean chunks have *higher* noise rate). The
  cost (~1.5 hr rerun) buys us nothing the duration filter doesn't already
  give.
- **DO NOT tighten the duration filter past `<300 ms`.** `<250 ms` costs 771
  real USVs to remove 107 noise events (7.2:1 trade). `<200 ms` costs ~2,900
  real USVs to remove ~290 noise events.
- **DO NOT modify production CNN, `notch.py`, or the wild-data detection
  pipeline.** The lab-specific contamination is handled by post-hoc
  filtering on the events parquet — not by changing the pipeline. The
  wild-mouse datasets (5970, 3452, 9252) depend on byte-identical
  detections (the default-off invariant in `src/usv_spectrogram/app/core/notch.py:30`).
- **DO NOT trust the audit-tonal flag** (`source == 'audit'` rows in
  `soft_notch_applied.parquet`) as a noise predictor. Counter-intuitively,
  chunks with 0 unmatched tonals have HIGHER noise rates than chunks with
  3+. The labeling evidence is in `labels.csv`.

---

## Open questions for later sessions

1. **m3fm3 paradox**: 50% labeled noise rate despite m3fm3 being in the
   calibration sample. Noise events have 0 unmatched tonals. Hypothesis:
   m3fm3 chunks have a *matched* library tonal (50.7 kHz) that's stronger
   than calibration assumed, OR a broadband artifact the audit can't see.
   Worth one focused investigation if Phase 2 surfaces an m3fm3-heavy
   noise cluster.
2. **Window-extension failure mode**: 4 of 200 events labeled `unsure`
   (real USV inside CNN window, but window also extends through noise,
   inflating measured duration). If Phase 2 cares about per-event duration,
   may need a window-tightening preprocessing step. Not on critical path.
3. **Generalization to other lab batches**: this verdict is for lab_131204
   specifically. When a 2nd lab batch arrives, rerun the diagnostic (`scripts/diagnose_lab_unmatched_tonals.py`)
   first; if the unmatched-rate is comparable and the noise-prone couples
   reappear, the same `<300 ms` filter and Phase-2 guards apply. If
   different, re-label.

---

## Vault / corpus constraints to respect

- **Canonical physical constants** live in `src/usv_spectrogram/corpus.py`
  (SAMPLE_RATE_HZ=300000, USV_FREQ_{MIN,MAX}_HZ=20000/120000, STFT_N_FFT=512,
  STFT_HOP=128). Any new script touching these MUST import from `corpus.py`,
  not redeclare. The post-hoc duration filter (`<300 ms`) is a Layer-2
  empirical decision, not Layer-1 physics — currently lives in
  `events_clean.parquet` provenance, not in `corpus_facts/`. Promote to
  `data/corpus_facts/lab_131204.json` only if a 2nd lab batch validates
  the same threshold.
- **Soft-notch architecture**: `docs/modules/corpus-constants.md` §
  "Layer-2 fact: data/lab_tonal_lines" is the design reference. Library
  schema is at `usv_spectrogram.app.core.notch.TonalLibrary`. The
  `_UNMATCHED_RATE_WARNING_THRESHOLD = 0.10` in `scripts/run_batch_detection.py`
  is what triggered the original warning on this batch.
- **CNN freeze**: production model is
  `models/hard_neg_retrain/best_model.pt`. Trained on 20–120 kHz spectrogram
  grid; ExtractionConfig literals enforce this. Do NOT modify
  `ExtractionConfig` without retraining.

---

## Verification commands (run before acting)

```bash
# Confirm the canonical Phase 2 input still has the expected event count
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('results/batch_lab_full_softnotch_20260513_1538/events_clean.parquet')
print(f'events_clean: {len(df):,} events')
assert len(df) == 41061, f'Unexpected count {len(df)} — expected 41,061'
print('OK')
"

# Confirm the 200 labels are intact
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('results/batch_lab_full_softnotch_20260513_1538/eyeball_labeling/labels.csv')
print(f'Labels: {len(df)} ({(df[\"label\"]==\"noise\").sum()} noise, '
      f'{(df[\"label\"]==\"real\").sum()} real, '
      f'{(df[\"label\"]==\"unsure\").sum()} unsure)')
assert len(df) == 200
print('OK')
"
```

---

## Immediate next action options

Pick one:

1. **Start Phase 2 classification.** Run Scattoni-7 + UMAP/HDBSCAN +
   DeepSqueak bridge on `events_clean.parquet`. Mirror the 5970 and 3452
   classification scripts. This is the natural next step on
   `HANDOFF_05_LAB_DATA_PIPELINE.md`'s Phase 2.
2. **`/reduce`** the inbox memos (`lab_131204_duration_filter_decision.md`
   and `lab_131204_labeling_verdict.md`) to promote into `notes/`. Quick
   housekeeping; ensures the verdict survives in the knowledge graph
   beyond auto-memory.
3. **Report to Mickey.** First lab batch is processed, characterized, and
   ready for comparison. Natural checkpoint to share findings (especially
   the couple-specific noise pattern) and align on lab-strain expectations
   before sinking time into the wild-vs-lab Phase 3.
4. **Investigate the m3fm3 paradox.** Pull all m3fm3 noise events from
   `labels.csv`, render their spectrograms, look at the failure mode
   pattern. Worth ~30 min if curious.

Default if none specified: option 1 (Phase 2 classification).
