# Handoff: Lab-Data CNN Fine-Tune — Phase 0 + Phase 1 Complete
Date: 2026-05-09
From: Claude Code (lab-cnn-finetune chat — labeling phase)
To: Claude Code (next chat — extraction + training phase)

This handoff supersedes `docs/handoffs/2026-05-09_lab-cnn-finetune-handoff.md`'s
Phase 0 (re-elicit labels) and Phase 1 (mine 500 hard negatives). Both are DONE.
The remaining work is Phases 2–5: extract → train → calibrate → wild-regress →
lab-batch.

## What's done

### Phase 0 — Audit re-labeled (71 unique events)
- All 72 audit PNGs in `results/batch_lab_131204_full/audit_2026-05-08/{cleanest,typical,borderline,long_event_survivors,high_overlap_survivors}/` were re-labeled.
- Output: `data/lab_finetune_v1/labels_audit_72.csv` (72 rows; 71 unique events because bor05 = lng10 same event in two buckets).
- Tally: **45 USV / 26 NOISE** (71 unique).
- All four buckets matched their audit-aggregate constraint exactly (cleanest 15/0, typical 4/11, borderline 5/5, long_event 4/6). high_overlap_survivors finalized at 17/5.

### Phase 1 — 500 hard-negative candidates mined and labeled
- Mining script: `scripts/mine_lab_finetune_candidates.py` (NEW).
- Sample: 350 typical + 80 long_event + 70 borderline = 500, sampled from `auto_accept` events excluding the 71 already-labeled audit events. Seed 42 for reproducibility.
- Output: `data/lab_finetune_v1/mining_candidates_500/candidates_seed42.csv` + 500 rendered PNGs at `mining_candidates_500/renders/`.
- Tally: **152 USV / 348 NOISE** in 500 mined events.
- All long_event Y verdicts were center-verified (training window is centered 43 ms; the broad detection box can include noise gaps that misalign the label). Two flips applied: min377, min426, min430 → NOISE. Center-verify utility: `scripts/zoom_event_png.py` (NEW; supports `--center-only-ms` for long events).

### Combined dataset
**571 lab-labeled events: 197 USV / 374 NOISE.**

Both CSVs use `(chunk_stem, event_idx)` as the join key against
`results/batch_lab_131204_full/merged_events_with_filter.parquet`.

## Stratum-level findings (audit prior vs measured)

| Stratum | Audit prior | Measured (n=70–350) | Direction |
|---|---|---|---|
| typical (SEF 0.05–0.15, 50–250 ms) | 27% Y | **21.4% Y** | -5.6 pp (close) |
| long_event (dur > 600 ms) | 40% Y | **18.8% Y** | **-21.3 pp** (audit overestimated USVs) |
| borderline (SEF 0.20–0.50) | 50% Y | **88.6% Y** | **+38.6 pp** (audit massively underestimated USVs) |
| high_overlap (SEF > 0.50, audit only) | — | 77% Y | (consistent with borderline finding) |

**Critical insight:** the original handoff's plan to mine "all events flagged
`noise_band_overlap` AND filter SEF > 0.5" was a misdirection. Events with high
SEF are dominantly **real USVs that overlap noise bands**, not noise. The
hard-negative-rich strata are typical (78.6% noise) and long_event (81.2%
noise). Mining the strata as we did was correct; mining by `noise_band_overlap`
flag would have been counterproductive.

## Behavioral / per-recording fingerprints captured

Lab CNN's FP problem is concentrated in **silent recording-sessions** where mouse
pairs aren't vocalizing — the model fires on background acoustics with nothing
real to detect. Examples observed during labeling:
- `131204_1800_m{1,2,3,4}` — 0 USVs across 18 sampled events
- `131211_1800_m2fm3` — 1 USV across 11 events (~9%)
- `131205_1400_m{1,2,6}` — 0/9 USVs
- `131209_1000_m{2,3,4}` — high-FP-rate sessions (multiple full-chunk-spanning detections, mostly noise)

By contrast: `131204_1400_m3fm3` was 80% USV (heavy vocalizer at that session).
This is per-pair-per-time-of-day vocal behavior variance, NOT a labeling artifact.

**For data splits in Phase 3 (training):** naive random splits will leak
recording context across train/val. **Use recording-stratified splits** —
hold out entire `original_filename`s from train.

## Files to read first (next chat)

1. **This file** — covers Phase 0 + Phase 1 outcomes
2. `docs/handoffs/2026-05-09_lab-cnn-finetune-handoff.md` — original spec; Phases 2–5 details still apply
3. `data/lab_finetune_v1/labels_audit_72.csv` — audit labels (71 unique events)
4. `data/lab_finetune_v1/mining_candidates_500/candidates_seed42.csv` — mining labels (500 events)
5. `scripts/extract_hard_negatives.py` — wild-data precedent for the extraction step (Phase 2). DO NOT modify; lab extraction should be a new sibling script.
6. `docs/handoffs/v2-full-pipeline-results.md` — production training pipeline precedent

## Phase 2 — Extract labeled lab events to training-format PNGs (NEXT)

Need to convert 571 labeled events → 100-column training PNGs matching the existing pipeline format (`data/training/hard_negatives/`, `data/training/hard_usvs/`).

**Recommendation:** write a NEW script `scripts/extract_lab_finetune_pngs.py` that:
- Reads both CSVs (audit_72 + mining_500).
- Joins to `merged_events_with_filter.parquet` by `(chunk_stem, event_idx)` to get `start_col`/`end_col`/timing.
- Loads chunk WAVs from `USV_lab_131204_chunked_2s_full/<stem>.wav`.
- Uses the EXACT same pipeline as `scripts/extract_hard_negatives.py` (global MAD norm, 100-column window centered on event, magma colormap, 256px height) — import constants from there or `corpus.py` (do NOT redeclare). Constants:
  - `WINDOW_COLUMNS = 100`
  - `IMAGE_HEIGHT = 256`
  - `COLORMAP = "magma"`
  - `MAD_VMIN_SCALE = 2.0`, `MAD_VMAX_SCALE = 4.0`
  - From `corpus.py`: `SAMPLE_RATE_HZ`, `STFT_N_FFT`, `STFT_HOP`, `USV_FREQ_MIN_HZ`, `USV_FREQ_MAX_HZ`
- Centers the window on event center (between `start_col` and `end_col`).
- Writes to `data/training/lab_finetune_v1/{usv,noise}/<chunk_stem>_ev<idx>.png`.
- Writes a manifest CSV with `(png_path, label, source_recording, original_chunk)` for the train/val split step.

**Validation step before training (mandatory per original handoff):**
spot-check 50 random extracted PNGs by eye. If any noise-labeled PNG looks like
a clean USV (or vice versa), the extraction pipeline has a bug.

## Phase 3 — Build train/val CSVs and run fine-tune

**Approved parameters from this session:**
- Model size: **mid** (filters [32, 96, 192], dense_units 64, ~207K params) — same as production. Do NOT scale up; the user's labeled set is small and bigger architectures overfit.
- Init from `models/hard_neg_retrain/best_model.pt` checkpoint.
- **Class weight: `pos_weight ~8×`** (vs production's ~35×). Reason: production was recall-biased to fix wild-data USV detection; lab fine-tune needs to *reduce* FPs, so we lower the recall bias. User confirmed this choice.
- **Frozen backbone + last-layer training** as the first attempt (less risk of catastrophic forgetting on wild data). Escalate to full-backbone with low LR only if precision insufficient after first attempt.
- **Recording-stratified split:** train/val/test should not share `original_filename`. The 17 lab recordings are: `131204_1400_m{1,2,3,4,5,6}fm{...}`, `131204_1800_m{1,...}`, `131205_1000_m{...}`, `131205_1400_m{...}`, `131205_1800_m{...}`, `131208_1000_m{...}`, `131209_1000_m{...}`, `131210_1000_m{...}`, `131211_1000_m{...}`, `131211_1800_m{...}`, `131212_1000_m{...}`, `131217_1400_m{...}`, `131218_1000_m{...}`. Hold out 2–3 recordings entirely for val.
- Mix lab labels with the **existing wild training data** from `splits/{train,val,test}.csv`. Keep wild data dominant in count but lab data weighted appropriately so it influences the gradient. Tune the lab/wild ratio empirically — start with 1:1 in batches via class-balanced sampling (lab and wild balanced equally).
- Output dir: `models/lab_finetune_v1/`.

**Do NOT touch:**
- `models/hard_neg_retrain/*` — production model is frozen.
- `src/usv_spectrogram/corpus.py`, `src/usv_spectrogram/detection/extraction_config.py` — CNN FREEZE.
- `src/usv_spectrogram/app/core/audio_loader.py`, `src/usv_spectrogram/app/core/denoise.py`, `scripts/run_batch_detection.py` (subtraction plumbing) — leave default-off subtraction code in place per the original handoff.

## Phase 4 — Refit calibration sidecars

After fine-tune (per original handoff vault constraint #3):
- `models/hard_neg_retrain/temperature.json` was fit on wild data; do NOT reuse. Fit a new `TemperatureScaler` on a **lab held-out set** (10–15% of lab data, recording-stratified).
- `models/hard_neg_retrain/fp_filter.pkl` was trained on wild events; refit `FalsePositiveFilter` on lab events.
- Output to `models/lab_finetune_v1/{temperature.json, fp_filter.pkl}`.

## Phase 5 — Validation (NON-NEGOTIABLE)

### Wild regression (HARD GUARD)
Run `lab_finetune_v1` on `5970_manual_review_reviewed/` (49 wild WAVs) and
compare to production-model output at `results/batch_5970_v2_full/`:
- Event count must match within **1% absolute**.
- Tier distribution must match within **1% absolute**.
- Per-chunk `max_probability` distribution should be virtually identical.

If delta > 1%: **fine-tune is rejected**. Catastrophic forgetting on wild data
is the dominant failure mode for naive fine-tuning; this guard catches it.

Use the canonical batch invocation from `CLAUDE.md`:
```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir 5970_manual_review_reviewed/ \
    --model models/lab_finetune_v1/best_model.pt \
    --output-dir results/batch_5970_finetune_v1_regression/ \
    --temperature models/lab_finetune_v1/temperature.json \
    --fp-filter models/lab_finetune_v1/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```

(Hysteresis-config stays from production — refitting it is a Phase 6 deferred
optimization per the original handoff.)

### Lab full-batch + audit (success metric)
Run `lab_finetune_v1` on the full 26,309-chunk lab batch:
```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir USV_lab_131204_chunked_2s_full/ \
    --model models/lab_finetune_v1/best_model.pt \
    --output-dir results/batch_lab_131204_finetune_v1/ \
    --temperature models/lab_finetune_v1/temperature.json \
    --fp-filter models/lab_finetune_v1/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
    # NOTE: no --subtract-baseline. Fine-tune handles bands natively.
```

Then re-run `scripts/audit_lab_auto_accepts.py` against the new parquet to
sample stratified PNG buckets. **User to eyeball-audit** the new typical bucket
and confirm noise rate dropped from ~73% to **<20%** (the success criterion
from the original handoff). Cleanest-bucket recall must remain ≥90%.

If lab auto_accept count falls below 15,000 (vs original 32,724): the model
over-suppressed. Expected range per original handoff: 25,000–32,000.

## Vault constraints to flatten (next chat may not have vault access)

1. **Mid-c CNN architecture** ([32, 96, 192] / dense 64 / 207K params, sample-to-param ratio ~71:1). Don't scale up.
2. **Active learning cycle is the established methodology** — this fine-tune is one cycle of label-train-evaluate-mine.
3. **Modern CNNs are systematically miscalibrated** — must refit temperature on held-out lab data.
4. **CORPUS-INVARIANT** — `ExtractionConfig.freq_min_hz`, `freq_max_hz`, `n_fft`, `hop_length` are FROZEN. Spectrogram extraction code MUST NOT change. Import constants from `src/usv_spectrogram/corpus.py`; never redeclare.

## Workflow notes worth preserving

- **Labeling protocol:** Option B (user dictates verdicts on a batch loaded into `data/lab_finetune_v1/labeling_queue/`). Stable IDs (typXX, borXX, lngXX, hovXX, minXXX) prefix the PNG filenames so the user can sort by name and dictate in order.
- **Long-event labeling:** broad-view labels need center-zoom verification because the trainer's 43 ms window is centered on the event center, not the broad box. Use `scripts/zoom_event_png.py --center-only-ms 200` for re-verification of any Y verdict on dur > 600 ms.
- **The 43 ms training window** is the load-bearing fact. Labels must describe what's in the centered 43 ms window, NOT the broader 2-second chunk. Brief USVs (10 ms) can still be valid Y if the chirp shape is at center.

## File inventory (added this session)

**Created:**
- `data/lab_finetune_v1/labels_audit_72.csv` — 72 rows, 45 USV / 26 NOISE (71 unique events)
- `data/lab_finetune_v1/mining_candidates_500/candidates_seed42.csv` — 500 rows, 152 USV / 348 NOISE
- `data/lab_finetune_v1/mining_candidates_500/renders/` — 500 mined-event PNGs
- `data/lab_finetune_v1/labeled/` — 575 archived PNGs (audit + mining + center-zoom verifications)
- `scripts/mine_lab_finetune_candidates.py` — stratified mining of auto_accept events with already-labeled exclusion
- `scripts/zoom_event_png.py` — render zoom or center-only PNG of a single event for labeling re-verification

**Unchanged (do not touch):**
- `models/hard_neg_retrain/*` — production model
- `src/usv_spectrogram/corpus.py`, `extraction_config.py` — CNN FREEZE
- The subtraction plumbing in `app/core/audio_loader.py`, `app/core/denoise.py`, `scripts/run_batch_detection.py` (default-off, leave as-is)

## Open questions for next chat

1. **Lab/wild mixing ratio** — what fraction of each batch is lab vs wild data? Start at 1:1 class-balanced and tune empirically. Document chosen ratio + rationale in the v1 training report.
2. **Frozen-backbone first attempt** — if precision insufficient, when does next chat escalate to full-backbone? Suggested: if held-out lab precision <85% with frozen backbone, unlock the last 1–2 conv blocks with LR/10 of last-layer LR.
3. **Lab cohort generalization** — 131204 is one recording session. Cohort 9252 (8 sessions, batch detection in progress) may have different equipment-line patterns. Plan for a follow-up 9252-fine-tune cycle if v1 doesn't transfer.
4. **Hysteresis-config refit** — defer until end. Only re-fit if final lab batch shows obvious threshold mismatch (e.g., auto_accept count outside expected 25k–32k range).

## Worth remembering

- **The OOD failure mode is per-recording-session, not uniform.** Silent sessions (mouse pair not vocalizing) produce near-100% FPs. Active sessions produce mostly correct detections. The fine-tune needs to learn "lab acoustic background ≠ USV" without losing detection on real lab USVs.
- **High-SEF events (>0.50) are mostly real USVs.** The post-hoc `noise_band_overlap` filter targets the wrong population. The fine-tune should NOT learn to suppress these.
- **The wild model's failure on lab data is concentrated in the typical-bucket signature** (low SEF, short duration, "empty-window" detections). 350 confirmed hard negatives from this stratum are the highest-value training material.
- **User's labeling burden is high but tolerated.** ~571 events labeled in two ~30-min sessions. User explicitly chose verified labeling over a 32% poison rate from bulk-labeling. Future labeling rounds should default to verified path.
- **Recording-stratified splits are mandatory** to avoid val-set leakage from the same `original_filename`.

## References

- `docs/handoffs/2026-05-09_lab-cnn-finetune-handoff.md` — original handoff (this supersedes Phases 0+1)
- `docs/handoffs/2026-05-08_pre-cnn-spectral-subtraction-lab.md` — abandoned subtraction approach; band-overlap pattern findings
- `docs/handoffs/v2-full-pipeline-results.md` — production training pipeline precedent
- `data/corpus_facts/lab_131204.json` — lab corpus facts (validated stationary bins at 50.4, 51.0, 63.3, 63.9 kHz)
