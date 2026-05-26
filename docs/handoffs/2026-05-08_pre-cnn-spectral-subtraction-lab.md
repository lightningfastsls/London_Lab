# Handoff: Pre-CNN Spectral Subtraction for Lab Detection
Date: 2026-05-08
From: Claude Code (prior chat)
To: Claude Code (next chat)

## Goal

Wire `scripts/preview_spectral_subtraction.py`'s temporal-baseline subtraction
into the production audio loader / inference pipeline as a **lab-only opt-in
preprocessing step** that runs *before* the CNN sees the spectrogram. Replace
the layered post-hoc filtering approach with a single intervention at the
right point in the pipeline.

**Acceptance:** Lab batch re-run with subtraction enabled produces lower
auto_accept noise rate than current 14% (target: <5%, measured by re-running
the same audit pipeline `scripts/audit_lab_auto_accepts.py`). Wild-mouse
detection (5970, 3452, 9252) is unaffected — the flag defaults off.

## Why This and Not More Post-Hoc Filters

User's exact words this session: *"I don't like that we are adding more and
more levels."* The prior chat had been stacking filters (stationary-band SEF
filter, May-6 max_duration_ms gate, proposed bandwidth filter) on top of CNN
output, each addressing a slice of the noise problem. The cleaner architecture
is to denoise at the source so the CNN never trains its confidence on the
noise patterns in the first place.

**Critical empirical finding from this session that motivated the pivot:**
A stratified eyeball audit of 72 lab `auto_accept` PNGs labeled by the user
showed:

| Bucket | Sample | True USV | Noise | Extrapolated noise |
|---|---|---|---|---|
| cleanest (SEF=0, 50–250 ms) | 15 | 15 | 0 | 0 of 8,497 |
| typical (SEF 5–15%, 50–250 ms) | 15 | **4** | **11** | **~4,057 of 5,557** |
| borderline (SEF 20–50%) | 10 | 5 | 5 | ~256 of 512 |
| high_overlap_survivors (SEF >50%) | 22 | TBD | TBD | TBD of 22 |
| long_event_survivors (>600 ms) | 10 | 4 | 6 | ~150 of 252 |

Headline: **~14% of `auto_accept` events on lab data are noise**, dominated
by typical-bucket false positives (low SEF, short duration) — i.e., NOT
band-overlap noise that the post-hoc filter targets.

Three of four typical-noise PNGs viewed had **empty event windows** — no
visible USV trace and no broadband click. Hypothesis: the wild-trained CNN
hallucinates USV-shape statistics from subtle band-pattern features in lab
acoustic backgrounds it never saw during training. **No spectrogram-domain
post-hoc filter can detect this, because there is no spectral feature in the
event window for the filter to reject.** The fix has to act on the input the
CNN sees.

## Files to Modify

1. **`src/usv_spectrogram/app/core/audio_loader.py`**
   Add `subtract_baseline: bool = False` kwarg to `AudioLoader.load(...)` (or
   to `AudioLoader.__init__`, whichever fits the existing pattern better).
   When True, apply temporal-baseline subtraction to the linear-magnitude
   spectrogram before computing dB and returning AudioData. Default off so
   wild-mouse runs are byte-identical.

2. **`src/usv_spectrogram/app/core/denoise.py`** *(new)*
   Extract the subtraction math currently in `scripts/preview_spectral_subtraction.py`
   into an importable function: `subtract_temporal_baseline(spec_linear: np.ndarray,
   percentile: float = 10.0) -> np.ndarray`. Return cleaned linear-magnitude
   spectrogram. Keep the existing preview script working by importing from the
   new module.

3. **`scripts/run_batch_detection.py`**
   Add `--subtract-baseline` flag. When set, pass through to AudioLoader.
   Default off. Document in `docs/modules/recording-triage.md` or similar.

4. **`scripts/audit_lab_auto_accepts.py`** *(may need update)*
   Already exists from this session. Should work unchanged on the re-run
   parquet, but verify it doesn't hard-code `merged_events_with_filter.parquet`
   (the post-hoc filter run); we want to audit the *new* batch directly.

## Relevant Constraints (from vault)

1. **Per-frequency-bin normalization removes frequency-dependent energy bias**
   Source: `notes/per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input.md`
   Verified: 2026-05-08
   Note: Spectral subtraction is structurally similar — both per-bin operations.
   Read this note before implementing to confirm conventions for per-bin
   stat computation aren't being violated.

2. **Normalization statistics must be computed on training set only — data leakage risk**
   Source: `notes/normalization statistics must be computed on training set only to prevent data leakage.md`
   Verified: 2026-05-08
   Caveat for this task: spectral subtraction here is *per-chunk* (10th
   percentile within each 2-second chunk), not training-set statistics, so
   this constraint is informational. But document the choice clearly.

3. **Per-recording normalization compensates for varying noise floors across recording sessions**
   Source: `notes/per-recording normalization compensates for varying noise floors across recording sessions.md`
   Verified: 2026-05-08
   Relevant: lab vs wild rigs have very different noise floors. Per-chunk
   subtraction is even more local than per-recording — likely the right
   granularity for lab equipment lines that are stable within a chunk but
   vary by recording session.

4. **Electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs**
   Source: `notes/electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs.md`
   Verified: 2026-05-08
   This is exactly the failure mode lab data exhibits — confirmed by
   `data/corpus_facts/lab_131204.json` flagging stationary bins at 50.4,
   51.0, 63.3, 63.9 kHz on chunk 194. Spectral subtraction targets these
   directly.

5. **PCEN is the gold standard adaptive normalization in bioacoustic literature**
   Source: `notes/PCEN is the gold standard adaptive normalization in bioacoustic literature.md`
   Verified: 2026-05-08
   Alternative to plain spectral subtraction. PCEN does adaptive AGC + power
   compression. **Worth flagging to user before implementation:** plain Boll
   subtraction is simpler and the math is already written; PCEN is more
   sophisticated but unimplemented. Recommend simple-first, escalate if
   results disappoint.

6. **CNN ExtractionConfig is locked to training grid — DO NOT change `freq_min_hz`/`freq_max_hz`/`n_fft`/`hop_length`**
   Source: `CLAUDE.md` — corpus invariant + CNN FREEZE constraint
   Verified: 2026-05-08
   The subtraction happens in the spectrogram domain *after* STFT but *before*
   normalization → CNN. No corpus parameters change. Subtraction is purely
   a magnitude transform; the time/frequency axes are untouched.

## Context

### Architecture you're plugging into

```
WAV → AudioLoader.load() → AudioData{spectrogram_db, times, frequencies}
                                       │
                                       ▼
                    SlidingInference.infer(spec_db, ...)
                       │  ├─ _normalize_window_to_training_distribution()
                       │  ├─ _apply_mad_normalization()
                       │  └─ CNN forward
                       ▼
                    Per-window probabilities
                       │
                       ▼
                    Hysteresis → triage → events parquet
```

The subtraction insertion point is **inside `AudioLoader.load()`** between
`np.abs(stft)` and the dB conversion, so everything downstream — sliding
inference, MAD normalization, hysteresis — sees the cleaned spectrogram with
no other changes.

### The math (already in `preview_spectral_subtraction.py`)

```
spec_linear = |STFT(audio)|                          # per-chunk
baseline = np.percentile(spec_linear, 10, axis=time) # per-bin temporal floor
cleaned = np.maximum(spec_linear - baseline[:, None], epsilon)
spec_db = 20 * np.log10(cleaned + epsilon)
```

Linear-magnitude subtraction (correct), 10th percentile per bin (robust to
USV bursts which are ≤10% of frames in any bin), `np.maximum` floor to
prevent log of negatives.

### Why this is structurally cleaner than post-hoc filters

- Single intervention at the *input* to inference
- Wild-mouse pipeline byte-identical (flag defaulted off)
- Existing post-hoc filters (`apply_stationary_band_filter.py`, May-6
  `max_duration_ms` gate) **stay in place but become redundant for the
  cases subtraction handles** — leave them as defense in depth, don't
  remove them in this pass

### Production model files referenced

- Model: `models/hard_neg_retrain/best_model.pt`
- Temperature: `models/hard_neg_retrain/temperature.json` (calibrated on
  WILD data — note this in the open-questions)
- FP filter: `models/hard_neg_retrain/fp_filter.pkl`
- Hysteresis: `models/hard_neg_retrain/hysteresis_optimization_v2.json`

Standard production batch invocation in `CLAUDE.md` under "Running Batch
Detection."

### Lab data location

- WAVs: `USV_lab_131204/` (17 raw recordings)
- 2-second chunks: `USV_lab_131204_chunked_2s_full/` (26,310 chunks)
- Current batch results: `results/batch_lab_131204_full/` (preserve — this
  is the baseline to compare against)
- New batch should go to: `results/batch_lab_131204_subtracted/` (suggested
  name)

### Audit infrastructure already built this session

- `scripts/audit_lab_auto_accepts.py` — stratified bucket sampler + PNG
  renderer. Reuse for the post-subtraction audit.
- `results/batch_lab_131204_full/audit_2026-05-08/` — 72 PNGs from the
  baseline audit, with user labels in this conversation's transcript.
  Keep around as the comparison reference.

## Validation

### Unit-level

1. `python -m py_compile` on each modified file.
2. Add unit test in `tests/test_denoise.py` (new):
   - Synthetic spectrogram with one stationary band (constant amplitude
     across all time frames at one frequency bin) and one transient
     (single-frame Gaussian blob at a different bin).
   - After subtraction: stationary band → ~0; transient → preserved.
3. Run existing test suite: `.venv/bin/python -m pytest tests/ -q` —
   should be green; subtraction is opt-in so no existing test changes
   behavior.

### Integration

1. Run subtraction on a single lab chunk known to have strong bands (e.g.,
   `131204_1400_m4fm4_chunk_194` — the chunk that flagged 4 stationary bins
   at 50.4, 51.0, 63.3, 63.9 kHz per `data/corpus_facts/lab_131204.json`).
   Inspect the resulting spectrogram visually — bands should be gone, any
   real USV should remain.
2. Verify a wild-data chunk run with `subtract_baseline=False` produces
   byte-identical output to the existing pipeline (regression guard for
   wild-mouse cohorts).

### End-to-end re-audit

1. Re-run lab batch with `--subtract-baseline` into
   `results/batch_lab_131204_subtracted/`.
2. Re-run `scripts/audit_lab_auto_accepts.py` against the new parquet
   into `results/batch_lab_131204_subtracted/audit_post_subtraction/`.
3. **Compare the same 5 buckets head-to-head** with the baseline audit at
   `results/batch_lab_131204_full/audit_2026-05-08/`. Specifically: render
   the same chunks-events pairs from both runs side-by-side if possible,
   so the user can see the per-detection effect.
4. **Have the user re-label** the typical bucket (this is the dominant
   noise source — the most informative bucket to re-eyeball). If typical
   noise rate falls from 73% to <20%, subtraction works; if it stays
   high, the empty-window FPs aren't band-pattern-driven and we need
   the calibration/retrain path instead.

### Numerical success criteria

| Metric | Baseline | Target | Disaster |
|---|---|---|---|
| Lab `auto_accept` count | 32,724 | 25,000–32,000 | <15,000 (over-subtraction killed real USVs) |
| Typical-bucket noise rate (eyeball, n=15) | 73% | <20% | Unchanged ~73% |
| Cleanest-bucket noise rate (eyeball, n=15) | 0% | 0% | Any nonzero (regression) |
| Wild-mouse 5970 detection count delta | n/a | 0 (flag off) | Any change |

## Open Questions / Known Risks

1. **Empty-window FPs may not be band-driven.** If subtraction kills the
   bands but the typical-bucket noise rate stays high, the failure mode
   is CNN miscalibration on quiet lab acoustics rather than band-pattern
   triggering. Fallback path: lab-specific temperature recalibration
   (`models/hard_neg_retrain/temperature.json` was fit on wild data).
   This was discussed with the user this session — they preferred to try
   subtraction first since it's a single intervention.

2. **Musical noise / over-subtraction.** Spectral subtraction can leave
   sparse sinusoidal artifacts that *might* fool the CNN in different
   ways. The 10th-percentile baseline is a conservative choice (literature
   standard). If artifacts appear in the integration step (a real USV is
   still recognizable but distorted), back off to a higher percentile
   (15th, 20th) at the cost of less aggressive band removal.

3. **Per-chunk vs. per-recording baseline.** Current implementation in
   the preview script computes baseline per 2-second chunk. If equipment
   bands drift across the full ~5-minute recording, per-recording baseline
   would be more accurate. But per-chunk is simpler, matches existing
   chunked pipeline architecture, and the lab `corpus_facts` validation
   was already done at chunk level. Stay with per-chunk unless integration
   testing reveals chunk-boundary artifacts.

4. **Subtraction interacts with MAD normalization in `sliding_inference.py`.**
   Verify the existing MAD normalization (`_apply_mad_normalization`,
   line ~388) still produces sensible window-level statistics after
   subtraction. The subtracted spectrogram has different distribution
   shape; MAD should still work but worth eyeballing one window.

5. **Should existing post-hoc filters be removed or left as defense
   in depth?** Recommendation: leave them in place this pass. The May-6
   `max_duration_ms=600` gate and the `noise_band_overlap` SEF tags both
   provide cheap regression guards. Re-evaluate after the post-subtraction
   audit shows what subtraction actually achieves.

6. **High-overlap-survivors labeling not finished.** This session ended
   before the user finished labeling the 22 high_overlap_survivors PNGs.
   Useful data point to collect during the post-subtraction audit since
   that bucket should largely *disappear* if subtraction works.

## Worth Remembering for Claude

- The lab CNN pipeline currently has ~14% `auto_accept` noise contamination
  on the 131204 cohort (26,310 chunks, 32,724 auto_accept events). It is
  **not science-grade for behavioral analysis** in its current form,
  regardless of how the post-hoc filter is tuned.
- The wild-trained CNN (`models/hard_neg_retrain/best_model.pt`) was never
  exposed to lab equipment-noise spectrograms. The "empty window" hallucinations
  observed on lab data are likely an out-of-distribution generalization
  failure rather than a tunable threshold problem.
- All filters added 2026-04-27 → 2026-05-06 (stationary-band SEF tagging,
  long-event `max_duration_ms` gate) operate **post-CNN** and cannot
  address noise the CNN itself was confused by during inference.
- The user explicitly rejected the "stack more layers" approach this
  session. Future Claude instances should propose root-cause fixes (input
  preprocessing, calibration, retraining) before adding more output-side
  filters.

## References

- This session's transcript covers the full diagnostic: 5-bucket audit,
  user labeling, the pivot from filter-stacking to pre-CNN subtraction.
  The audit PNGs at `results/batch_lab_131204_full/audit_2026-05-08/`
  are the labeled reference set.
- `docs/handoffs/2026-05-06_lab-detection-long-event-qc.md` — the May 6
  long-event QC handoff that this work supersedes architecturally
  (without removing its production code).
- `docs/handoffs/HANDOFF_05_LAB_DATA_PIPELINE.md` — original lab data
  pipeline planning doc; predates the data arrival, useful for run-order.
- `data/corpus_facts/lab_131204.json` — the lab corpus facts including
  the noise filter v2 parameters and validated stationary bins on
  reference chunks.
- `scripts/preview_spectral_subtraction.py` — the math you're promoting.
- `scripts/audit_lab_auto_accepts.py` — the audit tool to re-run
  post-subtraction.
