# Plan: Detecting the 1960 False Positive Pattern via Spectral Flatness

## Goal

Build a post-processing filter that catches the "1960 problem" — broadband noise burst + low-frequency streak false positives — without flagging real USV detections. This is Phase 1: **profiling/discovery**, not a production filter yet.

## Background & Reasoning

### Why previous approaches failed

All five previous attempts used **aggregate energy statistics** — comparing how much energy is in the low-frequency band vs. the USV band, or what fraction of bins are active. These fail because:

- Cage noise below 30 kHz is persistent in nearly all recordings, so low-band energy is always present
- The broadband burst distributes energy across ALL frequencies including the USV band (35–110 kHz), inflating the high-band statistics
- The 1960 pattern's metrics overlap completely with normal recordings at every percentile

The fundamental issue: the approaches measure **how much energy** but the discriminative signal is about **energy shape/structure**.

### The key visual observation

Looking at all 12 spectrograms (10 problem + 2 known-good), the discriminative pattern is:

**Real USVs (0000053, 0000054):** Inside detection windows, there are visible *narrow-band ridges* — thin bright traces concentrated in a few adjacent frequency bins per time column, typically sweeping through 50–100 kHz. The energy is **spectrally concentrated/peaked**.

**1960 pattern (all 10 problem files):** Inside detection windows, after the broadband vertical flash, the remaining columns have NO narrow-band structure above ~35 kHz. The energy above 35 kHz is either noise-floor or diffuse broadband — it's **spectrally flat** in the USV range.

This is a shape distinction, not an energy magnitude distinction. The right measurement is **spectral flatness** (also called Wiener entropy): the ratio of the geometric mean to the arithmetic mean of the power spectrum in a given frequency band. This yields:
- 0 when energy is concentrated in a single bin (pure tone / narrow-band ridge)
- 1 when energy is uniformly distributed across all bins (flat noise)

A real USV column will have low spectral flatness in the 35–110 kHz range. A 1960-pattern column will have high spectral flatness there.

### Why spectral flatness should work where band ratios didn't

Band energy ratios ask: "Is there more energy above 35 kHz or below 35 kHz?"
Spectral flatness asks: "Is the energy above 35 kHz **shaped like a tone** or **shaped like noise**?"

The broadband burst puts energy everywhere, so the band ratio sees high energy in the USV band and says "USV-like!" But spectral flatness looks at the *distribution shape* within the band — the broadband burst makes the USV band spectrally **flat**, not peaked. That's the opposite of what a real USV looks like.

### What could go wrong

1. **Weak USVs with low SNR** might also have high spectral flatness because the narrow-band peak barely rises above the noise floor — we'd get false rejections of genuine faint USVs. This is why we profile first before committing to thresholds.

2. **Mixed events** — if a detection window contains BOTH a broadband burst AND a real USV (which the handoff doc says can happen), the spectral flatness averaged across the window might be ambiguous. Per-column analysis (not window-average) should handle this: the real-USV columns will show low flatness, the burst columns will be masked out.

3. **Spectral flatness might not separate cleanly** — maybe the noise floor in the USV band already has some structure (e.g., harmonics from cage noise, electronic interference lines visible at ~65 kHz in some images). Profiling will reveal this.

4. **The 35 kHz lower bound might need tuning** — if real USVs have fundamentals down to 25 kHz, using 35 kHz as the lower bound might clip them. But the handoff doc says real USVs are 25–110 kHz, and the 1960 streak sits below 25–30 kHz, so 35 kHz should be safe. Check the actual frequency content of real USV detections to confirm.

## Implementation: Phase 1 — Profiling Script

### Step 0: Discover the repo structure

Find and examine:
- The detection JSON format in `results/batch_5970/detections/` — what fields are available (start_col, end_col, probabilities, etc.)
- How spectrograms are computed — find the existing spectrogram computation code (likely in `usv_spectrogram/` somewhere). We want to reuse the EXACT same spectrogram parameters (n_fft=512, hop=128, sr=300000, range 20–120 kHz) to ensure our analysis matches what the CNN sees.
- Where the WAV files live
- The summary parquet file at `results/batch_5970/summary_full.parquet` — check what columns/tiers are available

Report back what you find before proceeding to implementation.

### Step 1: Load and compute spectrograms for test files

For each of the 10 known 1960-problem files and ~50 randomly sampled known-good auto-accept files:
1. Load the WAV file
2. Compute the spectrogram using the same parameters as the pipeline
3. Load the detection JSON to get detection window boundaries (start_col, end_col)

### Step 2: Per-column spectral flatness computation

For each detection window in each file:

```
For each spectrogram column within [start_col, end_col]:
    1. Extract the power values in the USV band (35–110 kHz)
       - With sr=300000 and n_fft=512, freq resolution = 300000/512 ≈ 585.9 Hz/bin
       - 35 kHz → bin index ≈ 35000/585.9 ≈ 60 (but we're only displaying 20–120 kHz, 
         so the indexing depends on how the spectrogram is sliced — discover this from the code)
       - 110 kHz → bin index ≈ 110000/585.9 ≈ 188
    
    2. Compute active-bin fraction for broadband detection:
       - Threshold the column (e.g., > median + 6 dB)
       - If > 60% of ALL frequency bins are active, mark as "broadband column" and skip
    
    3. For non-broadband columns, compute spectral flatness in 35–110 kHz:
       - The spectrogram is in dB! Convert back to linear: power_values = 10^(dB_values / 10)
       - power_values = 10**(spectrogram[usv_band_bins, column] / 10.0)
       - geometric_mean = exp(mean(log(power_values + epsilon)))
       - arithmetic_mean = mean(power_values)
       - spectral_flatness = geometric_mean / arithmetic_mean
       - This gives a value in [0, 1]
    
    4. Classify the column as "tonal" if spectral_flatness < threshold (start with 0.5, will tune)
```

### Step 3: Per-detection aggregation

For each detection:
- `max_consecutive_tonal`: longest run of consecutive tonal (non-broadband, low-flatness) columns
- `tonal_fraction`: fraction of non-broadband columns that are tonal
- `mean_flatness`: mean spectral flatness across non-broadband columns
- `min_flatness`: lowest spectral flatness in any column (captures the strongest tonal peak)
- `n_broadband_columns`: how many columns were masked as broadband
- `n_total_columns`: total columns in detection window

### Step 4: Output and comparison

Produce a CSV or printed table with:
- File name
- Label (1960_problem or auto_accept)
- All Step 3 metrics for each detection in the file

Then compute summary statistics:
- Distribution of `max_consecutive_tonal` for problem files vs. good files
- Distribution of `mean_flatness` for problem files vs. good files
- Distribution of `min_flatness` for problem files vs. good files

**The key question: do these distributions separate?** If the problem files cluster at high flatness / low tonal-column count while the good files cluster at low flatness / high tonal-column count, we have a viable filter. If they overlap, we need a different approach.

### Step 5: Optional — generate diagnostic plots

For each of the 10 problem files and 5 randomly chosen good files, produce a matplotlib figure showing:
- The spectrogram patch inside the detection window
- A row of per-column spectral flatness values plotted below it
- Broadband columns highlighted in red
- Tonal columns highlighted in green

These will let Shachar visually verify the approach is capturing the right structure.

## File Locations

```
WAV files:              5970/ (6400 files, nested in USV1-5/usv_lmt_034/)
                        Also: 5970_reviewed/, 5970_manual_review/, 5970_manual_review_reviewed/
Detection JSONs:        results/batch_5970/detections/ (one JSON per recording)
Summary:                results/batch_5970/summary_full.parquet (columns: stem, tier, n_events, max_confidence, etc.)
Spectrogram loader:     src/usv_spectrogram/app/core/audio_loader.py (AudioLoader class — use this!)
Extraction config:      src/usv_spectrogram/detection/extraction_config.py (n_fft=512, hop=128, sr=300000, 20-120kHz)
Existing analysis:      scripts/flag_broadband_fp.py (previous failed attempts, useful reference for WAV path resolution)
Example PNGs:           results/batch_5970/1960_problem_examples/ (10 problem + 2 good)
```

**Important:** Use `AudioLoader` from `audio_loader.py` to compute spectrograms — it handles all the STFT
parameters correctly and returns normalized dB spectrograms matching what the CNN sees. Don't reimplement
the STFT. The spectrogram is stored as dB values with normalize_magnitude=True (max = 0 dB).

**WAV path resolution:** Files are nested in subdirectories. Use `rglob` to find them:
```python
for search_dir in [Path('5970'), Path('5970_reviewed'), ...]:
    matches = list(search_dir.rglob(f"{stem}.wav"))
```

## The 10 Known 1960-Problem Files

```
2024-09-30_17-45-49_0001960
2024-09-30_19-00-57_0002431
2024-09-30_19-20-03_0002522
2024-09-30_22-36-23_0003502
2024-09-30_22-36-29_0003503
2024-09-30_23-37-29_0003781
2024-09-30_23-39-35_0003794
2024-10-01_12-20-03_0005107
2024-10-01_17-18-59_0005656
2024-10-01_18-29-16_0006086
```

## Known-Good Reference Files (for sanity checks)

```
2024-09-30_11-22-17_0000053 (GOOD — real USVs)
2024-09-30_11-22-19_0000054 (GOOD — real USVs)
```

## Success Criteria

Phase 1 succeeds if the profiling shows a metric (or combination of metrics) where:
- ALL 10 problem files fall on one side of a threshold
- At least 90% of the 50 sampled good files fall on the other side

If this holds, Phase 2 would be integrating the filter into the post-processing pipeline (the existing four-stage pipeline with hysteresis detection, event-level logistic regression, etc.). The spectral flatness features could feed into the Stage 3 event-level classifier as additional features, rather than being a standalone hard filter.

## Alternative Approach to Keep in Mind

If spectral flatness also fails to separate, the conclusion is that **CNN retraining with hard negatives is the only viable path**. We have 10 labeled 1960-problem files and would need ~50-100 total hard negatives. This is a last resort but likely the most durable solution.

Another alternative: a **2D ridge filter** (Hessian-based ridge detection from image processing) applied to the spectrogram patch might work better. It directly detects the thin bright lines that characterize real USVs. More computationally expensive but captures exactly the structural difference visible to the eye. We'd measure "total ridge strength" in the 35–110 kHz band per detection window. But start with spectral flatness — it's simpler and faster.
