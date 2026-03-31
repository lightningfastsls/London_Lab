# Handoff: The "1960 Problem" — False Positive Pattern in USV Detection

## Context

We have a CNN-based ultrasonic vocalization (USV) detection pipeline for mouse recordings at 300 kHz. The pipeline processes ~6,400 WAV files and classifies them into tiers: auto-accept (1,344), manual-review (70), auto-reject (4,986).

During manual review of the 70 ambiguous files, we found **10 files** (14%) with a systematic false positive pattern we call the "1960 problem" (named after recording 0001960 where it was first spotted).

## The 1960 Pattern

The pattern consists of two parts in sequence:
1. **A broadband noise burst** — a brief flash of energy across many frequency bins simultaneously (but not all — typically 20-50% of bins, not full-range)
2. **A low-frequency streak** — immediately after the burst, sustained energy below ~25-30 kHz in consecutive spectrogram columns

The CNN misclassifies the low-frequency streak as a USV. Real mouse USVs are narrow-band calls typically between 25-110 kHz. The streak sits at the bottom of this range and fools the classifier.

**Important nuances:**
- Real USVs *can* have energy down to ~25 kHz (the boundary is not sharp)
- Real USV recordings also have persistent cage noise below 30 kHz (always present)
- The broadband burst is partial (20-50% of bins) — when broadband noise covers >50% of bins, the CNN actually handles it correctly and doesn't trigger false detections
- Many legitimate recordings have both broadband noise AND real USVs in the same detection window

## What We Tried

### Approach 1: Broadband fraction + low-freq streak (sequential)
- Check each detection window for a column where ≥55% of frequency bins are active (broadband), followed by consecutive columns with energy below 30 kHz (streak)
- **Result: 602/1,344 files flagged** — way too many. Cage noise below 30 kHz is persistent in nearly every recording, so the streak criterion is almost always met.

### Approach 2: Inverted broadband range (20-50% instead of >55%)
- Since the 1960 pattern has partial broadband, we tried flagging broadband between 20-50% of bins
- **Result: Even more files flagged (~750+)** — most recordings have columns in the 20-50% activity range; it's the "normal" noise level.

### Approach 3: Band energy ratio (USV band vs low band)
- Compare energy above 30 kHz vs below 30 kHz within detection windows
- **Result: Failed completely.** Real USV recordings also have high low-band energy from cage noise. The broadband burst puts energy across all frequencies including above 30 kHz, so both bands look active.

### Approach 4: Median band energy comparison
- Use median (not peak) energy per band to avoid the broadband burst inflating the USV band score
- **Result: Known problem files still showed high USV-band energy** (10-15 dB above median) because the broadband distributes energy everywhere.

### Approach 5: Tighter context windows and dB thresholds
- Reduced context around detections to 2 columns before/after
- **Result: Same high flag rate** — the cage noise is inside the detection windows themselves, not just around them.

### Profiling the problem files
We measured dB levels in detection windows. Normal detections (570 events from 100 random auto-accepts):

| Percentile | Low-band mean (dB above median) | Max broadband fraction |
|------------|--------------------------------|----------------------|
| P50 | -0.4 | 0.341 |
| P75 | +2.0 | 0.571 |
| P90 | +5.1 | 0.707 |
| P95 | +7.2 | 0.794 |

The problem files' metrics overlap heavily with normal detections. Simple spectral thresholds cannot separate this pattern from normal cage noise. The profiling was done on an incorrect list of problem files (see corrected list of 10 files above) — re-profiling with the correct files would be a good first step for fresh analysis.

## Why Simple Approaches Fail

The fundamental issue: the 1960 pattern's spectral signature (partial broadband + low-frequency energy) is **indistinguishable from normal cage noise** using aggregate frequency-band statistics. The difference is *structural/visual* — in the 1960 pattern, the detection contains ONLY the streak with no real USV, whereas in normal recordings there's actual narrow-band USV energy alongside the noise.

## The 10 Known 1960-Problem Files

These were identified during manual review of the 70 manual-review tier files. These specifically have the broadband burst + low-frequency streak pattern (NOT just general noise):

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

Note: Several other manual-review files are just broadband noise or correct detections — those are separate issues, not the 1960 pattern.

## What Pictures to Provide

When starting the new conversation, include these screenshots:

1. **A clear 1960 problem example** — open recording `0001960` or `0001930` in the PyQt6 app, screenshot the spectrogram showing the broadband burst + low-frequency streak inside the detection (cyan region). This is the pattern to solve.

2. **A normal detection for comparison** — open a known-good file like `0000053` or `0000054`, screenshot a detection with a real USV (you'll see a clear narrow-band trace above 30 kHz).

3. **An ambiguous case** — one of the weaker problem files like `0003503` or `0002765` where the pattern is less obvious, to show the difficulty of the boundary.

## Possible Paths Forward

1. **CNN retraining with hard negatives** — Add the 12 known problem files (plus more if found) as negative training examples. The CNN would learn to distinguish the low-freq streak from real USVs. Needs ~50-100 hard negative examples. This is likely the most durable solution but requires retraining effort.

2. **More sophisticated spectral analysis** — Instead of aggregate band statistics, look at the actual spatial pattern in the spectrogram (narrow-band vs broadband energy distribution within the detection). This is essentially what the CNN should be doing already.

3. **Accept the false positive rate** — 10/70 manual-review files had the pattern. If the same rate applies to auto-accept, that's ~190 affected files out of 1,344 (~14%). For some use cases this may be acceptable.

4. **There may be no simple post-processing fix** — All spectral threshold approaches failed because the pattern shares too many statistical properties with normal noisy recordings. The discrimination requires understanding the spatial structure of the spectrogram, which is fundamentally a classification task.

## Technical Details

- Spectrogram: n_fft=512, hop=128, sr=300,000 Hz, range 20-120 kHz
- Frequency resolution: ~586 Hz per bin
- Low band (20-30 kHz): rows 0-17 of spectrogram
- Detection JSONs: `results/batch_5970/detections/*.json` (contain start_col, end_col, probabilities)
- Summary: `results/batch_5970/summary_full.parquet`
- Script used for analysis: `scripts/flag_broadband_fp.py`
- PyQt6 app: `python -m usv_spectrogram.app.main` (uses model at `models/matched_windows/best_model.pt`)
