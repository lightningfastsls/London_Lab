# Event Features: Column Mapping Decision

**Date:** 2026-03-28
**Status:** PENDING — awaiting decision
**Context:** Phase 15.4 (Event Feature Extraction) — implementation complete, tests pass, but master-reviewer flagged a design question

---

## The Problem

`extract_event_features()` needs to pull spectrogram columns for each event to compute spectral features (tonality, peak frequency, frequency range, continuity, SNR). The question: **which columns?**

### Background: How SlidingInference Works

The CNN slides a 100-column window across the spectrogram with stride `hop_px=10`:

```
Spectrogram:  col 0   col 10   col 20   col 30   col 40   col 50   col 60
               |        |        |        |        |        |        |
Window 0:     [========= 100 columns =========]
Window 1:              [========= 100 columns =========]
Window 2:                       [========= 100 columns =========]
```

Each window → one probability value. An event is a sequence of consecutive windows where probability stayed above threshold.

### What We Extract Now (Option A — Consecutive)

For event `start_window=2, window_count=4, hop_px=10`:

```
start_col = 2 × 10 = 20
end_col   = 20 + 4 = 24

Extracted columns: [20, 21, 22, 23]  ← 4 consecutive columns
Time covered: 4 × 0.427ms ≈ 1.7ms
```

These 4 columns are all physically adjacent — they're within what a single CNN window sees.

### What We Could Extract (Option B — Hop-Spaced)

```
columns = [20, 30, 40, 50]  ← one column per window position
Time sampled: columns span 30 × 0.427ms ≈ 12.8ms
```

These 4 columns are at the actual positions where each window was centered. They sample across the full event duration.

### What the Reviewer Suggested (Option C — Full Span)

```
start_col = 20
end_col   = 20 + 4 × 10 = 60

Extracted columns: [20, 21, 22, ..., 59]  ← 40 columns
Time covered: 40 × 0.427ms ≈ 17.1ms
```

All 40 columns in the event's time extent. But now we have 40 spectral snapshots for 4 probability values — a 10:1 mismatch.

---

## Why It Matters — Concrete Example

A USV that sweeps from 40 kHz → 80 kHz over 50ms (12 windows):

```
         col 0    col 10    col 20    col 30    ...    col 110
80 kHz                                                    ★
70 kHz                                          ★
60 kHz                                ★
50 kHz                      ★
40 kHz            ★

★ = peak frequency at each window's position
```

| Feature | Option A (consecutive) | Option B (hop-spaced) |
|---------|------------------------|----------------------|
| `freq_range_bins` | ~2 (sees barely any sweep) | ~68 (sees full sweep) |
| `freq_continuity` | ~0.2 (adjacent cols are similar) | ~6.2 (captures real jumps) |
| `tonality` | Accurate for 1.7ms slice | Averaged across full event |
| `snr_db` | Accurate for 1.7ms slice | Averaged across full event |
| `mean_peak_freq_bin` | ~40 kHz (only sees start) | ~60 kHz (sees center) |

**For a classifier trying to separate USVs from noise, Option B gives much more informative frequency features.** Tonality and SNR are similar either way (they're per-column averages), but the frequency features are fundamentally different.

---

## Trade-off Summary

| | Option A (current) | Option B (hop-spaced) | Option C (full span) |
|---|---|---|---|
| **Columns per event** | window_count | window_count | window_count × hop_px |
| **1:1 with probabilities** | ✅ | ✅ | ❌ (10:1) |
| **Captures full event span** | ❌ | ✅ | ✅ |
| **Code complexity** | Simple slice | Fancy indexing | Simple slice |
| **Test changes needed** | None | 1 test (edge-end) | 1 test + conceptual |
| **Frequency features** | Narrow-slice approximation | Correct per-window mapping | Oversampled |

---

## My Recommendation

**Option B (hop-spaced)** — best of both worlds:
- Maintains 1:1 correspondence between probability values and spectral columns
- Samples from across the full event, giving meaningful frequency features
- Small code change: `cols = np.arange(window_count) * hop_px + start_col`
- One test update: increase `n_time` in `test_event_at_end_of_spectrogram`

---

## What's Currently Shipped

The module is **implemented and all 17 tests pass** with Option A. It works and produces finite, discriminative features. The question is whether the frequency features are discriminative *enough* for the downstream FP classifier, or if they'd be significantly better with Option B.

## Also Noted (Non-Blocking)

Two naming inversions flagged by reviewer — can address alongside or separately:
- `prob_smoothness`: high value = jagged signal (name suggests opposite). Consider `prob_roughness`.
- `freq_continuity`: high value = jumpy frequency (name suggests opposite). Consider `freq_modulation_rate`.

---

## To Resume

Tell Claude: "Resume event features — I've decided on Option [A/B/C]" and optionally "also fix the naming" or "keep the names as-is".
