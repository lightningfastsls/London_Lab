# spectrogram_filter — USV Spectrogram Pre-Filtering

Stateless DSP utility that cleans a magnitude spectrogram before ridge
extraction (§17.3) or autoencoder encoding (§17.6).  Shared infrastructure
for SIS-benchmark modules 17.3, 17.5, 17.6.

- **Module:** `src/usv_spectrogram/features/spectrogram_filter.py`
- **Package export:** `usv_spectrogram.features.{FilterConfig, prefilter_spectrogram}`
- **Spec:** `ROADMAP_SIS_BENCHMARK.md` §17.2
- **Review tier:** 3 (critical — 3 downstream consumers)

## Purpose

Wild-mouse recordings carry substantial cage noise (scratching transients,
broadband hiss, amplitude-modulated bands, equipment hum below 25 kHz and
artefacts above 120 kHz).  Naive argmax ridge extraction on unfiltered
spectrograms latches onto silent-column noise, broadband transients, and
low-SNR onset/offset columns.  AMVOC autoencoder training on unfiltered
spectrograms wastes capacity reconstructing noise.

Three defenses are applied in order:
1. **3×3 median filter** — removes isolated pixel outliers (mic clicks,
   single-bin EMI pickup, DSP artefacts) while preserving real USV ridges
   which span 3–5 bins due to STFT window leakage (ADR-002, Hann window,
   n_fft=512).
2. **Local noise-floor mask** — per time-column, threshold above
   `multiplier × rolling_median(per_column_median)` over a
   `noise_floor_window_cols`-wide window.  Adapts to slowly varying
   broadband noise (cage reverb, ventilation).
3. **Frequency band mask** — restricts to [25 kHz, 120 kHz], eliminating
   hum and ultrasonic artefacts that are structurally not USVs.

## Public interface

```python
from usv_spectrogram.features import FilterConfig, prefilter_spectrogram

cfg = FilterConfig()  # all defaults match ROADMAP §17.2
cleaned, mask = prefilter_spectrogram(magnitude, freqs_hz, cfg)
```

### `FilterConfig`

Frozen dataclass with validated defaults:

| Field | Default | Units | Notes |
|---|---|---|---|
| `sample_rate` | 300_000 | Hz | Propagated to downstream callers |
| `noise_floor_multiplier` | 3.0 | — | Must be `> 1` |
| `noise_floor_window_cols` | 20 | columns | Rolling-median window |
| `median_filter_size` | 3 | pixels | Must be odd |
| `freq_min_hz` | 25_000 | Hz | Inclusive lower bound |
| `freq_max_hz` | 120_000 | Hz | Inclusive upper bound |

### `prefilter_spectrogram(magnitude, freqs_hz, cfg)`

**Parameters:**
- `magnitude: np.ndarray[(F, T)]` — linear-magnitude STFT
- `freqs_hz: np.ndarray[(F,)]` — frequency of each row
- `cfg: FilterConfig`

**Returns:** `(cleaned, mask)`
- `cleaned`: same shape/dtype as `magnitude`, masked cells are zero
- `mask`: same shape, dtype `bool`, True where pixel survived both
  amplitude and frequency criteria

## Algorithm

```
1.  filtered      = median_filter(magnitude, size=3, mode='reflect')
2.  col_median    = np.median(filtered, axis=0)           # shape (T,)
3.  noise_floor   = median_filter(col_median, size=W, mode='reflect')
4.  amplitude_mask = filtered > multiplier × noise_floor[None, :]
5.  freq_mask      = (freqs_hz ≥ freq_min) & (freqs_hz ≤ freq_max)
6.  mask           = amplitude_mask & freq_mask[:, None]
7.  cleaned        = filtered × mask
```

Uses `scipy.ndimage.median_filter` with `mode='reflect'` for both the 2-D
and 1-D passes, which handles edges gracefully and makes the
`n_time_cols < noise_floor_window_cols` case crash-free.

## Integration

**Consumers** (modules that import this):
- `§17.3` DP-based ridge tracker — feeds cleaned magnitude into Viterbi DP
- `§17.4` iMSA classifier — pipeline step 2 before ridge tracking
- `§17.5` Oren 80-D vectorization — pipeline step 2 before ridge tracking
- `§17.6` AMVOC autoencoder — cleans spectrograms before 64×160 resize

**Callees** (what this imports):
- `scipy.ndimage.median_filter` — 2-D and 1-D median filtering
- `numpy` — array ops

## Usage example

```python
import numpy as np
from scipy.signal import stft

from usv_spectrogram.features import FilterConfig, prefilter_spectrogram

sample_rate = 300_000
f, t, Zxx = stft(audio, fs=sample_rate, nperseg=512, noverlap=384)
magnitude = np.abs(Zxx)

cfg = FilterConfig()
cleaned, mask = prefilter_spectrogram(magnitude, f, cfg)

# cleaned is ready for ridge tracking or autoencoder input
# mask tells downstream consumers which cells are "real" signal
```

## Key decisions

### 3×3 median filter (not 1×3 time-only)

The ROADMAP spec mandates a 3×3 kernel.  Reinterpretation as time-only was
considered during implementation (see handoff) but rejected: real USV
ridges span 3–5 bins (STFT leakage), so the 3×3 filter preserves ridges
while suppressing both time-axis AND frequency-axis isolated outliers
(e.g. single-bin EMI pickup).  Time-only filtering would only suppress
time-axis glitches.

### Rolling median for per-column noise floor

The spec text "rolling median over `noise_floor_window_cols` centered on
that column" is interpreted as:
1. Collapse each column to a scalar via `np.median(filtered, axis=0)`.
2. Smooth this per-column series with a 1-D rolling median.

This yields one noise-floor value per column — `O(T)` vector broadcast
against `(F, T)` magnitude, which is cheap.  The per-column collapse uses
the median of all frequency bins, which is robust to a strong ridge
occupying a few bins: it stays near the background level even for
loud-tone columns.

### Output uses filtered magnitude, not original

The spec text contains a minor ambiguity: step 3 says "mask = magnitude
> …" and step 5 says "cleaned = magnitude * mask".  Taken literally, an
isolated outlier pixel (amplitude 1000 against background 1) passes the
threshold and propagates to `cleaned` unchanged — which defeats the
purpose of the 3×3 median filter.  The implementation uses
`filtered * mask` so the filter's outlier-suppression is preserved.
This matches the ROADMAP test 2 (`test_single_outlier_pixel_removed_by_median_filter`).

## Exit criteria (all met)

- [x] Filter reduces broadband noise on synthetic noisy tone by >10 dB SNR
      improvement (`test_snr_improves_by_10db_on_noisy_tone`)
- [x] Frequency bins outside [25, 120] kHz are zero after filtering
      (`test_below_freq_min_fully_masked_to_zero`, `test_above_freq_max_fully_masked_to_zero`)
- [x] All 35 tests pass (20 post-fixes + 15 test-hardener additions)
- [x] `py_compile` passes

## References

- ROADMAP: `ROADMAP_SIS_BENCHMARK.md` §17.2 (lines 116–206)
- ADR-002: STFT parameters (n_fft=512, hop=128, sr=300_000)
- Paper: Oren et al. 2024 — pre-filtering rationale for murine USV ridge tracking
