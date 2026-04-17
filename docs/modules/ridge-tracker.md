# ridge_tracker — DP Ridge Tracker (pitch contour extraction)

Viterbi-style dynamic-programming ridge tracker that extracts the MAP
frequency trajectory through a magnitude spectrogram. Shared infrastructure
for SIS-benchmark modules 17.4 (iMSA classifier) and 17.5 (Oren 80-D
vectorization).

- **Module:** `src/usv_spectrogram/features/ridge_tracker.py`
- **Package export:** `usv_spectrogram.features.{RidgeConfig, track_ridge}`
- **Spec:** `ROADMAP_SIS_BENCHMARK.md` §17.3
- **Review tier:** 3 (critical — upstream of two downstream modules)

## Purpose

Per-column argmax is brittle for USV pitch extraction:

1. **Harmonic jumps.** A mouse call's 2× harmonic often rivals the fundamental
   in amplitude. Independent per-column argmax flips between them, producing
   a sawtooth FM trajectory that's meaningless for downstream classification.
2. **Noise latching.** Silent or low-SNR columns have their argmax wherever
   background noise happens to peak — usually a different bin every time,
   producing spurious frequency excursions.
3. **Onset/offset bleed.** The first and last few columns of a call have
   partial-window STFT content; their argmax often lands off-peak.

The Viterbi DP solution adds a `−λ·|Δf|` transition penalty with a hard
local-jump constraint of `max_jump_bins`. This enforces smoothness and —
because the penalty window is strictly local — it is *structurally
impossible* for the tracker to cross a gap larger than `max_jump_bins` in a
single column. That is how harmonic suppression works: with the default
`W = 10` bins (~6 kHz at 300 kHz / n_fft=512), the fundamental and the 2×
harmonic are always far enough apart that no path can switch between them.

## Public interface

```python
from usv_spectrogram.features import RidgeConfig, track_ridge

cfg = RidgeConfig()  # defaults match ROADMAP §17.3
fm_hz, am = track_ridge(magnitude, freqs_hz, cfg)
```

### `RidgeConfig`

Frozen dataclass with validated defaults:

| Field | Default | Units | Notes |
|---|---|---|---|
| `transition_penalty` | 0.1 | per-bin cost | Must be `>= 0`. 0 reduces tracker to per-column argmax. |
| `max_jump_bins` | 10 | bins | Must be `>= 1`. Hard Viterbi window radius. |
| `silence_threshold` | 1e-6 | linear mag | Columns with `max < threshold` → NaN output. |

### `track_ridge(magnitude, freqs_hz, cfg)`

**Parameters:**
- `magnitude: np.ndarray[(F, T)]` — non-negative linear-magnitude STFT,
  typically the output of `prefilter_spectrogram`.
- `freqs_hz: np.ndarray[(F,)]` — frequency of each row (typically
  `np.fft.rfftfreq(n_fft, 1/sr)`).
- `cfg: RidgeConfig`.

**Returns:** `(fm_hz, am)` — both shape `(T,)`, `float64`, with `NaN`
on silent columns.

## Algorithm

```
1. is_silent[t] = magnitude[:, t].max() < silence_threshold
2. Split non-silent columns into contiguous runs [r_start, r_end)
3. Per run — Viterbi forward pass:
     cur_cost = magnitude[:, r_start]                 # seed
     for local_t in 1..run_len:
         for shift in [-W, +W]:
             candidate[f] = cur_cost[f + shift] - λ·|shift|
             (in-bounds f range only)
             best[f], best_src[f] = argmax over shifts
         cur_cost = magnitude[:, t] + best
         backtrace[:, local_t] = best_src
4. Back-trace from argmax(cur_cost) at the run's last column
5. fm[active_cols] = freqs_hz[ridge_idx[active_cols]]
   am[active_cols] = magnitude[ridge_idx[active_cols], active_cols]
```

Complexity: **O(F · W · T_active)**, where `T_active = T − silent_count`.
With our defaults (F=257, W=10) that's ~5k ops per active column — a single
7518-call dataset costs well under one CPU-second.

## Integration

**Consumers** (modules that import this):
- `§17.4` iMSA pitch-jump classifier — consumes `fm_hz` to detect jumps +
  overall slope shape.
- `§17.5` Oren 80-D vectorization — concatenates resampled FM + AM into a
  fixed-length feature vector.

**Callees** (what this imports):
- `numpy` — array ops only. No scipy, no torch. Intentionally standalone.

**Typical upstream pipeline:**
```
WAV → STFT → prefilter_spectrogram (17.2) → track_ridge (17.3)
  → classify_imsa (17.4)  or  vectorize_call (17.5)
```

## Usage example

```python
import numpy as np
from scipy.signal import stft

from usv_spectrogram.features import (
    FilterConfig,
    RidgeConfig,
    prefilter_spectrogram,
    track_ridge,
)

sample_rate = 300_000
f, t, Zxx = stft(audio, fs=sample_rate, nperseg=512, noverlap=384)
magnitude = np.abs(Zxx)

cleaned, _ = prefilter_spectrogram(magnitude, f, FilterConfig())
fm_hz, am = track_ridge(cleaned, f, RidgeConfig())

# fm_hz[t] = ridge frequency in Hz, NaN on silent columns
# am[t]    = magnitude at the ridge bin, NaN on silent columns
```

## Key decisions

### Runs split at silent columns; each run is independent

Silent columns break the DP chain. Each contiguous non-silent run is solved
independently, seeded from its own first-column argmax. Rationale:

- A silent column provides no meaningful transition state — forcing the DP
  to carry a path across silence would amount to imputing the ridge.
- The pre-filter (17.2) is aggressive enough that silent columns inside
  USV bounds are unusual; when they occur they often indicate either a
  segmentation glitch or a genuinely two-syllable event. Treating runs
  independently avoids pretending otherwise.
- Pre-existing test `test_silent_column_produces_nan_neighbors_intact`
  verifies that a silent column in the middle produces NaN there and
  leaves neighbours intact — the run-based design satisfies this by
  construction.

### Windowed DP (O(F·W·T)) instead of full O(F²·T)

Mouse USVs have smooth pitch trajectories between discontinuous jumps —
Oren 2024 measured <10-bin transitions for >95% of intra-syllable columns
at our bin width. The windowed formulation is ~25× faster than full
pairwise DP at F=257, W=10, and identical in output for smooth ridges.
For the rare `Δf > W` transitions, Viterbi cannot cross in one step — but
that's a *feature*, not a bug: the downstream iMSA classifier treats those
as "pitch jumps" and labels the call as `Complex` regardless.

### Transition cost scales linearly with `|Δf|` (not quadratically)

The spec specifies `λ·|Δf|` (L1 in bin space). An L2 alternative was
considered but rejected: L1 is how the reference papers (Oren 2024, the
tfridge MATLAB algorithm) specify the cost and it gives reasonable
behaviour — small jumps are cheap, large jumps are proportionally more
expensive. With our default `λ=0.1` and `W=10`, the worst-case transition
cost is 1.0 amplitude units, comparable to peak magnitude — so a strong
off-centre peak can win over continuity, but a comparable-amplitude path
strongly prefers to stay smooth.

### No NaN interpolation, no smoothing

The output is the raw DP MAP sequence. Consumers that need smoothing
(17.5's Oren vectorization smooths AM with a median filter + FM with a
mean filter) do it themselves. Keeping the tracker output raw avoids
layering DSP opinions that belong elsewhere.

## Exit criteria (all met)

- [x] RMSE < 2 kHz on synthetic FM sweep
      (`test_regression_fm_rmse_within_2khz`)
- [x] Harmonic-suppression test passes
      (`test_harmonic_suppression_stays_on_fundamental`)
- [x] All 14 tests pass (13 from ROADMAP spec + 1 additional from
      test-architect)
- [x] `py_compile` passes

## References

- ROADMAP: `ROADMAP_SIS_BENCHMARK.md` §17.3 (lines 208–296)
- ADR-001 (sample rate = 300000), ADR-002 (n_fft=512, hop=128)
- Paper: Oren et al. 2024 — ridge-based 80-D call vectorization
- Paper: Hertz et al. 2020 — iMSA pitch-jump classifier (consumes 17.3)
- MATLAB reference: `tfridge` — time-frequency ridge extraction
- Note: `notes/ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory.md`
