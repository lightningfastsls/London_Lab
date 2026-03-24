# Acoustic Properties Handoff

**Module:** `usv_language/analysis/acoustic_properties.py`
**Date:** 2026-02-24
**Review Tier:** Tier 1 (utility module, no DSP-critical math)

## What Was Built

Pure NumPy module that extracts ground-truth acoustic properties from dB-scaled spectrogram columns. These serve as **probing targets** — labels for linear classifiers that test whether transformer hidden states encode specific acoustic features.

## Files Changed

| File | Action | Lines |
|------|--------|-------|
| `usv_language/analysis/acoustic_properties.py` | Created | ~270 |
| `usv_language/tests/test_acoustic_properties.py` | Created | ~190 |
| `usv_language/analysis/__init__.py` | Modified | +3 (import + docstring line) |

## Architecture

### Config
- `AcousticPropertyConfig` — frozen dataclass with `__post_init__` validation
- Defaults match ADR-001 (sr=300kHz), ADR-002 (hop=128), and existing `AnalysisConfig` freq range

### Public Functions (8)
| Function | Input | Output |
|----------|-------|--------|
| `peak_frequency(col, cfg)` | 1-D dB column | float Hz |
| `spectral_centroid(col, cfg)` | 1-D dB column | float Hz |
| `energy(col)` | 1-D dB column | float (linear power) |
| `is_voiced(col, cfg)` | 1-D dB column | bool |
| `frequency_direction(prev, curr, cfg)` | Two consecutive columns | `'rising'`/`'falling'`/`'flat'` |
| `bout_position(idx, length)` | Frame index + bout length | float [0,1] |
| `time_since_last_usv(idx, onsets, cfg)` | Frame index + onset array | float ms or -1.0 |
| `extract_all_properties(spec, cfg, onsets)` | Full spectrogram (n_freq, T) | dict[str, ndarray] |

### Key Design Decisions
- **dB→linear conversion:** `10^(S_db/10)` gives power (|STFT|²), correct for energy summation
- **Frequency axis:** `np.linspace(freq_min, freq_max, n_bins)` — matches `codebook_viz.py` line 150
- **Vectorization:** `extract_all_properties` vectorizes peak/centroid/energy/voiced/direction/position using NumPy ops; only `time_since_last_usv` uses an O(T+N) linear scan
- **Edge cases:** Empty spec → empty arrays; silence → center-freq centroid fallback; single-frame → 0.0 position; no preceding onset → -1.0

## Test Coverage

17 tests covering all 8 functions + config validation + batch extractor + empty-input edge case.
All 249 usv_language tests pass (1 skipped: HMM requires hmmlearn).

## Dependencies
- NumPy only (no torch/scipy/matplotlib)
- No new external dependencies added

## Risks
- None significant — pure computation, no I/O, no model weights
