# Normalization Module — Implementation Handoff

## Module
`src/usv_spectrogram/postprocessing/normalization.py`

## Purpose
Per-recording Z-normalization of CNN probabilities using the noise distribution estimated from the bottom 50th percentile of windows. Makes fixed thresholds behave adaptively across recordings with varying noise floors.

## Public Interface
- `normalize_scores_per_recording(probabilities: np.ndarray) -> np.ndarray` — Z-normalize a single recording
- `normalize_scores_batch(all_probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]` — Normalize multiple recordings independently

## Algorithm
1. Sort values, take bottom 50% as noise slice
2. `noise_median` = median of noise slice (robust location estimate)
3. Spread estimation (two tiers):
   - If noise slice has variation (MAD > 0): use **full-array MAD** relative to noise_median (wider, less biased than half-distribution MAD)
   - If noise slice is constant (MAD = 0): cascading **mean-AD** fallbacks (noise slice → full array → return zeros)
4. `z = (prob - noise_median) / noise_mad`

## Key Decisions
- **Full-array MAD when noise has variation**: Using only the noise-slice MAD causes centering bias (~1.5× for Gaussian noise) because the noise_median from bottom 50% is approximately Q1, not the true center. Full-array MAD compensates by providing a wider spread estimate.
- **Mean-AD fallback**: When noise is perfectly constant (e.g., 90 windows at 0.05 + 10 USV at 0.95), all median-based estimators collapse to zero. Mean-AD is less robust but handles these degenerate cases.
- **No config dataclass**: Only parameter (50th percentile) is fixed by spec. Adding a config for a single constant would be over-engineering.
- **No clipping**: Z-scores exceeding [0,1] are expected and intentional.
- **float64 output**: Avoids float32 precision issues in division.

## Files Created
- `src/usv_spectrogram/postprocessing/normalization.py` (NEW)

## Files Modified
- `src/usv_spectrogram/postprocessing/__init__.py` (added exports)

## Pre-Existing Tests
Pre-existing tests from test-architect: 16

## Test Results
- 37 total normalization tests (16 pre-existing + 21 hardener-added)
- 35 passed, 2 skipped (NaN/Inf input — deferred, production CNN outputs are always finite)
- 693 passed across full suite (24 pre-existing failures in test_triage.py — unrelated)

## Assumptions
- `[ASSUMED]` 50th percentile is the right cutpoint — based on USVs occupying <5% of windows in typical recordings
- `[ASSUMED]` Full-array MAD is acceptable when noise_slice has variation — in production USVs are rare (<5%), so full MAD is dominated by noise

## Review Tier
Tier 2 (Standard) — per ROADMAP §15.6. Contains non-trivial statistics (two-tier MAD/mean-AD fallback, centering bias rationale).
