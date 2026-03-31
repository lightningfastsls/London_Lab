# Hysteresis Detection (Post-Processing)

**Module:** `src/usv_spectrogram/postprocessing/hysteresis.py`
**Package:** `src/usv_spectrogram/postprocessing/`
**Tests:** `tests/test_hysteresis.py` (21 tests)

## Purpose

Standalone batch-processing module that converts CNN per-window probabilities into discrete USV events using dual-threshold (onset/sustain) hysteresis with bidirectional extension.

## Relationship to App Detector

The desktop app has its own `HysteresisDetector` in `app/core/detection_logic.py`. This module differs:

| Feature | App Detector | Postprocessing Module |
|---------|-------------|----------------------|
| Extension direction | Forward-only scan | Bidirectional (forward + backward) |
| Index space | Column indices | Abstract window indices |
| Output type | `DetectedUSV` | `USVEvent` |
| Use case | Interactive app | Batch pipeline |

The bidirectional extension captures the full rising edge of USVs where probability was above the sustain threshold but hadn't yet reached onset.

## API

### `HysteresisConfig` (frozen dataclass)

| Field | Default | Description |
|-------|---------|-------------|
| `onset_threshold` | 0.75 | Probability to seed a new event |
| `sustain_threshold` | 0.40 | Probability to extend an event |
| `gap_fill_windows` | 3 | Merge events <= this many windows apart |
| `min_duration_windows` | 5 | Drop events shorter than this |

Validates: `0 < sustain <= onset <= 1.0`, `gap >= 0`, `min_dur >= 1`.

### `hysteresis_detect(probabilities, times, config) → List[USVEvent]`

Main entry point. Takes 1-D probability and time arrays from `InferenceResult`, returns list of `USVEvent`.

### `USVEvent` (frozen dataclass)

Fields: `start_window`, `end_window` (inclusive), `start_time_s`, `end_time_s`, `duration_ms` (center-to-center; 0 for single-window events), `peak_probability`, `mean_probability`, `window_count`, `probabilities`.

### `convert_to_detection_format(events, column_indices) → List[Dict]`

Converts events to ADR-010 / LabelStorage compatible dicts with `start_col`/`end_col` mapping.

## Algorithm

1. **Seed** — find windows where `prob >= onset_threshold`
2. **Extend** — from each seed, grow bidirectionally while `prob >= sustain_threshold`
3. **Extract** — find contiguous marked regions using `np.diff`
4. **Gap-fill** — merge regions separated by `<= gap_fill_windows`
5. **Min-duration filter** — drop short events
6. **Build** — compute times and stats for each surviving region
