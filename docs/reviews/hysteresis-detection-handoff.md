# Handoff: Hysteresis Detection Post-Processing Module

**Date:** 2026-03-27
**Status:** Implementation complete, tests passing

## What Was Done

Created `src/usv_spectrogram/postprocessing/` package with hysteresis-based post-processing for converting CNN probabilities into discrete USV events.

### Files Created
- `src/usv_spectrogram/postprocessing/__init__.py` — Package init, public exports
- `src/usv_spectrogram/postprocessing/hysteresis.py` — Core logic (~160 lines)
- `tests/test_hysteresis.py` — 18 tests (~250 lines)
- `docs/modules/hysteresis-detection.md` — Module documentation

### Key Design Decisions

1. **Bidirectional extension** — Unlike the app's forward-only `HysteresisDetector`, this module extends backward from seed windows to capture the rising edge of USVs. This prevents clipping the start of vocalizations where probability was above sustain but hadn't yet reached onset.

2. **Window-index space** — Works on abstract window indices rather than column indices. The `convert_to_detection_format` function handles the mapping to column space when needed for LabelStorage compatibility.

3. **Frozen config** — `HysteresisConfig` is a frozen dataclass with `__post_init__` validation, preventing accidental mutation during batch processing.

## Verification

- `py_compile` passes on both source files
- 18/18 tests pass in `tests/test_hysteresis.py`
- Full test suite: 449 passed (5 pre-existing failures from missing `openpyxl`, unrelated)

## Test Coverage

| # | Scenario | Status |
|---|----------|--------|
| 1 | Single sustained peak → 1 event | PASS |
| 2 | Two peaks, large gap → 2 events | PASS |
| 3 | Two peaks, small gap → merged to 1 | PASS |
| 4 | Short spike below min_duration → filtered | PASS |
| 5 | Peak with sustain-level shoulders → extends | PASS |
| 6 | All-noise → empty list | PASS |
| 7a | Peak at array start → bounded | PASS |
| 7b | Peak at array end → bounded | PASS |
| 8a | Config: sustain > onset → ValueError | PASS |
| 8b | Config: negative gap → ValueError | PASS |
| 8c | Config: zero min_duration → ValueError | PASS |
| 9 | Times array → correct timestamps | PASS |
| 10 | ADR-010 format conversion → valid dicts | PASS |
| 11 | Empty input → empty list | PASS |
| 12 | Length mismatch → ValueError | PASS |
| 13 | Overlapping seed extensions → single event | PASS |
| 14 | Short column_indices → IndexError | PASS |
| 15 | Probabilities stored as copy | PASS |

## Next Steps

This module is ready for integration into the batch classification pipeline (`scripts/batch_classify_candidates.py`). The `InferenceResult` from `SlidingInference` feeds directly into `hysteresis_detect`.
