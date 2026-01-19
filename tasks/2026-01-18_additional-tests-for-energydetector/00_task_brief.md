# Task Brief: Additional Tests for EnergyDetector

## Goal
Add missing unit tests for `EnergyDetector` behaviors (batch detection, CSV export, utility functions, and internal helpers) in `tests/test_energy_detector.py` without modifying production code.

## Scope
- Add new pytest test classes/methods covering:
  - `detect_batch` across multiple WAV files, error isolation, iterator yields
  - `save_candidates_csv` headers, empty list handling, parent dir creation
  - `analyze_threshold_sensitivity` threshold->count mapping
  - `verify_detection_coverage` coverage calculation with known timestamps
  - Internal helpers: `_compute_band_spectrogram`, `_frames_to_segments`, `_merge_segments`, `_filter_by_duration`
- Use existing fixtures from `tests/conftest.py` (`create_tone_wav`, `create_multi_tone_wav`).
- Keep diffs small; follow existing test class structure (e.g., `TestBatchDetection`, `TestCSVExport`).

## Non-scope
- No changes to production code under `src/`.
- No new dependencies or external data downloads.
- No refactors or test framework changes.

## Constraints
- Tests only in `tests/test_energy_detector.py`.
- Must align with existing pytest patterns and naming conventions.
- Use existing fixtures; avoid new helper modules.
- Assume Windows paths and PowerShell for any local commands.

## Acceptance Criteria
- [ ] New tests pass with `\.\.venv\Scripts\python.exe -m pytest tests/test_energy_detector.py -v`.
- [ ] Tests cover `detect_batch`, `save_candidates_csv`, `analyze_threshold_sensitivity`, `verify_detection_coverage`.
- [ ] Tests cover `_compute_band_spectrogram`, `_frames_to_segments`, `_merge_segments`, `_filter_by_duration`.
- [ ] No production code changes.
- [ ] Tests use existing fixtures from `tests/conftest.py`.
- [ ] New test class names follow `TestXxxYyy` pattern.

## File Touch List
- `tests/test_energy_detector.py`
- `tasks/2026-01-18_additional-tests-for-energydetector/10_impl_notes.md`
- `tasks/2026-01-18_additional-tests-for-energydetector/20_verification.md`

## Assumptions
- `EnergyDetector` exposes `detect_batch`, `save_candidates_csv`, `analyze_threshold_sensitivity`, and `verify_detection_coverage` as referenced in existing tests or module exports.
- Internal helper methods are accessible on the class for direct testing (as in existing tests).
- Existing fixtures provide temporary file paths that can be used in batch directory tests.

## Plan (Small Diffs)
- Stage 1: Review existing tests to mirror structure, locate related helpers, and identify best insertion points.
- Stage 2: Add tests for `detect_batch` and `save_candidates_csv`.
- Stage 3: Add tests for utility functions (`analyze_threshold_sensitivity`, `verify_detection_coverage`).
- Stage 4: Add tests for internal helpers (`_compute_band_spectrogram`, `_frames_to_segments`, `_merge_segments`, `_filter_by_duration`).
- Stage 5: Run targeted pytest for `tests/test_energy_detector.py` and record results.

## Open Questions
- None.
