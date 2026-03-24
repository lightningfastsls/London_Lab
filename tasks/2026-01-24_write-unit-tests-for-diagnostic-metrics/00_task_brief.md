# Task Brief

Title: Write Unit Tests for Diagnostic Metrics
Date: 2026-01-24

## Goal
Create pytest tests for diagnostic metric functions to cover happy paths and error cases.

## Context
Assumptions:
- `compute_metrics_at_threshold` exists in `scripts/threshold_sweep.py`.
- `compute_spectrogram_stats` exists in `scripts/analyze_recording_performance.py`.
- Tests can import these functions directly without moving files.
Uncertainties:
- Current error-handling behavior of `compute_spectrogram_stats` (may return `None`, empty dict, or raise). Tests should reflect actual behavior once inspected.

## Scope
In scope:
- Create `tests/test_diagnostic_metrics.py` with pytest tests for both functions.
- Add minimal refactors (if necessary) to make functions importable without changing behavior (e.g., move helper functions above `if __name__ == "__main__":`).
Out of scope:
- Large refactors or moving code into new packages.
- Adding new dependencies beyond pytest.

## Constraints
Dependencies: Do not add new packages. Use pytest only if already available in the environment.
Performance: Tests should be fast and use small arrays/images.
File ownership: Prefer touching only `tests/test_diagnostic_metrics.py`; edit scripts only if importability requires it.
API stability: Do not change function signatures or behavior.
Style: Use `pytest.approx` for floats and `tmp_path` for temporary files.

## Acceptance criteria
- Test file `tests/test_diagnostic_metrics.py` exists and covers:
  - `compute_metrics_at_threshold`: perfect predictions, all wrong, all positive, all negative, mixed predictions, and threshold variation.
  - `compute_spectrogram_stats`: valid image, missing file, corrupt file, all-white image, all-black image.
- Tests assert the function's actual, documented behavior for error cases.
- Tests run via `.venv\Scripts\python.exe -m pytest tests/test_diagnostic_metrics.py -v` if `.venv` exists.

## File touch list
New files:
- `tests/test_diagnostic_metrics.py`
Modified files:
- `scripts/threshold_sweep.py` (only if importability requires it)
- `scripts/analyze_recording_performance.py` (only if importability requires it)

## Plan (small diffs)
1) Inspect the two target scripts to confirm function signatures and error behavior.
2) Write the pytest file with the required cases and fixtures.
3) If importability issues exist, apply the smallest safe refactor to expose functions.
4) Run the pytest command (use `.venv` python if available).

## Implementer instructions
Do:
- Keep tests independent and deterministic.
- Use tiny arrays and small images to keep runtime short.
Do not:
- Move functions into new modules without explicit approval.
- Change logic inside the functions under test.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run pytest command (using `.venv` python if present).
- Record commands and results in `20_verification.md`.
