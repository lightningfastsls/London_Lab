# Acoustic Properties Module Review

**Reviewed by:** Master Reviewer
**Date:** 2026-02-24
**Module:** `usv_language/analysis/acoustic_properties.py`
**Tier:** 1 (utility module, pure NumPy)
**Verdict:** APPROVED (all warnings fixed)

---

## Findings Summary

| Severity | Count | Items |
|----------|-------|-------|
| BLOCKER | 0 | -- |
| WARNING | 6 | W1-W6 (all fixed) |
| SUGGESTION | 5 | S1-S5 (S2, S3 fixed; S1, S4, S5 noted) |

## Warnings (all resolved)

- **W1**: Unsorted onsets caused silent wrong results in `time_since_last_usv` and batch scan
- **W2**: `direction_threshold` could be negative, causing all-rising classification
- **W3**: `bout_position` docstring claimed [0,1] range but code produces [0, (T-1)/T]
- **W4**: Non-2D `spec` input silently returned empty arrays instead of raising
- **W5**: Module doc missing
- **W6**: `IMPLEMENTATION_PROGRESS.md` not updated

## DSP Correctness

- **dB-to-linear** (`10^(S_db/10)`): Correct — gives power (|STFT|^2) from amplitude dB
- **Frequency axis**: Matches `codebook_viz.py` line 149-151
- **ADR-001/ADR-002**: `sample_rate=300000` and `hop_length=128` are explicit defaults

## Test Results (post-fix)

22 tests pass. Full suite: 254 passed, 1 skipped (HMM/hmmlearn).

## Fix Log

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| W1 | FIXED | acoustic_properties.py:242,289 | 2026-02-24 | Defensive sort when onsets unsorted; 2 new tests |
| W2 | FIXED | acoustic_properties.py:74 | 2026-02-24 | `direction_threshold >= 0` validation; 2 new tests |
| W3 | FIXED | acoustic_properties.py:216,285 | 2026-02-24 | Docstrings now say [0, (T-1)/T] |
| W4 | FIXED | acoustic_properties.py:291 | 2026-02-24 | ValueError for non-2D spec; 1 new test |
| W5 | FIXED | docs/modules/acoustic-properties.md | 2026-02-24 | Module doc created |
| W6 | FIXED | IMPLEMENTATION_PROGRESS.md | 2026-02-24 | Entry added |
| S1 | NOTED | docs/modules/acoustic-properties.md | 2026-02-24 | Documented n_freq_bins vs n_freq correspondence |
| S2 | FIXED | analysis/__init__.py | 2026-02-24 | extract_all_properties added to __all__ |
| S3 | FIXED | test_acoustic_properties.py:1 | 2026-02-24 | Docstring count updated to 21 (now 22 with W4 test) |
| S4 | DEFERRED | -- | -- | ROADMAP entry deferred to next planning session |
| S5 | NOTED | -- | -- | Complexity note is in the correct location (docstring of the function that does binary search) |
