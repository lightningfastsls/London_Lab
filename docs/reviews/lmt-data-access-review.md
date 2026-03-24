# LMT Data Access Layer Module Review

**Date:** 2026-02-24
**Reviewer:** master-reviewer
**Review Tier:** 2 (new module with data access, timestamp math, alignment logic; no DSP/ML changes)
**Handoff:** `docs/reviews/lmt-data-access-handoff.md`

---

## Summary

The LMT Data Access Layer is a clean, well-structured module that achieves its core goals: read-only SQLite access, coordinate conversion between the three time domains, and behavioral event alignment. The coordinate math is correct, the tests pass, and the architecture (frozen dataclasses, context manager, SQL pre-filtering) is sound.

No blockers were found. There are two warnings that carry real integration risk (the key-naming mismatch will cause runtime errors when integrating with the rest of the pipeline), and three minor warnings around documentation and test completeness.

**Test results:** 21/21 passed. 330/330 passed (full suite, excluding pre-existing notion_notes import errors).

---

## DSP Correctness

Not applicable to this module. No FFT, no STFT, no dB scaling.

Coordinate math verified analytically:

| Conversion | Formula | Expected | Actual |
|------------|---------|----------|--------|
| Frame 30 at 30 fps | 30 / 30.0 | 1.0s | CORRECT |
| 1.0s at 300 kHz | round(1.0 * 300000) | 300000 | CORRECT |
| 1.0s to spec frame (hop=128) | int(1.0 * 300000 / 128) | 2343 | CORRECT |

ADR-001 (sample_rate=300000): `SyncConfig.wav_sample_rate = 300_000` — compliant.
ADR-002 (hop_length=128): `seconds_to_spectrogram_frame(time_s, hop_length=128)` default — compliant.

---

## ML Rigor

Not applicable. No ML, no training, no splits, no evaluation metrics.

---

## Spec Compliance

The implementation was built from a standalone plan. All required files are present and match the spec skeleton.

Schema deviations from the spec are improvements, not regressions:
- `BehavioralEvent.animal_id` spec says `int`, impl says `Optional[int]`. The impl is correct: environmental events (e.g., "night") have no animal, which the spec did not account for.
- `AnimalInfo.rfid` spec says `str`, impl says `Optional[str]`. Correct: minimal-schema databases may omit RFID.
- `AnimalInfo.name` not in spec but present in impl. Harmless addition.

---

## Integration Correctness

**Pattern 1 (Config Dataclass):** `SyncConfig` is frozen, has `__post_init__`, uses `_s` unit suffix. Fully compliant. `BehavioralEvent` and `AnimalInfo` are frozen but lack `__post_init__` — acceptable for data-holder dataclasses populated by the factory loader (see SUGGESTION S1).

**Pattern 2 (Candidate Data Flow):** Not directly used; LMT introduces a new data flow (SQLite -> BehavioralEvent) that feeds into the existing flow.

**Pattern 3 (Test Fixtures):** All tests use in-memory SQLite databases, never real files. Fixtures use `yield` with cleanup. Compliant.

**Read-only connection:** The `?mode=ro` URI approach is correct and verified. In-memory bypass for testing is appropriate and well-documented.

**Variable schema handling:** `cursor.description` column-name approach correctly handles 3-9 column ANIMAL tables across LMT versions.

---

## Findings

### WARNING W1: SQL end_frame off-by-one at upper time boundary

**What:** The `time_range` filter computes `end_frame = int(end_s * fps) + 1`. This causes the SQL query to include events whose STARTFRAME equals exactly `int(end_s * fps)` — i.e., events that start precisely at the upper boundary of the query range.

**Where:** `src/usv_spectrogram/lmt/db_loader.py`, line ~144.

**Why it matters:** The `get_events` docstring says the filter returns events "overlapping this time window." The Python overlap condition used in `align_events_with_detections` is strict: `ev.start_time_s < det_end` (strict less-than). These two are inconsistent.

**Fix:** Replace with `math.ceil`:
```python
import math
end_frame = math.ceil(end_s * self._frame_rate)
```

### WARNING W2: align_events uses `start_time`/`end_time` keys — inconsistent with project convention

**What:** `LMTSynchronizer.align_events_with_detections` reads `det["start_time"]` and `det["end_time"]` from detection dicts.

**Why it matters:** The rest of the project uses `_s` suffixes: `DetectedUSV.start_time_s`, JSON keys `"start_time_s"`, `BehavioralEvent.start_time_s`. This will cause `KeyError` when integrating with real pipeline dicts.

**Fix:** Change to `det["start_time_s"]`/`det["end_time_s"]` and update all test fixtures.

### WARNING W3: IMPLEMENTATION_PROGRESS.md not updated

**Fix:** Add LMT entry to `IMPLEMENTATION_PROGRESS.md`.

### WARNING W4: Missing integration test stub for real SQLite file

**Fix:** Add skip-guarded integration test.

### WARNING W5: No test for time_range upper-boundary exclusion

**Fix:** Add boundary test case.

### SUGGESTION S1: BehavioralEvent lacks start/end ordering validation

**Fix (deferred):** Add `__post_init__` with `end_frame >= start_frame` check.

### SUGGESTION S2: Post-close queries produce cryptic AttributeError

**Fix (deferred):** Add `_require_open()` guard method.

---

## Verdict

**CHANGES NEEDED** — W1-W5 before integration. S1-S2 deferred.

## Fixes Applied

All 5 warnings resolved. Suggestions S1-S2 deferred (non-blocking).

### W1: SQL off-by-one → math.ceil
- **File:** `src/usv_spectrogram/lmt/db_loader.py`
- **Change:** Added `import math`, replaced `int(end_s * fps) + 1` with `math.ceil(end_s * fps)`
- **Why:** `math.ceil` correctly excludes events starting at exactly `end_s`, matching the strict overlap semantics used in `align_events_with_detections`

### W2: Key naming → _s suffix
- **File:** `src/usv_spectrogram/lmt/synchronizer.py`
- **Change:** `det["start_time"]`/`det["end_time"]` → `det["start_time_s"]`/`det["end_time_s"]`, updated docstring
- **File:** `tests/test_lmt.py` — updated all 4 test detection dicts to use `start_time_s`/`end_time_s`
- **Why:** Matches project-wide `_s` suffix convention (DetectedUSV, label_storage JSON, BehavioralEvent)

### W3: IMPLEMENTATION_PROGRESS.md updated
- **File:** `IMPLEMENTATION_PROGRESS.md`
- **Change:** Added LMT Data Access Layer entry at top of Current Status section

### W4: Integration test stub added
- **File:** `tests/test_lmt.py`
- **Change:** Added `test_loader_opens_real_sqlite` with `@pytest.mark.skipif(not LMT_TEST_DB_PATH, ...)`
- **Verification:** Test correctly skips when env var not set (1 skipped in test output)

### W5: Boundary exclusion test added
- **File:** `tests/test_lmt.py`
- **Change:** Added `test_get_events_time_range_boundary_exclusion` — queries with end_s=1.5, verifies "Oral-oral Contact" (starting at frame 45 = 1.5s) is excluded
- **Verification:** Test passes with the math.ceil fix (W1)

### Post-fix verification
- `py_compile`: All files compile cleanly
- `pytest tests/test_lmt.py -v`: 22 passed, 1 skipped
- Full suite: 331 passed (was 330 before, +1 from new boundary test; integration test skipped)
