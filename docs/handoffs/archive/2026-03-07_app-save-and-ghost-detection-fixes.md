# Handoff: App Save And Ghost Detection Fixes
Date: 2026-03-07

## Task

Continue the bug hunt beyond the classification bridge and carry the work through implementation and validation.

Delivered:
- tightened saved-detection deduplication so partial overlap no longer counts as “already saved”
- fixed selection mapping between displayed detections and editable current detections
- made previously saved ghost detections visible in the spectrogram, matching the existing UI text

## Files Changed

- `src/usv_spectrogram/app/core/saved_detection_tracker.py`
  Replaced overlap-based duplicate detection with boundary matching within a small tolerance.
- `src/usv_spectrogram/app/core/selection_mapping.py`
  Added a lightweight helper for mapping a canvas selection index back to a current editable detection.
- `src/usv_spectrogram/app/main_window.py`
  Updated ghost/current matching to use the new tracker semantics and fixed removal logic to resolve selection against the displayed list instead of indexing blindly into `detection_result.usvs`.
- `src/usv_spectrogram/app/widgets/spectrogram_view.py`
  Render ghost detections in gray instead of skipping them completely.
- `tests/test_saved_detection_tracker.py`
  Added tracker coverage for exact matches, small tolerated drift, partial overlap, deleted detections, adjusted detections, and manual detections.
- `tests/test_app_selection_mapping.py`
  Added focused coverage for display-selection mapping without importing the full Qt/torch stack.

## Reasoning

There were two connected app bugs around saved detections:

1. `SavedDetectionTracker.is_saved()` treated any overlap as a duplicate. That is too broad for reviewed detections because partially overlapping events can be genuinely distinct, and deleted detections were also broad enough to suppress new overlapping detections unintentionally.

2. The canvas selection index was taken from the displayed list (`current + ghosts`), while delete operations indexed into only the current detection list. That could target the wrong detection or drop the action entirely when ghost detections were involved.

I kept the duplicate-matching rule time-based, but changed it from overlap to boundary identity with a 1 ms tolerance. That preserves resilience against minor float drift without collapsing nearby but distinct events.

I moved the selection helper into `app/core/selection_mapping.py` so it can be tested in isolation. Importing `main_window.py` directly in tests pulled in the whole app stack, including `torch`, which is unnecessary for validating this mapping logic.

## Validation

- `python -m py_compile src/usv_spectrogram/app/core/saved_detection_tracker.py src/usv_spectrogram/app/core/selection_mapping.py src/usv_spectrogram/app/main_window.py src/usv_spectrogram/app/widgets/spectrogram_view.py tests/test_saved_detection_tracker.py tests/test_app_selection_mapping.py` : PASS
- `python -m pytest tests/test_saved_detection_tracker.py tests/test_app_selection_mapping.py -q` : PASS (`9 passed`)
- `python -m pytest tests -q` : PASS (`613 passed, 1 skipped`)

## Open Questions / Known Risks

- The duplicate tolerance is currently fixed at 1 ms in `saved_detection_tracker.py`. That seems appropriate for float jitter, but if the app later stores boundaries with coarser quantization, this tolerance may need to become configurable.
- There are still non-blocking warning-only items in the suite:
  - `render_tiles.py` tight-layout warning
  - `storage_zarr.py` Zarr deprecation warnings

## Worth Remembering For Claude

- The app now uses three separate but aligned concepts:
  - current detections: editable and savable
  - saved-current detections: current detections already matched to tracker records
  - saved-previous ghost detections: historical records shown in gray for context
- The key invariant is that tracker matching is now “same boundaries within tolerance,” not “any overlap.”
- The selection mapping helper was intentionally placed in `app/core/selection_mapping.py` to keep it testable without importing the full GUI/inference stack.
