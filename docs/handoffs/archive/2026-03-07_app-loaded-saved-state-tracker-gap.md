# Handoff: App Loaded Saved State Tracker Gap
Date: 2026-03-07

## Task

Continue the app-focused bug hunt after the load/reload guard pass, concentrating on post-load review behavior when labels are loaded from JSON without corresponding tracker history.

Delivered:

- fixed loaded detections with `save_state="saved_current"` so they are still treated as saved even when `_saved_tracking.json` is absent
- added regressions for both regular and adjusted loaded detections in that state

## Files Changed

- `src/usv_spectrogram/app/core/saved_detection_tracker.py`
  `is_saved()` now treats explicit `save_state="saved_current"` as authoritative saved state, even without tracker records.
- `tests/test_saved_detection_tracker.py`
  Added regressions for loaded saved-current detections, including adjusted detections, when tracker records are missing.

## Reasoning

This was a real follow-on bug from the earlier label JSON `save_state` preservation fix.

The app now saves and reloads `save_state`, but `SavedDetectionTracker.is_saved()` still relied on tracker records for most non-manual detections. That meant a labels JSON could correctly reload a detection as `save_state="saved_current"`, yet the app would still treat it as unsaved if `_saved_tracking.json` was absent. In practice that could trigger false unsaved prompts or duplicate save behavior after loading labels from JSON alone.

The safest fix was to treat `save_state="saved_current"` as authoritative at the tracker check boundary. This matches the meaning of the field, does not affect newly detected events (which remain `unsaved` unless tracker-backed), and still preserves the existing special handling for unsaved adjusted detections.

## Validation

- `python -m py_compile src/usv_spectrogram/app/core/saved_detection_tracker.py tests/test_saved_detection_tracker.py` : PASS
- `python -m pytest tests/test_saved_detection_tracker.py tests/test_app_save_workflows.py tests/test_label_storage.py tests/test_saved_detection_ghosts.py tests/test_app_selection_mapping.py -q` : PASS (`22 passed`)
- `python -m pytest tests -q` : PASS (`626 passed, 1 skipped`)

## Open Questions / Known Risks

No reproduced bug remains from this pass.

The remaining app risk is still mostly full GUI lifecycle coverage rather than state-model correctness. The next likely value is real open/load/save/scroll interactions in an instantiated Qt session, especially around partial viewport saves with ghosts.

## Worth Remembering For Claude

- Label JSON `save_state` and tracker history are no longer tightly coupled: loaded `saved_current` detections are considered saved even if tracker records are unavailable.
- This specifically closes the gap where reloaded label files could trigger false unsaved prompts or duplicate export behavior.
- Current validated app bug-hunt baseline after this pass is `626 passed, 1 skipped`
