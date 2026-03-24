# Handoff: Ghost Overlay And Noise Tracker Persistence Fixes
Date: 2026-03-07

## Task

Continue the app-focused long bug hunt from the restart brief, concentrating on save/delete/ghost workflow behavior in the detection review UI.

Delivered:
- prevented deleted detections from reappearing as gray ghost overlays
- made noise labeling clear persisted saved-detection tracker state, not just in-memory state
- added focused regression coverage for both workflow bugs

## Files Changed

- `src/usv_spectrogram/app/core/saved_detection_ghosts.py`
  Added a testable helper that converts saved tracker records into read-only ghost detections while excluding deleted records and suppressing already-current saved detections.
- `src/usv_spectrogram/app/main_window.py`
  Switched ghost loading to the new helper and changed noise labeling to clear tracker records persistently.
- `src/usv_spectrogram/app/core/saved_detection_tracker.py`
  Added `clear_records()` so the app can clear and persist tracker state safely.
- `tests/test_saved_detection_ghosts.py`
  Added regression coverage for deleted records not becoming ghosts, current saved detections suppressing duplicate ghosts, and unmatched saved records still appearing as ghosts.
- `tests/test_saved_detection_tracker.py`
  Added regression coverage proving tracker clearing survives reload from disk.

## Reasoning

The first bug came from a mismatch between tracker semantics and overlay semantics.

`SavedDetectionTracker` intentionally stores both accepted saved detections and deleted detections so duplicate decisions can persist across sessions. But the ghost-overlay loader treated every tracker record as a historical accepted detection, which meant a detection deleted by the user could come back visually as a gray, read-only ghost in later review sessions. That undermines deletion as a workflow action.

I fixed that by extracting ghost construction into a small core helper and filtering `user_action="deleted_by_user"` records there. The helper also preserves the newer exact-boundary matching rule when deciding whether a saved record is already represented by a current detection.

The second bug was persistence-related. Labeling a file as noise cleared `saved_tracker.saved_detections` only in memory. The `_saved_tracking.json` file on disk was left intact, so reopening the file could restore old saved detections and ghost overlays. I added a dedicated `clear_records()` method that persists the cleared state and updated the noise-label path to use it.

## Validation

- `python -m py_compile src/usv_spectrogram/app/core/saved_detection_tracker.py src/usv_spectrogram/app/core/saved_detection_ghosts.py src/usv_spectrogram/app/main_window.py tests/test_saved_detection_tracker.py tests/test_saved_detection_ghosts.py` : PASS
- `python -m pytest tests/test_saved_detection_tracker.py tests/test_saved_detection_ghosts.py tests/test_app_selection_mapping.py -q` : PASS (`13 passed`)
- `python -m pytest tests -q` : PASS (`617 passed, 1 skipped`)

## Open Questions / Known Risks

No new known failures from this pass.

One remaining high-value area is still full workflow coverage around `MainWindow` methods themselves. The bug fixes are now backed by core-unit tests, but the app still lacks broader end-to-end GUI-light tests for save current view, save all, load labels, and post-edit reload behavior.

## Worth Remembering For Claude

- Tracker records are not all equivalent:
  - accepted saved detections may appear as gray `saved_previous` ghosts
  - deleted detections must stay in the tracker for duplicate suppression, but must not become ghost overlays
- Noise labeling now has persistence semantics: it clears `_saved_tracking.json` state for the current file through `SavedDetectionTracker.clear_records()`
- The current full-suite baseline after this pass is `617 passed, 1 skipped`
