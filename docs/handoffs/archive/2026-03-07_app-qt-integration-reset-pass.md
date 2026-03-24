# Handoff: App Qt Integration Reset Pass
Date: 2026-03-07

## Task

Continue the app-focused bug hunt until the Qt-instantiated integration layer is exhausted.

Delivered:

- fixed stale review UI state leaking across WAV switches
- added real Qt-instantiated integration coverage for file-load reset behavior and viewport-based visible-detection filtering

## Files Changed

- `src/usv_spectrogram/app/main_window.py`
  Loading a new WAV now explicitly clears prior probability/detection visuals, disables review actions until detection or label load occurs, and resets the detection info label.
- `src/usv_spectrogram/app/widgets/probability_view.py`
  Added `clear()` methods so the probability canvas/view can be reset cleanly during file switches.
- `tests/test_app_qt_integration.py`
  Added offscreen Qt integration tests for:
  - `_load_wav_file()` clearing stale review UI state on file switch
  - `_get_visible_detections()` using real viewport/scroll geometry

## Reasoning

By this point the app bugs were no longer in the persistence model; they were in the actual widget lifecycle.

The key issue was that `_load_wav_file()` reset internal detection state but left the visible review UI partially stale:

- old probability data could remain drawn
- old detection overlays could remain attached to the new spectrogram view until detection reran
- save/remove/apply controls could stay enabled from the previous file even though the new file had not yet been reviewed

That is exactly the kind of bug that lightweight method tests tend to miss, so I moved to Qt-instantiated offscreen tests and fixed the reset path directly.

I also added a real viewport-based visibility test, because the remaining risk area around "save current view" is geometry-dependent behavior rather than pure data transformation.

## Validation

- `python -m py_compile src/usv_spectrogram/app/main_window.py src/usv_spectrogram/app/widgets/probability_view.py tests/test_app_qt_integration.py` : PASS
- `python -m pytest tests/test_app_qt_integration.py -q` : PASS (`2 passed`)
- `python -m pytest tests/test_app_qt_integration.py tests/test_app_save_workflows.py tests/test_label_storage.py tests/test_saved_detection_tracker.py tests/test_saved_detection_ghosts.py tests/test_app_selection_mapping.py -q` : PASS (`24 passed`)

Wider validation note:

- A full `tests -q` run is currently blocked intermittently by an environment-level Windows `torch` DLL initialization failure (`c10.dll` / WinError 1114) in non-app tests such as `tests/test_cnn_model.py` and `tests/test_training_cycle.py`.
- `tests/test_cnn_model.py -q` does pass when run directly in isolation in this environment, which points to environment/process instability rather than an app regression from this pass.

## Open Questions / Known Risks

I do not have another concrete app bug queued from this layer.

If app work continues, the next step would be more full-window interaction coverage, not more core bug-hunting:

- selection and deletion through actual canvas clicks
- save-current with ghosts present under real scroll/viewport state
- load-labels then save/delete through the real widget event loop

But at this point that would be proactive integration coverage rather than a bug-led pass.

## Worth Remembering For Claude

- The app bug-hunt now covers both state-model bugs and one level of real Qt-instantiated widget lifecycle behavior.
- New file loads now fully reset stale review UI state instead of only clearing backend objects.
- The app-focused integration set is green at `24 passed`.
- The remaining validation limitation is an intermittent environment-level `torch` DLL load failure outside the app path, not a reproduced app bug.
