# Handoff: App Selection Reload And Qt Event Pass
Date: 2026-03-07

## Task

Continue the app-focused bug hunt in the real Qt integration layer and keep chaining until the current pass is genuinely exhausted.

Delivered:

- fixed stale canvas selection surviving label reloads
- added real offscreen Qt coverage for canvas-click deletion
- added real offscreen Qt coverage for load-labels then adjust-and-save
- added real offscreen Qt coverage for save-current under real scroll state with visible ghosts present

## Files Changed

- `src/usv_spectrogram/app/widgets/spectrogram_view.py`
  Added `SpectrogramCanvas.clear_selection()` to clear selected detection, drag state, and cursor state explicitly.
- `src/usv_spectrogram/app/main_window.py`
  Clear spectrogram selection on WAV load reset and before applying loaded labels so a stale pre-load selection cannot target newly loaded detections.
- `tests/test_app_qt_integration.py`
  Added offscreen Qt integration tests for:
  - stale selection being cleared on label reload
  - deletion through an actual canvas click
  - load-labels then boundary-adjust then save-current through the event loop
  - save-current with ghosts present under real viewport/scroll state
- `tests/test_app_save_workflows.py`
  Updated the lightweight load-label test doubles to provide the new `spectrogram_view.canvas.clear_selection()` surface and `QMessageBox.critical()`.

## Reasoning

The concrete reproduced bug was stale selection carry-over across label reload.

Before this pass, if the user had selected a detection boundary, then loaded a labels JSON for the same WAV, the old `_selected_detection_idx` stayed live inside `SpectrogramCanvas`. Because the new detections were drawn into the same indexed canvas list, `Remove Detection` could immediately act on the first newly loaded detection without any fresh user selection.

That bug belongs at the widget lifecycle boundary, not in the selection-mapping helper. The safest fix was to give the canvas an explicit selection reset API and call it in the two places that replace the reviewed detection context wholesale:

1. new WAV load
2. label JSON load

I kept normal in-file refresh behavior unchanged so regular redraws, save-state updates, and ghost refreshes do not clear selection unnecessarily.

After fixing the bug, I used the same Qt-instantiated test file to cover the remaining high-value event flows the previous handoff had identified. Those flows now exercise actual canvas events, scroll geometry, and post-load widget behavior instead of only method-level state helpers.

## Validation

- `python -m py_compile src/usv_spectrogram/app/main_window.py src/usv_spectrogram/app/widgets/spectrogram_view.py tests/test_app_qt_integration.py tests/test_app_save_workflows.py` : PASS
- `python -m pytest tests/test_app_qt_integration.py -q` : PASS (`6 passed`)
- `python -m pytest tests/test_app_qt_integration.py tests/test_app_save_workflows.py tests/test_label_storage.py tests/test_saved_detection_tracker.py tests/test_saved_detection_ghosts.py tests/test_app_selection_mapping.py -q` : PASS (`28 passed`)
- `python -m pytest tests -q` : FAIL outside the app path during collection with the known environment-level Windows torch DLL initialization error (`WinError 1114` while loading `torch` / `c10.dll` in `tests/test_cnn_model.py`)

## Open Questions / Known Risks

No new concrete app integration bug is reproduced after this pass.

The remaining work in this layer would be proactive coverage expansion rather than bug-led fixing. The most plausible next additions, if desired, are full-window shortcut flows (`Delete`, `Escape`) and more canvas drag edge cases under heavy ghost overlap.

## Worth Remembering For Claude

- Canvas selection is now treated as session-local review state and is explicitly cleared when the reviewed detection context is replaced by WAV load or label load.
- The Qt integration layer now covers:
  - file-load reset behavior
  - viewport visibility filtering
  - canvas-click deletion
  - load-labels then adjust/save
  - save-current with ghosts present under real scroll state
- Current app-focused validation baseline for this pass is `28 passed`.
- The only wider-suite failure reproduced here is still the known non-app Windows torch DLL initialization problem during test collection.
