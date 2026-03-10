# Handoff: App Load Threshold And Noise Reload Guards
Date: 2026-03-07

## Task

Continue the app-focused bug hunt after the save/export resync pass, concentrating on integrated load/reload behavior:

- load labels, then edit/delete/save again
- reload behavior after noise labeling and clearing noise
- prevent silent state corruption when loading the wrong labels file for the current WAV

Delivered:

- restored saved thresholds when loading labels
- removed stale auto-saved noise JSON when clearing a noise label
- rejected label files that were saved for a different WAV
- added focused regressions for all three flows

## Files Changed

- `src/usv_spectrogram/app/main_window.py`
  `Load Labels` now restores `high_threshold` and `low_threshold` from the label JSON, aborts on mismatched source WAV names, and clearing a noise label now removes stale persisted noise JSON.
- `tests/test_app_save_workflows.py`
  Added workflow regressions for threshold restoration on load, stale noise JSON cleanup when clearing noise, and mismatched-WAV label rejection.
- `docs/handoffs/current_bug_hunt.md`
  Updated the rolling handoff with this pass and the new validation baseline.

## Reasoning

There were three remaining integration failures clustered around label reload:

1. Label JSON already stored `detection_params`, but `MainWindow._load_labels()` ignored them. That meant a user could load reviewed labels, then adjust/delete/save again under whatever thresholds happened to be in the current UI session rather than the thresholds that produced the loaded review state.

2. Noise labeling auto-saved a dedicated JSON file under `noise_labeled_files/`, but clearing the noise label only changed in-memory state. The stale noise JSON stayed on disk, which meant later reload behavior could still reflect an obsolete `file_label="noise"` snapshot for the same WAV.

3. `Load Labels` accepted any JSON file without checking whether it belonged to the currently loaded WAV. That made it possible to apply detections from the wrong recording onto the current spectrogram and then continue editing/saving from a corrupted state.

The fixes are deliberately conservative:

- threshold restoration only touches the saved high/low thresholds already present in the label JSON
- mismatch validation checks the saved `metadata.wav_file` basename against the currently loaded WAV basename, allowing path moves while still blocking clear cross-file mistakes
- stale noise JSON cleanup is best-effort and warns if cleanup fails instead of breaking the clear-noise operation

## Validation

- `python -m py_compile src/usv_spectrogram/app/main_window.py tests/test_app_save_workflows.py` : PASS
- `python -m pytest tests/test_app_save_workflows.py tests/test_label_storage.py tests/test_saved_detection_tracker.py tests/test_saved_detection_ghosts.py tests/test_app_selection_mapping.py -q` : PASS (`20 passed`)
- `python -m pytest tests -q` : PASS (`624 passed, 1 skipped`)

## Open Questions / Known Risks

No reproduced bug remains from this pass.

The remaining high-value app risk is still end-to-end Qt-driven lifecycle coverage rather than core state logic. The current tests exercise the workflow methods directly with lightweight stubs, not a fully instantiated GUI session.

## Worth Remembering For Claude

- Loaded labels now restore saved thresholds before the user continues editing or saving.
- Clearing a noise label removes the stale `noise_labeled_files/<wav>.json` snapshot so reload behavior cannot reintroduce obsolete noise state for that WAV.
- `Load Labels` is now guarded against cross-WAV mistakes by checking the saved source WAV basename.
- Current validated app bug-hunt baseline is `624 passed, 1 skipped`
