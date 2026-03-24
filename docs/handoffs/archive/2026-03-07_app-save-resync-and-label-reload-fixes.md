# Handoff: App Save Resync And Label Reload Fixes
Date: 2026-03-07

## Task

Continue the app-focused bug hunt, concentrating on the remaining save/load/export workflow risks:

- save current view after manual edits and boundary adjustments
- save all with mixed saved/unsaved/manual/adjusted detections
- load labels, then edit/delete/save again
- export metadata/index consistency after deletions and reordered detections

Delivered:

- preserved detection `save_state` across label JSON save/load round-trips
- made adjusted save-current replace stale accepted exports instead of leaving duplicate old PNG/JSON files behind
- turned save-all into a true accepted-export resync that rewrites current detections in current order while preserving deletion-history tracker records
- added focused workflow regressions for the adjusted re-save and save-all reindex flows

## Files Changed

- `src/usv_spectrogram/app/core/label_storage.py`
  Persisted and restored `save_state` in label JSON files.
- `src/usv_spectrogram/app/core/detection_exporter.py`
  Added accepted-export cleanup helpers and changed summary CSV generation to rebuild from the JSON metadata currently on disk.
- `src/usv_spectrogram/app/core/saved_detection_tracker.py`
  Added helpers to remove matching accepted records and to clear non-deleted records while preserving deletion history.
- `src/usv_spectrogram/app/main_window.py`
  Save-current now removes stale accepted exports before re-saving adjusted detections, delete removes prior accepted exports before rejected export, and save-all now clears/rebuilds accepted exports for the WAV in current order.
- `tests/test_label_storage.py`
  Extended label-storage round-trip coverage to include saved/manual/adjusted detection metadata.
- `tests/test_app_save_workflows.py`
  Added workflow regressions for adjusted save-current replacement and save-all reindexing after reorder/deletion.
- `docs/handoffs/current_bug_hunt.md`
  Updated the rolling handoff with the new save/load/export state and validation baseline.

## Reasoning

The remaining app bugs were no longer isolated helper issues; they were state-reconciliation issues across three persistence layers:

1. label JSON state
2. accepted export files (`detection_*.png/.json` + `detections_summary.csv`)
3. tracker history (`_saved_tracking.json`)

The first gap was label reload. Manual detections are intentionally tracker-independent, and adjusted detections are intentionally treated as needing a new save when their boundaries change. But label JSON did not preserve `save_state`, so reloading labels downgraded already-saved manual and adjusted detections back to `unsaved`. That made post-load editing and save-again workflows drift from what the user had already reviewed.

The second gap was stale accepted exports. Adjusting a saved detection and exporting it again produced a new PNG/JSON pair while the old accepted pair stayed on disk. The repo’s current untracked output examples already showed that failure pattern with duplicate `detection_003_*` files for different boundaries. I fixed that by removing matching prior accepted exports before re-saving adjusted detections.

The third gap was index drift after deletions/reordering. Accepted export filenames and JSON metadata include `detection_index`, but the old save-all path only exported unsaved detections. That left previously saved accepted outputs with stale indices after the current detection list changed. I fixed this by making save-all the explicit accepted-export resync path: clear existing accepted exports for the WAV, preserve deletion-history tracker entries, then export every current detection in current order.

I also changed the summary CSV to rebuild from the JSON files currently on disk instead of appending forever. Without that change, even correct file cleanup would still leave stale rows in `detections_summary.csv`.

## Validation

- `python -m py_compile src/usv_spectrogram/app/core/label_storage.py src/usv_spectrogram/app/core/saved_detection_tracker.py src/usv_spectrogram/app/core/detection_exporter.py src/usv_spectrogram/app/main_window.py tests/test_label_storage.py tests/test_app_save_workflows.py` : PASS
- `python -m pytest tests/test_label_storage.py tests/test_app_save_workflows.py tests/test_saved_detection_tracker.py tests/test_saved_detection_ghosts.py tests/test_app_selection_mapping.py -q` : PASS (`17 passed`)
- `python -m pytest tests -q` : PASS (`621 passed, 1 skipped`)

## Open Questions / Known Risks

No reproduced failure remains from this pass.

The main remaining risk is that these are still GUI-light workflow tests. They cover the state transitions and file outputs directly, but they do not yet drive the full Qt window lifecycle for open/load/save interactions.

## Worth Remembering For Claude

- `save_state` is now part of the label JSON contract. Without it, loaded manual detections and previously saved adjusted detections regress into false-unsaved state.
- Accepted export truth now comes from the set of `detection_*.json` files on disk; `detections_summary.csv` is derived from that set, not an append-only log.
- `Save All` should now be thought of as “resync accepted exports for this WAV,” not “export only unsaved detections.”
- Tracker semantics remain split:
  - accepted/manual saved detections can be cleared/resynced
  - `deleted_by_user` records must survive save-all resync so duplicate suppression and delete history are preserved
- Current validated bug-hunt baseline is now `621 passed, 1 skipped`
