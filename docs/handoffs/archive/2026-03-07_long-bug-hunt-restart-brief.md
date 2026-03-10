# Handoff: Long Bug Hunt Restart Brief
Date: 2026-03-07

## Task

Give a fresh Codex chat enough context to continue a long bug hunt in `mickey_london_lab` without relying on prior conversation history.

This handoff is meant to be the first file a new Codex session reads after `AGENTS.md`.

## Workflow To Follow

Start-of-session read order for Codex in this repo:

1. `AGENTS.md`
2. `ops/goals.md`
3. `ops/reminders.md`
4. `docs/codex_index.md`
5. this file
6. `docs/handoffs/current_bug_hunt.md`
7. the newest dated handoffs in `docs/handoffs/`

Repo-specific rules that matter during bug hunts:

- Default writable areas:
  - `src/`
  - `tests/`
  - `scripts/`
  - `usv_language/`
  - `docs/handoffs/`
- Default read-only areas unless explicitly requested otherwise:
  - `.claude/`
  - `ops/`
  - `notes/`
  - `methodology/`
  - `reference/`
  - `templates/`
  - `inbox/`
- If touching DSP, STFT, WAV loading, or detection thresholds, read the relevant docs first and always specify `sr=300000` explicitly where applicable.
- Never change tests just to force a pass.
- Validate before claiming completion:
  - `python -m py_compile` on every changed Python file
  - targeted `pytest`
  - widen to broader `pytest` if practical
- Use `docs/handoffs/` for durable context.

## Environment Notes

- Windows / PowerShell environment.
- `rg` now works and resolves to the user-installed ripgrep rather than the inaccessible bundled app copy.
- `.venv` now includes `notion-client`, which had previously blocked collection of some tests.
- Current useful commands:

```powershell
.\.venv\Scripts\python.exe -m py_compile <file.py>
.\.venv\Scripts\python.exe -m pytest tests -q
rg -n "pattern" src tests usv_language scripts
```

## What Was Fixed In This Bug Hunt

### 1. DeepSqueak import/export round-trip mismatch

Problem:
- Raven export supported prefix-matched detection directory names.
- DeepSqueak import required exact directory-name equality.
- Result: valid round-trip data could appear as paired `unmatched_ds` and `unmatched_det`.

Files:
- `src/usv_spectrogram/classification/deepsqueak_import.py`
- `tests/test_classification/test_deepsqueak_import.py`
- Handoff: `docs/handoffs/2026-03-07_deepsqueak-import-prefix-match-fix.md`

Status:
- Fixed and regression-tested.

### 2. Saved-detection deduplication too broad

Problem:
- `SavedDetectionTracker.is_saved()` used overlap semantics.
- Partial overlap could incorrectly suppress distinct detections.

Files:
- `src/usv_spectrogram/app/core/saved_detection_tracker.py`
- `tests/test_saved_detection_tracker.py`

Status:
- Fixed.
- Matching now requires both boundaries to match within 1 ms tolerance.

### 3. Displayed-selection vs editable-selection mismatch

Problem:
- The canvas selection index came from `current + ghost detections`.
- Delete/remove logic indexed directly into `detection_result.usvs`.
- This could target the wrong detection or fail when ghosts were involved.

Files:
- `src/usv_spectrogram/app/core/selection_mapping.py`
- `src/usv_spectrogram/app/main_window.py`
- `tests/test_app_selection_mapping.py`

Status:
- Fixed and tested.

### 4. Ghost detections hidden despite UI implying they were visible

Problem:
- UI text described previously saved detections as visible in gray.
- Spectrogram canvas skipped drawing them entirely.

Files:
- `src/usv_spectrogram/app/widgets/spectrogram_view.py`
- `src/usv_spectrogram/app/main_window.py`

Status:
- Fixed.
- Ghost detections now render in gray.

### 5. Warning cleanup

Problem:
- Full suite was green but emitted warnings from:
  - `storage_zarr.py`
  - `render_tiles.py`
  - `repertoire_stats.py`

Files:
- `src/usv_spectrogram/storage_zarr.py`
- `src/usv_spectrogram/render_tiles.py`
- `src/usv_spectrogram/classification/repertoire_stats.py`
- Handoff: `docs/handoffs/2026-03-07_warning-cleanup-and-plot-guard.md`

Status:
- Fixed.
- Full `tests/` now runs cleanly with no warnings in this environment.

## Current Validation State

Latest known suite result:

- `python -m pytest tests -q`
- Result: `613 passed, 1 skipped`

This is the current baseline. If a new bug-hunt change regresses this, treat it seriously.

## Most Useful Existing Handoffs

Read these if the work overlaps those areas:

- `docs/handoffs/current_bug_hunt.md`
- `docs/handoffs/2026-03-07_deepsqueak-import-prefix-match-fix.md`
- `docs/handoffs/2026-03-07_app-save-and-ghost-detection-fixes.md`
- `docs/handoffs/2026-03-07_warning-cleanup-and-plot-guard.md`

## Best Next Targets For Another Long Bug Hunt

### Option 1: App workflow regression hunt

Highest-value next target.

Focus:
- `src/usv_spectrogram/app/main_window.py`
- `src/usv_spectrogram/app/core/label_storage.py`
- `src/usv_spectrogram/app/core/detection_exporter.py`
- `src/usv_spectrogram/app/widgets/`

Why:
- Multiple recent fixes touched app review state, save state, ghost state, and deletion flow.
- Remaining risk is now in user workflow integration, not isolated helper logic.

Suggested attack:
- Add direct workflow-style tests around:
  - save current view
  - save all detections
  - reload labels after manual edits
  - delete current detection when ghosts are present
  - ensure ghost detections stay read-only

### Option 2: Export/import pipeline invariants

Focus:
- `src/usv_spectrogram/app/core/detection_exporter.py`
- `src/usv_spectrogram/classification/raven_export.py`
- `src/usv_spectrogram/classification/deepsqueak_import.py`

Why:
- This is a cross-module round-trip pipeline.
- Silent format drift here is expensive.

Suggested attack:
- Build end-to-end tests covering:
  - export JSON -> Raven table -> synthetic DS results -> merged output
  - user-adjusted detections
  - deleted detections
  - manual detections
  - prefix-matched folder names

### Option 3: `usv_language` edge-case bug hunt

Focus:
- `usv_language/training/train_transformer.py`
- `usv_language/training/train_vqvae.py`
- `usv_language/training/extract_hidden_states.py`
- related tests in `usv_language/tests/`

Why:
- Higher complexity and likely richer edge cases.
- Good target if the app path feels stable enough.

## Practical Instructions For The Next Codex

- Do not waste time re-finding already fixed issues above.
- Use `current_bug_hunt.md` as the rolling summary and write a new dated handoff for any non-trivial new pass.
- After writing a handoff, explicitly tell the user where it is.
- The user wants long bug-hunt sessions, not single-bug stop-and-report behavior.
- Prefer carrying a pass through:
  - identify area
  - reproduce or reason concretely
  - implement
  - validate
  - update handoffs

## Worth Remembering For Claude

- The repo is currently in a strong state: full `tests/` clean, no warnings in this environment.
- The next valuable work is likely user-workflow robustness, not basic warning cleanup.
- If Claude is reviewing this Codex work, the rolling handoff is:
  - `docs/handoffs/current_bug_hunt.md`
