# Task Brief

Title: Reset labeling progress for spectrogram review
Date: 2026-01-22

## Goal
Allow users to re-label all candidates when spectrograms have been regenerated, without getting stuck on "all labeled".

## Context
Assumptions:
- The labeling app currently loads `labels.csv` and treats all labeled candidates as complete.
- The user wants to start a fresh labeling pass after regenerating `spectrograms_review`.
Uncertainties:
- Whether the user wants to preserve previous labels or overwrite them (we will back them up before reset).

## Scope
In scope:
- Add a UI control to reset labeling progress (clear labels for a fresh pass).
- Back up the existing labels file before clearing so prior work is preserved.
- Update labeling quickstart to mention how to restart labeling.
Out of scope:
- Changes to detection or spectrogram extraction logic.
- Changes to candidate ordering or data schema.

## Constraints
Dependencies:
- No new Python dependencies.
Performance:
- No performance-sensitive changes.
File ownership:
- Implementer edits labeling app and docs only.
API stability:
- No breaking changes to public CLI interfaces.
Style:
- Small, focused diff; add docstrings only for new/changed public functions.

## Acceptance criteria
- Labeling app provides a clear way to restart labeling from scratch.
- Existing `labels.csv` is backed up before reset (timestamped or similar).
- After reset, navigation treats all candidates as unlabeled.
- `LABELING_TOOL_QUICKSTART.md` mentions the reset workflow.

## File touch list
New files:
- None expected (backup labels are runtime artifacts).
Modified files:
- src/usv_spectrogram/labeling/labeling_app.py
- LABELING_TOOL_QUICKSTART.md

## Plan (small diffs)
1) Add reset-labels helper + UI control in labeling app (sidebar or main).
2) Update labeling quickstart with new reset instructions.

## Implementer instructions
Do:
- Preserve old labels by renaming or copying before reset.
- Keep UI wording explicit ("Reset labeling / start fresh").
Do not:
- Do not change candidate CSV format or detection parameters.
- Do not add dependencies.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
