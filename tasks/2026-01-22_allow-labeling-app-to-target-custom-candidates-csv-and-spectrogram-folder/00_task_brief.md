# Task Brief

Title: Allow labeling app to target custom candidates CSV and spectrogram folder
Date: 2026-01-22

## Goal
Let the labeling app load candidate IDs and spectrograms from user-selected paths so it matches regenerated spectrogram sets.

## Context
Assumptions:
- Current defaults are hardcoded to `candidates_optimized.csv` and `spectrograms_review/` in repo root.
- User regenerated spectrograms from a different candidates CSV, causing filename mismatches.
Uncertainties:
- Which exact CSV and spectrogram directory the user wants to target.

## Scope
In scope:
- Add a simple UI control to choose the candidates CSV and spectrograms directory.
- Use selected paths for loading candidates and images.
- Keep current defaults as prefilled values.
- Update quickstart with instructions for choosing custom paths.
Out of scope:
- Changing detection/extraction outputs or filenames.
- Changing candidate schema or ordering rules.

## Constraints
Dependencies:
- No new Python dependencies.
Performance:
- Minimal; path selection should be lightweight.
File ownership:
- Implementer edits labeling app and docs only.
API stability:
- No breaking changes to CLI entrypoints.
Style:
- Small, focused diff; add docstrings only for new/changed public functions.

## Acceptance criteria
- User can point the app to a non-default candidates CSV and spectrograms directory.
- App loads candidates and displays images that match candidate_id filenames from the selected CSV.
- Quickstart mentions how to switch to a custom CSV/dir.

## File touch list
New files:
- None.
Modified files:
- src/usv_spectrogram/labeling/labeling_app.py
- LABELING_TOOL_QUICKSTART.md

## Plan (small diffs)
1) Add sidebar inputs for candidates CSV and spectrograms directory; reload on change.
2) Update quickstart documentation.

## Implementer instructions
Do:
- Validate paths and show clear errors if missing.
- Keep defaults pointing to repo root files.
Do not:
- Do not add dependencies.
- Do not change candidate CSV format.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
