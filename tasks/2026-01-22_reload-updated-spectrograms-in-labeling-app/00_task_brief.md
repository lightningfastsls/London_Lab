# Task Brief

Title: Reload updated spectrograms in labeling app
Date: 2026-01-22

## Goal
Ensure the labeling app shows newly regenerated spectrograms without stale image caching, and provide a simple refresh control.

## Context
Assumptions:
- Spectrogram PNGs are regenerated in-place under `spectrograms_review/` with the same filenames.
- The labeling app currently loads images by path, which can appear stale to the browser/Streamlit.
Uncertainties:
- Whether the user replaces `spectrograms_review/` while the app is running or between sessions.

## Scope
In scope:
- Load spectrograms by bytes to avoid stale image caching for updated PNGs.
- Add a small UI control to force a reload of data/session state.
- Update the labeling quickstart with the new refresh behavior.
Out of scope:
- Changes to detection or extraction logic.
- Changes to candidates CSV format.

## Constraints
Dependencies:
- No new Python dependencies.
Performance:
- Image loading per candidate is acceptable; no heavy caching required.
File ownership:
- Implementer edits labeling app and docs only.
API stability:
- No breaking changes to public CLI interfaces.
Style:
- Small, focused diff; add docstrings only for new/changed public functions.

## Acceptance criteria
- Newly regenerated spectrogram PNGs display correctly without stale images.
- App includes a clear refresh control.
- `LABELING_TOOL_QUICKSTART.md` describes the refresh behavior.

## File touch list
New files:
- None.
Modified files:
- src/usv_spectrogram/labeling/labeling_app.py
- LABELING_TOOL_QUICKSTART.md

## Plan (small diffs)
1) Update image loading to read bytes + add refresh control.
2) Update quickstart documentation.

## Implementer instructions
Do:
- Keep UI wording clear and minimal.
- Ensure refresh resets session state safely.
Do not:
- Do not add new dependencies.
- Do not change candidate ordering or schema.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
