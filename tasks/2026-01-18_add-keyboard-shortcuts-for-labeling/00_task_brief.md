# Task Brief

Title: Add keyboard shortcuts for labeling (1/2/3)
Date: 2026-01-18

## Goal
Enable keyboard shortcuts so pressing 1/2/3 triggers the USV / Not USV / Uncertain labels (and the existing auto-advance behavior), without requiring scrolling to the buttons.

## Context
Assumptions:
- Label buttons already exist in `render_labeling_controls()` with stable keys like `label_<label>`.
- `labeling_app.py` is the only file that needs logic changes.
- The preferred approach is to avoid new dependencies unless explicitly approved.
Uncertainties:
- Whether `requirements.txt` exists; if adding a dependency is acceptable.
- Exact button keys and button text; needs inspection to map shortcuts correctly.

## Scope
In scope:
- Add keyboard shortcut handling for 1/2/3 mapped to USV / Not USV / Uncertain.
- Ensure shortcuts trigger the same labeling path as button clicks and still advance.
- If adding a dependency is approved, wire it in and update requirements accordingly.
Out of scope:
- Changes to data formats, labeling schema, or UI redesigns.
- Global hotkeys beyond the labeling page.

## Constraints
Dependencies:
- Do not add new dependencies without explicit approval. If needed, ask first.
Performance:
- Keyboard handler must be lightweight and not degrade app responsiveness.
File ownership:
- Only modify files listed in the touch list.
API stability:
- Keep public interfaces stable; if function signatures change, update callers in the same file.
Style:
- Minimal diffs; follow existing code style and Streamlit patterns.

## Acceptance criteria
- Press 1 labels as USV and advances.
- Press 2 labels as Not USV and advances.
- Press 3 labels as Uncertain and advances.
- Shortcuts work without scrolling to the buttons.
- App still runs via `.\.venv\Scripts\streamlit.exe run scripts/usv_labeling_tool.py`.
- If a dependency is approved/added, update requirements file accordingly.

## File touch list
New files:
- None.
Modified files:
- `src/usv_spectrogram/labeling/labeling_app.py`
- `requirements.txt` (only if dependency is approved and repo uses it)

## Plan (small diffs)
1) Inspect `labeling_app.py` to confirm button keys and locate the best hook point for shortcut wiring.
2) Implement keyboard shortcut handling (prefer JS injection to avoid new dependencies) that triggers the same label actions.
3) Verify shortcut mapping and ensure no regressions in labeling or auto-advance.

## Implementer instructions
Do:
- Default to a no-new-dependency solution unless the user explicitly approves adding `streamlit-shortcuts`.
- Map keys 1/2/3 to the existing label buttons using their keys or deterministic identifiers.
- Keep changes localized and minimal.
Do not:
- Add dependencies without approval.
- Change labeling semantics or data outputs.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
