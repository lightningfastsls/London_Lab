# Task Brief

Title: Auto-advance to next spectrogram after labeling
Date: 2026-01-18

## Goal
Make the Streamlit labeling tool automatically advance to the next spectrogram (preferably next unlabeled) immediately after a label button is clicked, without crashing at the end of the list.

## Context
Assumptions:
- The labeling UI is controlled by `render_labeling_controls()` in `src/usv_spectrogram/labeling/labeling_app.py`.
- `find_next_unlabeled()` exists and can be used to skip already-labeled items.
- `st.session_state.current_index` and `st.session_state.total_candidates` are already used to track position.
Uncertainties:
- Whether `render_labeling_controls()` currently receives `candidates` or can be easily updated to do so.
- Whether `st.success()` is still desired; default to keeping unless it complicates rerun.

## Scope
In scope:
- Update `render_labeling_controls()` to advance after labeling.
- If feasible, use `find_next_unlabeled()` to skip labeled entries; otherwise advance by +1.
- Ensure last item does not crash and no out-of-range access occurs.
Out of scope:
- Changes to data loading, labeling schema, or UI layout beyond required wiring.
- New dependencies or major refactors.

## Constraints
Dependencies:
- No new dependencies without asking.
Performance:
- Must remain responsive; avoid full recomputation on each click beyond existing behavior.
File ownership:
- Only modify files listed in the touch list.
API stability:
- Keep public interfaces stable; if function signatures change, update callers in the same file.
Style:
- Minimal diffs; follow existing code style and Streamlit patterns in the file.

## Acceptance criteria
- Clicking USV/Not USV/Uncertain advances to the next spectrogram (or next unlabeled if available).
- Labeling the last spectrogram does not crash and does not advance past the end.
- App still runs via `.\.venv\Scripts\streamlit.exe run scripts/usv_labeling_tool.py`.

## File touch list
New files:
- None.
Modified files:
- `src/usv_spectrogram/labeling/labeling_app.py`

## Plan (small diffs)
1) Inspect `render_labeling_controls()` and related call sites to see if `candidates` is already in scope or needs to be passed; identify best auto-advance logic.
2) Implement auto-advance after `save_label()`; prefer `find_next_unlabeled()` fallback to `current_index + 1` when within bounds; then rerun.
3) Update any affected callers/signatures in the same file and keep behavior stable for last item; optionally keep or remove `st.success()`.

## Implementer instructions
Do:
- Advance index before `st.rerun()` after saving a label.
- If using `find_next_unlabeled()`, pass `candidates` and handle `None` cleanly.
- Keep diffs small and localized.
Do not:
- Add new dependencies.
- Modify unrelated labeling logic or data formats.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
