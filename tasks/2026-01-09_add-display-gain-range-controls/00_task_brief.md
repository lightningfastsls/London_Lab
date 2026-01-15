# Task Brief

Title: Add gain/range controls in USV Parameter Lab
Date: 2026-01-09

## Goal
Allow users to adjust gain and range (dB) in the Streamlit USV Parameter Lab so they can better isolate USVs during visual inspection and export.

## Context
Assumptions:
- The request refers to display gain/range controls within the Streamlit app.
- Gain/range are controlled separately for baseline and variant views.
- Difference view uses the variant display settings.
- Sweep export uses the baseline display settings unless explicitly overridden.
Uncertainties:
- None.

## Scope
In scope:
- Add separate UI controls for baseline and variant gain/range (dB).
- Ensure render paths honor the selected gain/range for their view.
Out of scope:
- Changes to the underlying spectrogram computation algorithm.
- Additional dependencies beyond what is already in the project.

## Constraints
Dependencies:
- No new dependencies without explicit approval.
Performance:
- No additional full-file loads; continue segment-only reads.
File ownership:
- Implementer edits: `src/usv_spectrogram/param_lab/app.py` (UI and wiring).
API stability:
- No breaking changes to public APIs.
Style:
- Keep diffs small; ASCII only.

## Acceptance criteria
- User can set gain (dB) and range (dB) independently for baseline and variant.
- Changes affect baseline/variant plots and the difference view uses variant settings.
- Sweep export visuals reflect the baseline gain/range settings.
- Documentation reflects the new/updated controls if any user-facing behavior changes.

## File touch list
New files:
Modified files:
- `src/usv_spectrogram/param_lab/app.py`
- `README.md` (only if user-facing instructions change)
- `tasks/2026-01-09_add-display-gain-range-controls/10_impl_notes.md`

## Plan (small diffs)
1) Add baseline/variant gain and range controls in the UI.
2) Wire gain/range values through plot and sweep paths; update docs if needed.

## Implementer instructions
Do:
- Keep changes minimal and localized to the app UI wiring.
- Respect existing segment-only behavior and caching.
Do not:
- Add dependencies or alter core spectrogram functions without approval.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
