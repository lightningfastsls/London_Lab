# Task Brief

Title: Fix variant updates and stack layout in USV Parameter Lab
Date: 2026-01-09

## Goal
Ensure variant parameters actually update the variant plot and difference view, and change the layout so baseline and variant plots are full-width stacked vertically (baseline on top, variant below), keeping the sidebar intact.

## Context
Assumptions:
- The variant plot not changing is due to caching or display wiring rather than the underlying spectrogram compute.
- The difference view currently renders, but is not reflecting changes when the variant changes.
- Layout change only affects main panel; sidebar remains unchanged.
Uncertainties:
- Whether cache keys need to include display-only settings or only audio/STFT settings.
- Whether the difference view should be full-width as well (assume yes).

## Scope
In scope:
- Fix variant plot updates (including cache behavior if needed).
- Fix difference view to reflect actual baseline/variant differences.
- Stack baseline and variant plots vertically, full width of the content area.
- Keep baseline/variant parameter controls and sidebar structure intact.
Out of scope:
- Changes to the underlying spectrogram computation algorithm.
- New dependencies or Streamlit custom components.

## Constraints
Dependencies:
- No new dependencies without explicit approval.
Performance:
- Continue segment-only reads; avoid full-file loads.
File ownership:
- Implementer edits: `src/usv_spectrogram/param_lab/app.py`.
API stability:
- No breaking changes to public APIs.
Style:
- Keep diffs small; ASCII only.

## Acceptance criteria
- Variant controls cause visible changes in the variant plot when parameters differ.
- Difference view changes when baseline/variant settings differ (and is full-width).
- Baseline plot spans full content width; variant plot spans full content width and appears below baseline.
- Sidebar layout and controls remain unchanged.

## File touch list
New files:
Modified files:
- `src/usv_spectrogram/param_lab/app.py`
- `README.md` (only if user-facing instructions change)
- `tasks/2026-01-09_fix-variant-updates-and-layout-stack/10_impl_notes.md`

## Plan (small diffs)
1) Diagnose why variant plot/diff view are not updating; fix caching or wiring.
2) Adjust layout to stack full-width baseline/variant plots; update docs if needed.

## Implementer instructions
Do:
- Verify cache keys include any values required to trigger recompute.
- Keep rendering logic consistent with selected baseline/variant settings.
Do not:
- Change core spectrogram functions or add dependencies.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
