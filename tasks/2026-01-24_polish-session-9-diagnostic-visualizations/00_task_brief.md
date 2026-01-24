# Task Brief

Title: Polish Session 9 Diagnostic Visualizations
Date: 2026-01-24

## Goal
Improve the aesthetics and readability of Session 9 diagnostic plots while preserving existing data and meaning.

## Context
Assumptions:
- `scripts/threshold_sweep.py` and `scripts/compare_probability_distributions.py` produce plots using matplotlib.
- Plot outputs are written under `analysis/` and should remain the same files/paths unless currently different.
Uncertainties:
- Where model name / dataset metadata is sourced for plot titles (confirm in script or use a safe fallback).

## Scope
In scope:
- Plot styling changes only in the two scripts listed above.
- Add annotations/lines for thresholds 0.25 and 0.50.
- Improve layout, fonts, legend placement, and table readability.
Out of scope:
- Any change to data loading, metrics, or calculation logic.
- Adding new dependencies or saving new output formats.

## Constraints
Dependencies: Do not add new Python packages.
Performance: Avoid heavy recomputation or extra data passes solely for styling.
File ownership: Modify only the two plotting scripts.
API stability: Do not change public interfaces or CLI arguments.
Style: Use a consistent, colorblind-friendly palette and subtle gridlines.

## Acceptance criteria
- Threshold sweep plot: improved palette, subtle gridlines, readable fonts, legend placed neatly, annotations for 0.25 and 0.50, and a title that includes model/dataset metadata (or a safe fallback like "Model: <name> | Dataset: <name>").
- Probability distribution plots: consistent colors across subplots, ~50 bins for histograms, vertical reference lines at 0.25 and 0.50, readable table formatting, and labels legible at 150 DPI.
- Outputs still generate without errors using the existing scripts.

## File touch list
New files: None.
Modified files:
- `scripts/threshold_sweep.py`
- `scripts/compare_probability_distributions.py`

## Plan (small diffs)
1) Inspect both scripts to identify current plot configuration and output paths.
2) Implement palette + typography constants and apply to both plots.
3) Add annotations/reference lines and adjust layout/table formatting.
4) Run the scripts to confirm plots render and output files exist.

## Implementer instructions
Do:
- Prefer centralized style variables (colors, font sizes) for consistency.
- Keep changes limited to plotting configuration and labels.
Do not:
- Change computed metrics, thresholds, or data pipelines.
- Introduce new dependencies or file outputs.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run the two scripts per `codex_tasks/task2_visualization_polish.md` using `.venv` if available.
- Confirm output images exist in `analysis/` and are non-empty.
- Record commands and results in `20_verification.md`.
