# Implementation Notes

Date: 2026-01-24
Task: Extract USV >120ms spectrograms for review

## Decisions
- Use `candidates_with_onsets_extracted.csv` as the source (model-facing spectrograms without axes/green lines).
- Filter to labeled `USV` only, joining on `candidate_id` from `labels.csv`.
- Output folder default: `spectrograms_training_usv_over_120ms/`.
- No new script; perform a one-off copy via PowerShell.

## Changes
- No code changes kept; removed the previously added helper script per request.

## Notes
- Copy operation performed via PowerShell in the working tree.
