# Implementation Notes

## Summary of implementation
- Added separate baseline and variant gain/range controls in the Streamlit sidebar.
- Wired baseline/variant display settings to their plots, sweep export, and difference view scaling.

## Decisions and tradeoffs
- Used variant range to set the symmetric scale for the difference view to keep controls consistent.

## Commands used during development
- None.

## How to run
- `streamlit run scripts/usv_parameter_lab.py`

## Known limitations / TODOs
- None.

## Files changed
- src/usv_spectrogram/param_lab/app.py
