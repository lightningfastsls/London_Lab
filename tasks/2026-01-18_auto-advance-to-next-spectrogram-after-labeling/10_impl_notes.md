# Implementation Notes

## Summary of implementation
- Updated labeling controls to advance after saving a label, preferring the next unlabeled candidate with a safe fallback.

## Decisions and tradeoffs
- Expanded `render_labeling_controls` parameters to accept candidates, labels, and index to keep auto-advance logic localized.

## Commands used during development
- `Get-Content -LiteralPath src\\usv_spectrogram\\labeling\\labeling_app.py`

## How to run
- `.\.venv\Scripts\streamlit.exe run scripts\usv_labeling_tool.py`

## Known limitations / TODOs
- None.

## Files changed
- src/usv_spectrogram/labeling/labeling_app.py
- tasks/2026-01-18_auto-advance-to-next-spectrogram-after-labeling/10_impl_notes.md
