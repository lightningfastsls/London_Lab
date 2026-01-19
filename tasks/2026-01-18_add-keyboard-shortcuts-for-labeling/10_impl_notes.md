# Implementation Notes

## Summary of implementation
- Stage 1 complete: inspected `labeling_app.py` to confirm label button keys (`label_<label>`) and current labeling flow.
- Stage 2 complete: added JS keydown listener + query param hook to trigger label actions for 1/2/3 without new dependencies.
- Stage 3 complete: launched Streamlit to verify app loads with shortcut changes.
- Follow-up fix: replaced query-param shortcut handling with parent-document keydown listener that clicks label buttons directly.

## Decisions and tradeoffs
- Used `st.components.v1.html` with a guarded keydown listener and query params to avoid new dependencies.
- Switched to DOM button click dispatch to avoid relying on query param navigation from an iframe.

## Commands used during development
- `Get-Content -LiteralPath src\\usv_spectrogram\\labeling\\labeling_app.py`
- `.\.venv\Scripts\streamlit.exe run scripts\usv_labeling_tool.py`

## How to run

## Known limitations / TODOs
- Shortcuts need manual browser interaction to validate; app launched successfully.
- Shortcut JS relies on button text prefix matching; adjust if button labels change.

## Files changed
- tasks/2026-01-18_add-keyboard-shortcuts-for-labeling/10_impl_notes.md
- src/usv_spectrogram/labeling/labeling_app.py
