# Implementation Notes

## Summary of implementation
- Load spectrogram images as bytes to avoid stale caching when PNGs are regenerated in-place.
- Added a sidebar control to reload data from disk by clearing session state.
- Updated labeling quickstart with refresh guidance.

## Decisions and tradeoffs
- Byte loading favors correctness over caching; per-image read cost is acceptable for interactive review.
- Refresh control clears session state keys without touching files.

## Commands used during development
- `.\.venv\Scripts\python.exe -m py_compile .\src\usv_spectrogram\labeling\labeling_app.py`

## How to run
- `.\.venv\Scripts\python.exe -m streamlit run .\scripts\usv_labeling_tool.py`

## Known limitations / TODOs
- If candidate IDs change, regenerate `candidates_optimized.csv` to match new PNG filenames.

## Files changed
- src/usv_spectrogram/labeling/labeling_app.py
- LABELING_TOOL_QUICKSTART.md
