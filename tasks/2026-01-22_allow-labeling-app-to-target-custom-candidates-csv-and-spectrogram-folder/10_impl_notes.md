# Implementation Notes

## Summary of implementation
- Added sidebar inputs for candidates CSV and spectrograms folder, with a form button to apply paths.
- Resolved user-provided paths (absolute or repo-relative) and used them for candidate loading and image display.
- Updated labeling quickstart with custom path selector instructions.

## Decisions and tradeoffs
- Uses a form submit to avoid reruns on every keystroke.
- Keeps default paths to repo root files for backward compatibility.

## Commands used during development
- `.\.venv\Scripts\python.exe -m py_compile .\src\usv_spectrogram\labeling\labeling_app.py`

## How to run
- `.\.venv\Scripts\python.exe -m streamlit run .\scripts\usv_labeling_tool.py`

## Known limitations / TODOs
- Labels still save to the default `labels.csv` unless we add a separate selector.

## Files changed
- src/usv_spectrogram/labeling/labeling_app.py
- LABELING_TOOL_QUICKSTART.md
