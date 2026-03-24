# Implementation Notes

## Summary of implementation
- Added an archive/reset helper that backs up `labels.csv`, moves labeled PNGs to a timestamped archive folder, and clears labels.
- Added sidebar UI with confirmation checkbox + button to trigger the archive/reset flow and show a success notice.
- Updated labeling quickstart with the archive/reset workflow.

## Decisions and tradeoffs
- Only labeled PNGs are moved; unlabeled images remain in `spectrograms_review`.
- Archive uses a timestamped folder under `labeling_archives/` to avoid collisions.
- If `labels.csv` is missing, the archive is reconstructed from in-memory labels.

## Commands used during development
- `@'...python script...'@ | .\.venv\Scripts\python.exe -`
- `.\.venv\Scripts\python.exe -m py_compile .\src\usv_spectrogram\labeling\labeling_app.py`

## How to run
- `.\.venv\Scripts\python.exe -m streamlit run .\scripts\usv_labeling_tool.py`

## Known limitations / TODOs
- If labeled PNGs are already missing, they are counted as missing and not archived.
- After archiving, `spectrograms_review` may need to be regenerated before re-labeling.

## Files changed
- src/usv_spectrogram/labeling/labeling_app.py
- LABELING_TOOL_QUICKSTART.md
