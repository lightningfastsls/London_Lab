# Implementation Notes

## Summary of implementation
- Added expand helper for re-extracting spectrograms with expanded boundaries.
- Initialized expand-related session state and wired WAV dir resolution for future UI.
- Added expand UI (input/preview/save), expansion persistence in labels.csv, and expanded PNG output under `spectrograms_dir/expanded`.
- Updated labeling README and in-app guide text with expand instructions and CSV changes.
- Saving an expanded spectrogram now advances to the next candidate when a label exists.
- Pending expansions are retained per-candidate until a label is saved, preventing loss after navigation.

## Decisions and tradeoffs
- Reused SpectrogramExtractor with a suffixed candidate_id (`_expanded`) and clamped expansion start/end (including WAV duration if available).
- Stored `expand_ms` alongside labels in `labels.csv` (extra column) to avoid sidecar files while keeping compatibility.

## Commands used during development
- `python -m py_compile src\usv_spectrogram\labeling\labeling_app.py`

## How to run
- Use existing Streamlit entrypoint once remaining stages are implemented.

## Known limitations / TODOs
- README updates and sanity runs still pending (next stage).

## Files changed
- `src/usv_spectrogram/labeling/labeling_app.py`
