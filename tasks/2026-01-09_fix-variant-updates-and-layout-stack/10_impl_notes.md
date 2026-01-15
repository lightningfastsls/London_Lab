# Implementation Notes

## Summary of implementation
- Fixed the Streamlit cache key so variant STFT changes invalidate cached results.
- Stacked baseline and variant plots vertically at full width; made the difference view full width.

## Decisions and tradeoffs
- Introduced an explicit cache key tuple to avoid relying on hashing the dataclass dict.

## Commands used during development
- None.

## How to run
- `streamlit run scripts/usv_parameter_lab.py`

## Known limitations / TODOs
- None.

## Files changed
- src/usv_spectrogram/param_lab/app.py
