# Implementation Notes

## Summary of implementation
- Adjusted dataset loading to resolve `_noise_` candidate PNGs based on label, with fallback to the alternate folder.
- Recreated dataset splits so skip-noise USV samples are included.

## Decisions and tradeoffs
- For `_noise_` candidate_ids labeled USV, prefer `spectrograms_review/` and fallback to `noise_samples/`.
- For `_noise_` labeled Not USV, prefer `noise_samples/` and fallback to `spectrograms_review/`.

## Commands used during development
- `.\.venv\Scripts\python.exe -m py_compile .\src\usv_spectrogram\dataset\splits.py`
- `.\.venv\Scripts\python.exe .\scripts\prepare_dataset.py --create-splits`

## How to run
- `.\.venv\Scripts\python.exe .\scripts\prepare_dataset.py --create-splits`

## Known limitations / TODOs
- None noted.

## Files changed
- src/usv_spectrogram/dataset/splits.py
