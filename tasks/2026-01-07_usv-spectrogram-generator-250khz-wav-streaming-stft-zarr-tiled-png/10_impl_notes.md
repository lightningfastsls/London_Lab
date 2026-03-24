# Stage 1-5 Implementation Notes

## Run instructions

```powershell
python -m pip install -r requirements.txt
python scripts/make_spectrogram.py --input path\to\file.wav
```

Optional output:

```powershell
python scripts/make_spectrogram.py --input path\to\file.wav --output out.png
```

## Decisions

- Chose `requirements.txt` for packaging with `numpy`, `scipy`, `soundfile`, `matplotlib`, `zarr`, `tqdm`.
- Implemented in-memory STFT via `scipy.signal.stft` and render via matplotlib `pcolormesh`.
- Script inserts `src/` into `sys.path` for direct execution without installation.
- Implemented streaming STFT with overlap carry and FFT via `np.fft.rfft`.
- Added Zarr storage with int16 dB encoding and metadata.
- Default input WAV folder set to `D:\mickey_london_lab\5970 USV` for relative inputs.
- Added tiled page rendering with configurable tile size and page layout.
- Align in-memory STFT implementation with streaming FFT framing to support equivalence testing.
- Added `--auto-sample-rate` flag for non-250 kHz WAVs.

## Files changed

- `requirements.txt`
- `README.md`
- `src/usv_spectrogram/__init__.py`
- `src/usv_spectrogram/config.py`
- `src/usv_spectrogram/io_wav.py`
- `src/usv_spectrogram/spectrogram.py`
- `src/usv_spectrogram/render_tiles.py`
- `src/usv_spectrogram/utils.py`
- `src/usv_spectrogram/stft_stream.py`
- `src/usv_spectrogram/storage_zarr.py`
- `scripts/make_spectrogram.py`
- `tests/test_streaming_equivalence.py`
- `tasks/2026-01-07_usv-spectrogram-generator-250khz-wav-streaming-stft-zarr-tiled-png/10_impl_notes.md`
@@
## Stage 2 status

- In-memory spectrogram + PNG rendering for small WAVs is complete.

## Stage 3 status

- Streaming STFT with carry buffer implemented in `src/usv_spectrogram/stft_stream.py`.
- Incremental Zarr storage implemented in `src/usv_spectrogram/storage_zarr.py`.

## Stage 4 status

- Tiled page rendering implemented in `src/usv_spectrogram/render_tiles.py` and exposed via `scripts/make_spectrogram.py --tiled`.

## Stage 5 status

- Streaming vs in-memory equivalence test added in `tests/test_streaming_equivalence.py`.
- README updated with current feature summary and module API pointers.

## Checks run

```powershell
python -m py_compile scripts/make_spectrogram.py src/usv_spectrogram/config.py src/usv_spectrogram/io_wav.py src/usv_spectrogram/spectrogram.py src/usv_spectrogram/render_tiles.py src/usv_spectrogram/utils.py
```

## Verification

- Verifier note: Stage 1 marked complete (see 20_verification.md).
