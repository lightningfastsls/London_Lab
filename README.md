# USV Spectrogram Generator

This project provides in-memory and streaming spectrogram generation for
250 kHz USV WAVs, plus tiled PNG rendering and incremental Zarr storage APIs.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python scripts/make_spectrogram.py --input path\to\file.wav
```

Optional output path:

```powershell
python scripts/make_spectrogram.py --input path\to\file.wav --output out.png
```

Tiled pages for long files:

```powershell
python scripts/make_spectrogram.py --input path\to\file.wav --tiled
```

Sample-rate handling (non-250 kHz data):

```powershell
python scripts/make_spectrogram.py --input path\to\file.wav --auto-sample-rate
```

Streaming + Zarr API (module usage):

- `usv_spectrogram.stft_stream.stream_wav_spectrogram_db`
- `usv_spectrogram.storage_zarr.init_spectrogram_store`
- `usv_spectrogram.storage_zarr.append_spectrogram_chunk`

## DSP notes

- Window: 2048 samples, Hann (about 8.2 ms at 250 kHz).
- Hop: 0.5 ms (125 samples at 250 kHz).
- FFT: zero padding factor 2 (n_fft = 4096) for smoother bins.
- Band: 30 kHz to 125 kHz crop for USV focus.
- dB scaling: magnitude to dB with a small epsilon; display uses gain 20 dB and
  range 40 dB for consistent contrast.
