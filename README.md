# USV Analysis Toolkit

Python tools for analyzing ultrasonic vocalization (USV) recordings from mice at 300 kHz sample rate. Covers the full pipeline from raw WAV files through spectrogram generation, automated detection, CNN classification, repertoire clustering, and compositional analysis.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Run scripts with the project venv:

```powershell
.\.venv\Scripts\python.exe <script>
```

WAV files: set `$env:USV_WAV_DIR` or place in `5970 USV/` at the repo root.

## Core Pipeline

**Spectrogram generation** — STFT-based spectrograms with configurable window, hop, and FFT parameters. Supports single-file PNGs, tiled rendering for long recordings, streaming for memory-constrained use, and Zarr incremental storage.

```powershell
python scripts/make_spectrogram.py --input path\to\file.wav
python scripts/make_spectrogram.py --input path\to\file.wav --tiled
```

**USV detection** — Energy-based candidate detection in the 30-125 kHz band with configurable thresholds, merge gaps, and boundary adjustment.

```powershell
python scripts/run_detection.py --input path\to\file.wav
```

**CNN classification** — Binary USV/noise classifier trained on labeled spectrograms. Includes training loop, evaluation metrics, threshold optimization, and active learning cycle support.

```powershell
python scripts/train_cnn.py --config config.yaml
python scripts/evaluate_model.py --model model.pt --data test/
```

## Desktop App (PyQt6)

Interactive detection application with synchronized spectrogram and probability views, real-time sliding-window inference, and label management.

```powershell
python scripts/run_app.py
```

## Streamlit Tools

- **Parameter Lab** — Interactive STFT parameter exploration with baseline vs variant comparison and sweep reports.
- **Labeling Tool** — Manual USV candidate labeling with spectrogram review.
- **Noise Review** — Review and curate noise samples for training.

```powershell
streamlit run scripts/usv_parameter_lab.py
streamlit run scripts/usv_labeling_tool.py
```

## Clustering & Classification Bridge

**Repertoire clustering** — Extract acoustic features from detected USVs, cluster into call types, and visualize the repertoire.

**DeepSqueak / Raven bridge** — Export detections as Raven selection tables for import into DeepSqueak (MATLAB) for syllable classification. Import results back for repertoire statistics.

## VQ-VAE / Transformer (usv_language)

Compositional structure analysis of USV sequences using Vector-Quantized Variational Autoencoders and Transformer models. Includes codebook visualization, sequence analysis, and compositionality metrics.

## LMT Integration

Event-triggered USV analysis synchronized with Live Mouse Tracker behavioral data.

## DSP Notes

- Sample rate: **300 kHz** (always specify `sr=300000` explicitly)
- Window: 2048 samples, Hann (~6.8 ms at 300 kHz)
- Hop: 0.5 ms (150 samples at 300 kHz)
- FFT: zero-padded to 4096 for smoother bins
- Band: 30-125 kHz crop for USV focus
- dB scaling: magnitude-to-dB with epsilon floor; display gain 20 dB, range 40 dB

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## Project Structure

See `CLAUDE.md` for the full annotated structure. Key directories:

- `src/usv_spectrogram/` — Core library (detection, models, dataset, app, clustering, classification, labeling, param_lab, lmt)
- `usv_language/` — Transformer + VQ-VAE compositional analysis
- `scripts/` — ~76 entry points ([index](docs/scripts-index.md))
- `tests/` — pytest suite
