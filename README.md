# USV Analysis Toolkit

Python tools for analyzing ultrasonic vocalization (USV) recordings from mice at 300 kHz sample rate. Covers the full pipeline from raw WAV files through spectrogram generation, automated detection, CNN classification, repertoire clustering, and compositional analysis.

> **New here?** Read **[`docs/SUCCESSOR_ONBOARDING.md`](docs/SUCCESSOR_ONBOARDING.md)** for a clone → install → run walkthrough.

## Setup

Developed on **WSL/Linux (POSIX), Python 3.12**. (Older docs show Windows
`.\.venv\Scripts\python.exe` paths — ignore those; this environment is POSIX.)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements_frozen.txt   # pinned, reproduces the production env
.venv/bin/python -m pip install -e .                         # makes `import usv_spectrogram` work everywhere
```

`requirements_frozen.txt` is pinned — prefer it over the unpinned `requirements.txt`.
The editable install replaces the legacy `sys.path.insert(".../src")` hacks.

**WAV files** are spread across multiple directories (`5970/`, `USV_3452_sample/`,
`USV_9252/`, `USV_lab_131204*/`, …) — there is no single canonical folder. See
[`docs/DATA_LOCATIONS.md`](docs/DATA_LOCATIONS.md).

## Production detection pipeline

The supported batch command. **All five flags are required** — omitting the
fp-filter or hysteresis-config silently produces incomplete triage:

```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir <WAV_FOLDER>/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_<NAME>/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```

Production CNN: `models/hard_neg_retrain/best_model.pt` (precision 90.55%, 16/18
known-noise files eliminated). Do **not** use `models/matched_windows/` or
`models/production/` — deprecated baselines.

## Spectrogram generation

STFT-based spectrograms: single-file PNGs, tiled rendering for long recordings,
streaming, and Zarr incremental storage.

```bash
.venv/bin/python scripts/make_spectrogram.py --input path/to/file.wav
.venv/bin/python scripts/make_spectrogram.py --input path/to/file.wav --tiled
```

## CNN classification

Binary USV/noise classifier on labeled spectrograms: training loop, evaluation,
threshold optimization, active-learning support.

```bash
.venv/bin/python scripts/train_cnn.py --config config.yaml
.venv/bin/python scripts/evaluate_model.py --model model.pt --data test/
```

## Desktop App (PyQt6)

Interactive detection: synchronized spectrogram + probability views, real-time
sliding-window inference, label management. **This is the canonical interactive tool.**

```bash
.venv/bin/python scripts/run_app.py
```

> The app's **Detect** = CNN + hysteresis only (raw probabilities; no
> fp-filter/temperature/soft-notch). Those run in the batch pipeline above.

> **Deprecated:** the old Streamlit Parameter Lab / Labeling / Noise-Review tools
> are superseded by the PyQt6 app and are not maintained.

## Clustering & Classification Bridge

- **Repertoire clustering** — extract acoustic features, cluster into call types, visualize.
- **DeepSqueak / Raven bridge** — export detections as Raven selection tables for DeepSqueak (MATLAB) classification; import results back for repertoire statistics.

## VQ-VAE / Transformer (`usv_language/`)

Compositional structure analysis of USV sequences using VQ-VAEs and Transformers.
`usv_language/` is a **separate package** at the repo root (not under `src/`, not
installed by `pip install -e .`) — run its scripts directly.

## LMT Integration

Event-triggered USV analysis synchronized with Live Mouse Tracker behavioral data.

## DSP conventions

Canonical signal-processing constants (sample rate, USV band, STFT params) are
**enforced in code** at `src/usv_spectrogram/corpus.py` — import them, never
redeclare. Rationale in `DECISIONS.md` (ADR-001 sample rate, ADR-002 STFT).
Key rule: always pass `sr=300000` explicitly (or `corpus.SAMPLE_RATE_HZ`); never
rely on library defaults.

## Tests

```bash
.venv/bin/python -m pytest          # ~1584 tests; config in pyproject.toml
```

Dead-collection tests (targeting removed/archived code) live in `tests/archive/`
and are excluded from discovery — see `tests/archive/README.md`.

## Project Structure

See `CLAUDE.md` for the full annotated structure. Key directories:

- `src/usv_spectrogram/` — core library (detection, models, dataset, app, clustering, classification, labeling, param_lab, lmt)
- `usv_language/` — Transformer + VQ-VAE compositional analysis (separate package)
- `scripts/` — 154 top-level entry points + `scripts/experiments/` ([index](docs/scripts-index.md))
- `tests/` — pytest suite (`tests/archive/` = retired)
- `docs/` — handoffs, plans, module docs, onboarding
