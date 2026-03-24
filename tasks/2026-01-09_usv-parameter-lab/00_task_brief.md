# Task Brief

Title: USV Parameter Lab (Streamlit)
Date: 2026-01-09

## Goal
Create a Streamlit-based USV Parameter Lab that reuses existing streaming/in-memory spectrogram components to let users explore parameter effects on a selected WAV segment, compare baseline vs variant settings, run heuristic overlays, and export sweep reports.

## Context
Assumptions:
- User approves adding Streamlit as a dependency (explicitly requested in prompt).
- Existing spectrogram pipeline (config, in-memory STFT, streaming STFT, render helpers) is the foundation; new code composes these APIs rather than rewrites them.
- The app runs from `scripts/usv_parameter_lab.py` and uses `src/usv_spectrogram` modules for core logic.
Uncertainties:
- Exact entrypoint selection if repo has preferred script conventions beyond `scripts/make_spectrogram.py`.
- Whether to add new helper(s) in `io_wav.py` for segment reads vs a new module under `param_lab/`.

## Scope
In scope:
- Streamlit UI for segment selection, parameter controls, side-by-side comparison, optional difference view, and heuristic overlay.
- Segment-based WAV reads using SoundFile seek/read (no full-file loads).
- Reuse of `SpectrogramConfig`, `compute_spectrogram_db`, and/or `stream_wav_spectrogram_db` where applicable.
- Metrics computation, overlay stats, and timing measurements for baseline/variant.
- Sweep grid export with images and a Markdown/HTML report plus config JSON/YAML.
- README update with how-to-run, and minimal tests per requirements.
Out of scope:
- ML-based detection or model training.
- GUI frameworks beyond Streamlit.
- Full-file processing inside the interactive app.

## Constraints
Dependencies:
- Add Streamlit only (plus any minimal supporting dependency already in the stack if required). Confirm no extra dependencies.
Performance:
- Segment-only reads for large WAVs; cache STFT results to avoid recompute on display-only changes.
File ownership:
- Implementer owns new `src/usv_spectrogram/param_lab/*` and `scripts/usv_parameter_lab.py`.
- Avoid editing unrelated scripts or existing pipeline modules unless adding narrowly-scoped helpers.
API stability:
- No breaking changes to existing public functions; new helpers are additive.
Style:
- Keep small diffs; add docstrings for new public functions; keep ASCII.

## Acceptance criteria
- Streamlit app can load a WAV, select a segment, and update spectrograms on parameter changes.
- Baseline can be locked; variant can be edited; both views share axes and dB scaling.
- Optional heuristic overlay shows candidate boxes/contours and outputs candidate metrics.
- Metrics panel shows noise floor, band energy stats, contrast proxy, counts, and timings.
- Sweep tool exports a folder with images, a compact report, and saved configs.
- README includes run instructions: `pip install -r requirements.txt` and `streamlit run scripts/usv_parameter_lab.py`.
- No full-file loads during interactive use; segment reads only.

## File touch list
New files:
- `src/usv_spectrogram/param_lab/app.py`
- `src/usv_spectrogram/param_lab/metrics.py`
- `src/usv_spectrogram/param_lab/heuristic_detect.py`
- `src/usv_spectrogram/param_lab/sweep.py`
- `src/usv_spectrogram/param_lab/explain.py`
- `scripts/usv_parameter_lab.py`
- `tests/test_param_lab_segment.py`
- `tests/test_param_lab_heuristic.py`
Modified files:
- `requirements.txt`
- `README.md`
- Optional: `src/usv_spectrogram/io_wav.py` (only if adding a segment-read helper)
- Optional: `src/usv_spectrogram/__init__.py` (export new utilities if needed)

## Plan (small diffs)
1) Stage 1: Segment read helper + single spectrogram render + parameter explanations panel.
2) Stage 2: Baseline/variant side-by-side comparison + shared scaling + cache logic.
3) Stage 3: Heuristic overlay + metrics computation + timing capture.
4) Stage 4: Sweep/report export + README update + tests + quick sanity checks.

## Implementer instructions
Do:
- Reuse `SpectrogramConfig`, `compute_spectrogram_db`, and `stream_wav_spectrogram_db` for consistency.
- Keep the app segment-first and cache STFT results keyed by audio segment + STFT params.
- Document new public functions with concise docstrings.
- Record decisions in `10_impl_notes.md`.
Do not:
- Load entire WAVs in the interactive app.
- Change existing APIs or default behavior without explicit approval.
- Add extra dependencies beyond Streamlit without asking.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
