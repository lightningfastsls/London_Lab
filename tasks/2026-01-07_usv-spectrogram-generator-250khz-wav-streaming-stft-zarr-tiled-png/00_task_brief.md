# Task Brief

Title: USV spectrogram generator (250kHz WAV, streaming STFT, zarr, tiled PNG)
Date: 2026-01-07

## Goal
Build a PyCharm-ready Python program that generates USV spectrograms from 250 kHz WAVs with sub-ms hop, supports streaming for large files, stores dB spectrograms incrementally (Zarr default), and renders tiled PNG pages for large files.

## Context
Assumptions:
- New repo structure can be created (src/, scripts/, tests/, README, pyproject/requirements).
- soundfile, numpy, scipy, matplotlib, zarr, tqdm are acceptable dependencies if requested.
Uncertainties:
- Preferred packaging file: pyproject.toml vs requirements.txt (ask before adding dependencies).
- Whether mel-warp display should be implemented or stubbed as an optional plotting-only transform.

## Scope
In scope:
- Implement modules: config, wav IO, streaming STFT, spectrogram math, tile renderer, zarr storage, utils.
- Provide a simple argparse script to run spectrogram generation.
- Ensure streaming STFT continuity with carry buffer across chunk boundaries.
- Default params match Audacity-like settings and USV band.
- Provide tiled PNG output for large files and incremental Zarr storage.
- Add at least one test for streaming vs in-memory equivalence.
- Update README with run instructions and DSP notes.
Out of scope:
- Full-featured CLI or GUI.
- Mel filter bank processing unless explicitly configured.
- Performance micro-optimizations beyond correctness/clarity.

## Constraints
Dependencies:
- Ask before adding new dependencies or changing public APIs.
- Prefer numpy/scipy/soundfile/matplotlib/zarr/tqdm; avoid librosa unless necessary.
Performance:
- Streaming for large WAVs; no full-file load for ~270 MB inputs.
- Chunked Zarr writes; no hold-then-dump.
File ownership:
- Spec Refiner touches only this task brief.
API stability:
- Keep public interfaces minimal; config is single source of truth.
Style:
- ASCII-only edits; concise comments, docstrings for new/changed public code.
- Parameter explanations co-located in config with effects and safe ranges.

## Acceptance criteria
- Repo has `src/usv_spectrogram/` modules, `scripts/make_spectrogram.py`, `tests/test_streaming_equivalence.py`, `README.md`, and packaging file (pyproject or requirements).
- Config dataclass defines all tunable parameters with explanatory comments per requirement.
- In-memory spectrogram works for small WAVs and renders PNG in USV band.
- Streaming STFT matches in-memory (test passes within tolerance).
- Large-file tiled pages render as `<basename>_page###.png` with default tile/page settings.
- Zarr storage is incremental and stores dB int16 plus required metadata.
- README includes run instructions and DSP notes (window vs hop, zero padding, band cropping).

## File touch list
New files:
- README.md
- pyproject.toml or requirements.txt
- src/usv_spectrogram/__init__.py
- src/usv_spectrogram/config.py
- src/usv_spectrogram/io_wav.py
- src/usv_spectrogram/stft_stream.py
- src/usv_spectrogram/spectrogram.py
- src/usv_spectrogram/render_tiles.py
- src/usv_spectrogram/storage_zarr.py
- src/usv_spectrogram/utils.py
- scripts/make_spectrogram.py
- tests/test_streaming_equivalence.py
Modified files:
- None expected unless repo already contains overlapping files.

## Plan (small diffs)
1) Create repo structure and config dataclass with parameter docs; add README skeleton and entrypoint script stub.
2) Implement in-memory spectrogram path and PNG rendering for small files.
3) Implement streaming STFT with carry buffer; add incremental Zarr storage.
4) Implement tiled-page rendering for large files.
5) Add streaming vs in-memory equivalence test; finalize README DSP notes.

## Implementer instructions
Do:
- Follow Audacity-like defaults: window 2048, Hann, zero padding factor 2 (n_fft 4096), hop_ms 0.5, f_min 30k, f_max 125k, gain_db 20, range_db 40.
- Use soundfile.SoundFile for streaming and maintain overlap buffer (window_length - hop_length).
- Store dB spectrogram as int16 with fixed scale in Zarr, with metadata.
- Use matplotlib for PNGs; tile into pages for long files.
- Add tqdm progress for large files.
Do not:
- Do not add librosa unless absolutely required (ask first).
- Do not implement a complex CLI; argparse only.
- Do not load full large WAV into memory.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
