# Implementer Notes

- Added segment-only WAV reads via `load_wav_segment_mono` for the Streamlit lab.
- Built a Streamlit UI that reuses `SpectrogramConfig` and `compute_spectrogram_db` with cached segment + STFT results.
- Implemented heuristic candidate detection with connected components and summary metrics.
- Added sweep export that renders PNGs and writes a compact markdown report plus JSON configs/metrics.
- Kept display gain/range shared across baseline and variant to satisfy shared scaling requirement.
- Guarded the Streamlit input so empty paths or non-WAV paths do not trigger `soundfile` errors.
