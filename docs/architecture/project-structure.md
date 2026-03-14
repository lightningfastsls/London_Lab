# Project Structure

Full directory tree for the USV Spectrogram Generator project.

```
src/usv_spectrogram/          # Core USV library
  config.py                   # SpectrogramConfig dataclass
  io_wav.py                   # WAV loading (always specify sr=300000)
  spectrogram.py              # STFT computation
  _stft_core.py               # Low-level STFT internals
  stft_stream.py              # Streaming STFT for long files
  render_tiles.py             # Tiled PNG rendering
  storage_zarr.py             # Zarr incremental storage
  detection/                  # Energy-based USV candidate detection
    energy_detector.py        # EnergyDetector class
    candidate.py              # Candidate dataclass
    spectrogram_extractor.py  # Extract candidate spectrograms
  models/                     # CNN classifier
    cnn_classifier.py         # USVClassifier architecture
    trainer.py                # Training loop
    data_loader.py            # Dataset/DataLoader
    evaluate.py               # Evaluation metrics
  dataset/                    # Training data assembly (Phase 9.1)
    assembler.py              # DatasetAssembler pipeline
    splits.py                 # Train/val/test splitting
  app/                        # PyQt6 detection desktop app
    main_window.py            # Main application window
    core/                     # Detection logic, audio, inference
    widgets/                  # SpectrogramView, ProbabilityView
  clustering/                 # USV repertoire clustering
  classification/             # Raven export, DeepSqueak import, repertoire stats
  lmt/                        # Live Mouse Tracker integration
  param_lab/                  # Streamlit parameter lab
  labeling/                   # Streamlit labeling + noise review tools

usv_language/                 # Transformer + VQ-VAE for USV compositional structure
  models/                     # transformer.py, vqvae.py
  data/                       # Bout extraction, normalization, spectrogram prep
  training/                   # train_vqvae.py, train_transformer.py
  analysis/                   # Codebook viz, sequence analysis, compositionality

parts-finder/                 # Israeli auto parts lookup (separate subproject)
notion_notes/                 # Notion KB automation CLI

scripts/                      # ~76 entry points (see docs/scripts-index.md)
tests/                        # pytest test suite
methodology/                  # arscontexta research claims (249 files, READ-ONLY)
reference/                    # arscontexta structured reference (READ-ONLY)
```

All `src/` paths above are relative to `src/usv_spectrogram/` unless they start with `usv_language/`.
