# USV Detection & Analysis Pipeline
## Project Status Summary — February 28, 2026

**PI Meeting Briefing | Mickey London Lab**

---

## Executive Summary

An end-to-end computational pipeline for detecting, classifying, and analyzing ultrasonic vocalizations (USVs) from 300 kHz mouse recordings. The pipeline spans from raw WAV files through deep learning classification to linguistic structure analysis.

**Timeline:** 6 weeks (Jan 16 – Feb 28, 2026)
**Status:** Core pipeline complete. Blocked on GPU for final training stage.

---

## What's Been Built (Completed)

### Signal Processing & Detection
| Component | Key Metric | Tests |
|-----------|-----------|-------|
| Energy detector (high-recall first stage) | 300 kHz, 20–120 kHz band | 42 |
| Spectrogram extraction (STFT: n_fft=512, hop=128) | 170 freq bins, 0.43 ms resolution | — |
| CNN binary classifier (USV vs noise) | **F1 91.7%** (P=89.7%, R=93.8%) | 38 |
| PyQt6 desktop detection app | Interactive threshold tuning | — |
| Unsupervised clustering (k-means, HDBSCAN, GMM) | Feature space visualization | — |

### Data Collection & Labeling
| Metric | Value |
|--------|-------|
| WAV recordings available | 6,491 files @ 300 kHz |
| Human-labeled USV candidates | 840 (458 USV, 374 noise, 8 uncertain) |
| Labeled recordings used | 5 (93 detections exported) |
| Labeling tools | Streamlit UI + PyQt6 desktop app |

### Deep Learning Architecture (Code Complete, Untrained on Real Data)
| Model | Parameters | Purpose |
|-------|-----------|---------|
| Autoregressive Transformer | **25.6M** | Predict next spectrogram column (learns temporal structure) |
| Hidden-State VQ-VAE | **820K** | Discover discrete "concepts" in transformer representations |
| CNN Classifier | **101K** | Binary USV detection (trained, operational) |

### Analysis Toolkit (Ready for Real Data)
| Module | What It Does |
|--------|-------------|
| Information Theory Metrics | MLE Zipf, bias-corrected entropy rates, burstiness |
| Null Model Generators | Shuffle, Markov, frequency-matched surrogates |
| Statistical Comparison | z-scores + p-values vs null models, PERMANOVA |
| Probing Framework | Linear/MLP probes on transformer hidden states |
| Acoustic Property Extractors | Peak frequency, centroid, energy, voicing, direction |
| Concept Manipulation | Inject/scan codebook entries, generative probing |
| Compositionality Analysis | Bigram productivity, positional independence |
| Sequence Analysis | Transition matrices, conditional entropy, MI decay |
| Repertoire Statistics | Syllable proportions, diversity, comparison across populations |

### DeepSqueak Classification Bridge
| Step | Status |
|------|--------|
| Export detections as Raven selection tables | **DONE** (5 files, 93 detections) |
| Import into MATLAB DeepSqueak for syllable clustering | **In progress** (manual MATLAB step) |
| Ingest classified results back into Python | Code written (`repertoire_stats.py`) |

### LMT Integration (Behavioral Correlation)
| Component | Status |
|-----------|--------|
| LMT SQLite database loader | **DONE** (read-only, flexible schema) |
| Temporal synchronizer (LMT frames ↔ WAV samples) | **DONE** (frame-rate alignment) |
| Event-triggered USV rate analysis | Waiting for LMT data files |

---

## Architecture Overview

```
Raw WAV (300 kHz)
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Energy      │───▶│  CNN         │───▶│  DeepSqueak      │
│  Detector    │    │  Classifier  │    │  Syllable Types  │
│  (high recall)│    │  (F1=91.7%) │    │  (via MATLAB)    │
└─────────────┘    └──────┬───────┘    └──────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  Bout        │
                   │  Extraction  │
                   │  (500ms gap) │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐    ┌──────────────────┐
                   │  Transformer │───▶│  VQ-VAE          │
                   │  (25.6M)     │    │  (K=64 codes)    │
                   │  next-column │    │  discrete concepts│
                   └──────────────┘    └──────┬───────────┘
                                              │
                                              ▼
                                       ┌──────────────────┐
                                       │  Analysis Suite   │
                                       │  • Zipf's law     │
                                       │  • Entropy rates  │
                                       │  • Null models    │
                                       │  • Probing        │
                                       │  • Compositionality│
                                       └──────────────────┘
```

---

## Current Blocker

### GPU/HPC Access Required for Phase 11.2

The transformer model (25.6M params) is too large for the local AMD RX 5700 GPU. Need CUDA-capable GPU (cloud or HPC) to:

1. **Train the transformer** on 27 real bout spectrograms (~103K frames)
2. **Extract hidden states** from trained model
3. **Train VQ-VAE** on real hidden states
4. **Run full analysis** on real discrete codes

**Estimated compute:** 2–4 weeks of active training + analysis

**Options:**
- Google Colab Pro ($12/mo, A100 GPUs)
- University HPC cluster
- Lambda Cloud / AWS spot instances

---

## What's Next (Priority Order)

| Priority | Task | Depends On | Est. Duration |
|----------|------|-----------|---------------|
| 1 | **Complete DeepSqueak MATLAB clustering** | Manual MATLAB step | 1–2 days |
| 2 | **Secure GPU/HPC access** | PI approval | — |
| 3 | **Train transformer on real data** (Phase 11.2) | GPU access | 1–2 weeks |
| 4 | **VQ-VAE + full analysis** (Phase 11.3–11.4) | Phase 11.2 | 1 week |
| 5 | **Cross-population comparison** (Phase 12) | Phase 11.4 | 2 weeks |
| 6 | **Batch detection on full corpus** (Phase 13) | Phase 11.4 | 2 weeks |

---

## Test Coverage

**600+ automated tests** across the full pipeline:

| Module | Tests |
|--------|-------|
| Energy detector | 42 |
| Dataset splits & quality | 71 |
| CNN classifier | 38 |
| Bout data pipeline | 56 |
| Transformer + VQ-VAE | 32 |
| Analysis tools | ~90 |
| Classification bridge | 33 |
| Active learning | 34 |
| Dataset assembler | 10 |
| Probing + acoustic properties | 38 |
| Information theory + null models | 30 |
| LMT data access | 23 |
| Repertoire statistics | 33 |
| Other (spectrogram, config, etc.) | ~70 |

---

## Key Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| 300 kHz sample rate | Nyquist for USVs up to 120 kHz (ADR-001) |
| Two-stage detection (energy → CNN) | High recall first stage, precision from CNN |
| Train transformer BEFORE VQ-VAE | Let model freely learn representations, then discretize |
| K=64 codebook | Headroom beyond ~10–15 traditional syllable types |
| Recording-based train/test splits | Prevents data leakage from same recording |
| Pre-norm transformer architecture | Better training stability for small datasets |
| Bout-level (not call-level) input | Preserves temporal context between USVs |

---

## Gantt Chart

See `docs/timeline.png` (generated by `scripts/generate_project_timeline.py`).

```
Run: .venv/Scripts/python.exe scripts/generate_project_timeline.py
```

---

*Generated: Feb 28, 2026*
