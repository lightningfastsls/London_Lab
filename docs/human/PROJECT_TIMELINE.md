# USV Detection & Analysis — Project Timeline

> **Prepared for:** Lab PI meeting (week of 2026-03-02)
> **Last updated:** 2026-02-28
> **Chart:** See `timeline.png` / `timeline.pdf` in project root

---

## Executive Summary

End-to-end pipeline for analyzing mouse ultrasonic vocalizations (USVs) in 300 kHz recordings. The project asks: **do mouse USV sequences contain language-like structure?**

**Current status:** 15/23 phases complete (65%). Core infrastructure is built and validated. The main blocker is GPU access for training the transformer model on real data.

### Key Metrics

| Metric | Value |
|--------|-------|
| Phases complete | **15/23** (65%) |
| Test suite | **434+ tests** passing |
| CNN classifier | F1=91.7% (P=89.7%, R=93.8%) |
| Training labels | ~840 human-reviewed |
| Transformer model | ~25.6M params (built, not yet trained on real data) |
| VQ-VAE codebook | K=64 discrete codes (built, awaiting real hidden states) |
| Detected USVs | 93 detections across 5 recordings |

---

## Timeline Overview

```
2025-Oct ──── 2025-Nov ──── 2025-Dec ──── 2026-Jan ──── 2026-Feb ──── 2026-Mar ──── 2026-Apr+
    │              │              │              │              │              │
    ├─ Phase 1 ────┤              │              │              │              │
    │  Energy       ├─ Phase 3 ───┤              │              │              │
    │  Detector     │  Labeling    ├─ Phase 4 ───┤              │              │
    │  (42 tests)   │  Tool        │  Dataset     ├─ Phase 5 ───┤              │
    │               │  (~840       │  Prep        │  CNN         ├─ Phase 6 ───┤
    ├─ Phase 2 ────┤│  labels)     │  (71 tests)  │  Classifier  │  Desktop    │
    │  Spectrograms ┤              │              │  (F1=91.7%)  │  App (PyQt6)│
    │               │              │              │  (38 tests)  │             │
    │               │              │              │              ├─ Phase 7 ───┤
    │               │              │              │              │  Clustering  │
    │               │              │              │              │              │
    │               │              │              │              ├─ Phase 8 ──────────┤
    │               │              │              │              │  8.1 Bout prep      │
    │               │              │              │              │  8.2 Transformer    │
    │               │              │              │              │  8.3 VQ-VAE         │
    │               │              │              │              │  8.4 Analysis       │
    │               │              │              │              │                     │
    │               │              │              │              ├─ Phase 9-10 ────────┤
    │               │              │              │              │  Dataset assembly   │
    │               │              │              │              │  Active learning    │
    │               │              │              │              │                     │
    │               │              │              │              ├─ Phase 11.1 DONE    │
    │               │              │              │              │  Bout extraction    │
    │               │              │              │              │                     │
    │               │              │              │              ├─ Phase 14.1 DONE    │
    │               │              │              │              │  Raven export       │
    │               │              │              │              ├─ 14.2 ► IN PROGRESS │
    │               │              │              │              │  MATLAB clustering  │
                                                                │                     │
                                                         ██ TODAY (Feb 28) ██          │
                                                                │                     │
                                                                ├── BLOCKED ──────────┤
                                                                │  11.2 Transformer   │
                                                                │  training (GPU)     │
                                                                │  11.3 VQ-VAE real   │
                                                                │  11.4 Analysis real  │
                                                                │                     │
                                                                ├── FUTURE ───────────┤
                                                                │  12. Cross-pop      │
                                                                │  13. Batch detect   │
```

---

## Completed Work (17 phases)

### Core Pipeline (Phases 1-7) — ALL DONE

| Phase | What | Key Achievement |
|-------|------|-----------------|
| **1. Energy Detector** | Candidate detection at 300 kHz | 42 tests, high recall design |
| **2. Spectrogram Extraction** | PNG rendering from candidates | Matches detection STFT params |
| **3. Labeling Tool** | Streamlit UI for human review | ~840 labels (458 USV, 374 Not USV) |
| **4. Dataset Preparation** | Train/val/test splits | Recording-based (not candidate) to prevent leakage |
| **5. CNN Classifier** | Binary USV/noise classifier | **F1=91.7%** (P=89.7%, R=93.8%), ~101K params |
| **6. Desktop App** | PyQt6 detection + labeling GUI | Interactive thresholds, session tracking |
| **7. Clustering** | Unsupervised USV type discovery | k-means, HDBSCAN, GMM on CNN features |

### Transformer + VQ-VAE Architecture (Phase 8) — ALL DONE

| Phase | What | Key Achievement |
|-------|------|-----------------|
| **8.1** | Bout data preparation | PyTorch datasets with bucketed batching |
| **8.2** | Causal transformer | GPT-style, ~25.6M params, next-column prediction |
| **8.3** | Hidden-state VQ-VAE | K=64 codebook, EMA updates, anti-collapse defenses |
| **8.4** | Analysis & interpretation | 9 modules: Zipf, entropy, compositionality, context |

### Infrastructure & Real Data (Phases 9-11.1) — DONE

| Phase | What | Key Achievement |
|-------|------|-----------------|
| **9.1** | Dataset assembly pipeline | Automated training data generation |
| **10.1** | Active learning runner | Cycle automation for iterative improvement |
| **11.1** | Bout extraction (real data) | Real 300 kHz WAV processing confirmed |

### DeepSqueak Bridge (Phase 14.1) — DONE

| Phase | What | Key Achievement |
|-------|------|-----------------|
| **14.1** | Raven table export | 93 detections across 5 WAVs, 33 tests |

---

## Current Work

### Phase 14.2: DeepSqueak MATLAB Classification — IN PROGRESS

Importing our CNN-detected USVs into DeepSqueak (MATLAB) for syllable-level classification using its unsupervised clustering tools. This bridges our Python pipeline with the established USV classification ecosystem.

**Status:** DeepSqueak v3.1.0 installed. 5 Raven selection tables ready for import. Manual MATLAB steps in progress.

---

## Blocked Work (needs PI input)

### Phase 11.2-11.4: Real Data Training — BLOCKED ON GPU

The transformer model (~25.6M parameters) is too large for the available AMD RX 5700 GPU. Training requires an NVIDIA GPU with 8+ GB VRAM (e.g., HPC cluster or cloud GPU).

| Phase | What | Unblocks |
|-------|------|----------|
| **11.2** | Train transformer on real bout data | Everything downstream |
| **11.3** | Train VQ-VAE on real hidden states | Discrete code discovery |
| **11.4** | Run full analysis on real codes | **The key scientific result** |

**Action needed:** HPC cluster access or cloud GPU allocation (Colab Pro, Lambda, etc.)

### Phase 14.3-14.4: DeepSqueak Results — BLOCKED ON MATLAB STEP

Waiting for DeepSqueak classification output (Excel files) from the MATLAB step currently in progress.

---

## Future Work (not yet prioritized)

| Phase | What | Dependency |
|-------|------|------------|
| **12** | Cross-population USV comparison | Needs 11.4 + multiple population recordings |
| **13** | Batch detection pipeline | Needs stable detection params from 11.4 |
| **Vacation: Info Theory** | Rigorous Zipf/entropy + null models | Can start anytime (no deps) |
| **Vacation: Acoustic Probing** | Probe transformer hidden representations | Needs 11.2 |
| **Vacation: LMT Integration** | Link USVs to behavioral events | Needs LMT data from Prof. London |

---

## Architecture Overview

```
  300 kHz WAV recordings
         │
         ▼
  ┌─────────────────┐
  │  Energy Detector │ Phase 1  (high recall)
  │  (42 tests)      │
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  CNN Classifier  │ Phase 5  (F1=91.7%)
  │  ~101K params    │
  └────────┬────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
  ┌────────┐  ┌──────────────┐
  │DeepSqk │  │  Bout        │
  │Bridge  │  │  Extraction  │ Phase 8.1
  │(14.x)  │  └──────┬───────┘
  └────────┘         ▼
              ┌──────────────┐
              │  Transformer │ Phase 8.2  (~25.6M params)
              │  (causal)    │ ◄── TRAIN ON REAL DATA (blocked)
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │   VQ-VAE     │ Phase 8.3  (K=64)
              │  (codebook)  │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │   Analysis   │ Phase 8.4 + Vacation
              │  Zipf, H(X), │
              │  null models │
              └──────────────┘
                     ▼
              ┌──────────────┐
              │  Answer:     │
              │  Language-   │
              │  like USV    │
              │  structure?  │
              └──────────────┘
```

---

## Test Coverage Summary

| Area | Tests |
|------|-------|
| Energy detector | 42 |
| Dataset splits | 30 |
| Dataset quality | 41 |
| CNN model | 38 |
| Raven export | 33 |
| Transformer | 11 |
| VQ-VAE | 21 |
| Analysis tools | 17 |
| Dataset assembler | 10 |
| Other | 191+ |
| **Total** | **434+** |

---

*Generated 2026-02-28. Run `python scripts/generate_timeline.py` to regenerate the visual chart.*
