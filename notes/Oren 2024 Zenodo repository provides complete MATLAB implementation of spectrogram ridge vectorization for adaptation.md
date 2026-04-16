---
description: "Zenodo record 12721811 contains spectrogramming.m (core vectorization), RF_Generic.m (classification), and all figure data under CC-BY 4.0"
type: finding
confidence: proven
conditions: []
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[classification-tools]]"
---

# Oren 2024 Zenodo repository provides complete MATLAB implementation of spectrogram ridge vectorization for adaptation

The Zenodo repository for Oren et al. 2024 (DOI: 10.5281/zenodo.12721811) provides the complete MATLAB implementation of the 80D vectorization pipeline. Key files:

- **`spectrogramming.m`** — Core vectorization: STFT -> ridge extraction -> 80D vector. This is the implementation of [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]].
- **`calc_acoustic_features.m`** — The 16 named acoustic features (8 FM + 8 AM) used for explained-variance analysis.
- **`RF_Generic.m`** — Random forest training + OOB evaluation with per-caller normalization.
- **`RUS.m`** — Random Under-Sampling for class balancing.
- **`calc_proximity.m`** — RF leaf co-occurrence proximity.
- **`Fig_1.mat` through `Fig_4.mat`** — Figure data (Fig_4.mat is 125.7 MB, contains proximity data).

The repository is licensed CC-BY 4.0, 130 MB total, all MATLAB code. This establishes full reproducibility and provides a concrete implementation reference for Python adaptation of the vectorization technique.

The code availability contrasts with [[DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port|DeepSqueak's MATLAB lock-in]] — both are MATLAB, but the Oren code is small enough to port to Python directly.

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Zenodo: https://zenodo.org/records/12721811

Relevant Notes:
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- the method these files implement
- [[DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port]] -- another MATLAB-locked tool, but much heavier

Topics:
- [[classification-tools]]
