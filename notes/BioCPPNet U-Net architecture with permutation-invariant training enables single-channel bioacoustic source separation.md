---
description: "Raw waveform passes through encoder-masking-decoder pipeline with PIT criterion — tested on macaques, dolphins, bats, achieving ~10 dB SI-SDR for 2-3 concurrent vocalizers"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[detection]]"
  - "[[signal-processing]]"
---

# BioCPPNet U-Net architecture with permutation-invariant training enables single-channel bioacoustic source separation

BioCPPNet (Earth Species Project, Scientific Reports 2021) introduced the first neural network approach to single-channel bioacoustic source separation. The pipeline processes raw waveforms through an encoder (STFT or learned Conv1D), applies a 2D U-Net to predict per-source masks, multiplies masks with the mixture representation, and applies an inverse transform to recover separated waveforms. Training uses permutation-invariant training (PIT) which solves the label assignment problem — the loss considers all possible source-to-prediction permutations and uses the minimum. The architecture handles 2-3 concurrent vocalizers, with performance degrading as N increases. Notably, handcrafted STFT encoders outperformed learned filterbanks, suggesting that domain knowledge about time-frequency structure still matters. Open source at github.com/earthspecies/cocktail-party-problem.

---

Source:
- overlapping-usv-source-separation-state-of-art-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[no published single-channel USV source separation method exists as of 2026]] — BioCPPNet could be adapted for USVs

Topics:
- [[detection]]
- [[signal-processing]]
