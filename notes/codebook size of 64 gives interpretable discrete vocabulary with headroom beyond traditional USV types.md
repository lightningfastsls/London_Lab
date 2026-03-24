---
description: "Reduced from K=512 (v1, too large for analysis) to K=64 — 4-6x the ~10-15 traditional syllable types, enabling discovery of sub-types and transition states"
type: decision
confidence: experimental
conditions: []
meta_state: current
topics:
  - "[[representation-learning]]"
---

# Codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types

The VQ-VAE codebook size is set to K=64, chosen as a balance between interpretability and expressiveness. Traditional mouse USV taxonomy identifies approximately 10-15 syllable types (flat, chevron, frequency-modulated, complex, etc.). K=64 provides substantial headroom for discovering sub-types, transition states, or contextual variants that the traditional taxonomy may not capture. Starting with K=64 allows exploration of larger (128, 256) and smaller (32) codebook sizes to find the optimal granularity. The codebook was reduced from K=512 in the v1 architecture, which proved too large for interpretable analysis. Each codebook entry should ideally correspond to an interpretable "concept" that can be mapped back to acoustic features.

### ROADMAP Context

ROADMAP specifies the full VQ-VAE encoder and decoder architectures. Encoder: Conv1d(512→256, kernel_size=5, padding=2) → GELU → Linear(256→64) → L2-norm. Decoder: Linear(64→256) → GELU → Linear(256→512). Key hyperparameters: commitment_weight=0.25, ema_decay=0.99, dead_code_threshold=2.0, use_conv_encoder=True, conv_kernel_size=5. The conv encoder smooths temporal neighbors before projection, giving each code a small receptive field (~5 frames ≈ 2.1 ms at 300 kHz). Codebook visualization decodes entries through the full pipeline back to spectrogram space — see [[VQ-VAE codebook visualization decodes entries through the full pipeline back to spectrogram space]].

---

Source:
- DECISIONS.md (ADR-007) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- the architecture this codebook serves
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- preventing degenerate codebooks
- [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] -- alternative approach
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- empirical precedent for data-driven type discovery (MUPET finds 100-140 types, placing K=64 in a comparable order of magnitude)
- [[Tjandra et al 2020 applied transformer VQ-VAE for unsupervised unit discovery in human speech with K equals 128]] -- closest architectural analog uses K=128 for human speech
- [[single codebook with V=50 was insufficient for complex vocalization structure in discrete token experiments]] -- Sarkar 2025 found V=50 insufficient for marmoset calls, suggesting K=64 may also need RVQ or larger K
- [[discrete audio token taxonomy from 2025 survey covers quantization methods beyond simple VQ]] -- RVQ and PQ as alternatives if single codebook proves limiting

Topics:
- [[representation-learning]]
