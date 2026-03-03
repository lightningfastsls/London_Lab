---
description: "K-means discretization of mel-spectrograms into 16K tokens with skip-gram embeddings was computationally cheap but low accuracy"
type: finding
confidence: proven
conditions:
  - "BirdCLEF+ 2025 benchmark"
  - "16384 discrete tokens via K-means"
  - "bird, insect, amphibian, mammal sounds"
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# STSG spectrogram token skip-gram achieved only 0.559 AUC versus 0.810 for transfer learning on bioacoustic classification

The Spectrogram Token Skip-Gram (STSG) approach (BirdCLEF+ 2025) represents a simple discrete token approach to bioacoustics: K-means clustering of mel-spectrogram frames into 16,384 discrete tokens, followed by skip-gram word2vec-style embeddings learned on token sequences. Tested on bird, insect, amphibian, and mammal sounds.

Results: ROC-AUC of 0.559 versus 0.810 for a transfer learning baseline -- a dramatic underperformance of 25 percentage points. The approach was computationally efficient but sacrificed too much acoustic information through the fixed K-means discretization step.

This negative result, alongside [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]], strengthens the case that discrete token approaches in bioacoustics need end-to-end training rather than fixed clustering. K-means tokens (no learned codebook, no gradient flow) lose even more information than post-hoc VQ on learned features. The progression from worst to best is:

1. K-means tokens (STSG) -- fixed clustering, no learning -> 0.559 AUC
2. Post-hoc VQ on HuBERT (Sarkar) -- learned features but frozen during VQ -> 35% UAR vs 49% baseline
3. End-to-end VQ-VAE (our approach) -- joint optimization of encoder, codebook, decoder -> untested

The skip-gram embedding step is interesting independently as a way to discover sequential structure, similar to our [[bigram productivity ratio measures compositionality of USV code sequences]] analysis.

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- STSG (2025), BirdCLEF+. https://arxiv.org/abs/2507.08236

Relevant Notes:
- [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] -- another discrete approach that underperformed
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- K-means used for USV type discovery
- [[bigram productivity ratio measures compositionality of USV code sequences]] -- skip-gram embeddings relate to sequential structure analysis

Topics:
- [[unsupervised-usv-discovery]]
- [[classification]]
