---
description: "Web research synthesis on VQ-VAE and discrete representation learning in animal bioacoustics as of February 2026"
source_type: article
url: "multiple — see sources below"
author: "Claude /learn synthesis"
date_accessed: "2026-02-19"
research_tool: "WebSearch + WebFetch"
research_query: "VQ-VAE vector quantized variational autoencoder animal bioacoustics vocalization 2024 2025"
status: processed
---

# State of VQ-VAE and Discrete Representation Learning in Animal Bioacoustics (Feb 2026)

## Executive Summary

As of February 2026, VQ-VAE has still NOT been directly applied to animal vocalizations in published work. The vault's existing claim in [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] remains valid, though the gap is narrowing from adjacent approaches. The field is converging on discrete representations from multiple directions — self-supervised speech models with post-hoc quantization, convolutional autoencoders for clustering, and VAEs for latent space analysis — but none have combined VQ-VAE's end-to-end learned discrete codebook with animal vocalization data.

## Key Developments by Category

### 1. Direct Vector Quantization on Animal Vocalizations (Closest to VQ-VAE)

**Sarkar & Magimai-Doss (NeurIPS 2025 Workshop)** — "Towards Leveraging Sequential Structure in Animal Vocalizations"
- First published work applying discrete audio tokens to bioacoustics
- Used post-hoc VQ and Gumbel-softmax VQ on frozen HuBERT embeddings (NOT end-to-end VQ-VAE training)
- Vocabulary size V=50, tested on marmosets (3 datasets, up to 72K samples) and dogs (8K samples)
- Results: VQ tokens discriminated call-types and callers but substantially underperformed linear probing baselines (e.g., 35% vs 49% UAR on Bosshard marmoset dataset)
- Gumbel-softmax VQ suffered severe codebook collapse
- Key limitation: single codebook insufficient for complex vocalization structure
- Significance: demonstrates the concept works but is far from competitive with continuous representations
- URL: https://arxiv.org/abs/2511.10190

### 2. VAE-Based Latent Space Analysis (Continuous, Not Discrete)

**Goffinet et al. (eLife 2021)** — AVA tool (already in vault)
- VAE trained on mouse USVs (31,440 syllables) learned continuous latent representations
- Key finding: USVs form a continuum, not discrete clusters
- 64-95% of traditional feature information captured in latent space
- This is the strongest motivation for VQ-VAE: if the space is continuous but we want discrete units, VQ-VAE can find principled discretization points

**Garrobé Fonollosa et al. (arXiv Oct 2024)** — "Temporal Feature Learning in Weakly Labelled Bioacoustic Cetacean Datasets"
- VAE + Temporal Convolutional Network for sperm whale click classification
- VAE used for unsupervised feature extraction from 4-minute recordings
- Achieved AUC >0.9, outperforming handcrafted features
- Notable: VAE used for feature learning only, no discretization
- URL: https://arxiv.org/abs/2410.17006

### 3. Convolutional Autoencoders for Vocalization Clustering (No Quantization)

**Best, Paris, Glotin & Marxer (PLOS ONE 2023)** — "Deep Audio Embeddings for Vocalisation Clustering"
- Convolutional autoencoder (NOT variational, NOT VQ) with perceptual loss
- Tested on 8 datasets across 6 species (Bengalese finch, Cassin vireo, California thrasher, black-headed grosbeak, humpback whale, bottlenose dolphin)
- NMI scores 0.5-0.75 across datasets
- Generic (cross-species) autoencoders matched species-specific ones
- Published Python package for bioacoustics community
- Significance: shows learned representations beat handcrafted features for repertoire discovery, but does NOT use discrete codebooks
- URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283396

### 4. Self-Supervised Speech Models Transferred to Bioacoustics

**AVES (Hagiwara, ICASSP 2023)** — Animal Vocalization Encoder based on Self-Supervision
- HuBERT architecture pretrained on FSD50K, AudioSet, VGGSound
- Outperformed supervised baselines on bioacoustic classification/detection
- Continuous representations, no discrete codebook
- URL: https://arxiv.org/abs/2210.14493

**Sarkar & Magimai-Doss (ICASSP 2025)** — Comparing SSL Models for Bioacoustics
- Speech-pretrained models (HuBERT etc.) matched animal-pretrained ones
- Pre-training on bioacoustic data provides only marginal improvements
- Implication: human speech SSL models transfer well to animal vocalizations
- URL: https://arxiv.org/abs/2501.05987

### 5. Discrete Token Approaches in Broader Audio (Not Bioacoustics)

**FSQ (Mentzer et al., ICLR 2024)** — Finite Scalar Quantization
- Eliminates codebook learning entirely; fixed scalar quantization per dimension
- 100% codebook utilization by construction (no collapse)
- Competitive with VQ-VAE at simpler implementation
- Applied to speech codecs at 400-700 bps
- NOT applied to bioacoustics yet
- URL: https://proceedings.iclr.cc/paper_files/paper/2024/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf

**"Discrete Audio Tokens: More Than a Survey!" (2025)**
- Comprehensive taxonomy: K-means, RVQ, SVQ, GVQ, FSQ, MSRVQ, CSRVQ, PQ
- Covers speech, music, general audio — NO bioacoustics coverage
- Identifies codebook collapse as critical challenge
- URL: https://arxiv.org/abs/2506.10274

**STSG (BirdCLEF+ 2025)** — Spectrogram Token Skip-Gram
- K-means clustering of Mel-spectrograms into 16,384 discrete tokens
- Skip-gram embeddings learned on token sequences
- Tested on bird/insect/amphibian/mammal sounds
- ROC-AUC 0.559 vs 0.810 for transfer learning baseline — significantly worse
- Computationally efficient but low accuracy
- URL: https://arxiv.org/abs/2507.08236

## The Gap Analysis: Why VQ-VAE Remains Novel

| Approach | Discrete? | End-to-end? | Learned codebook? | Applied to animal vocalizations? |
|----------|-----------|-------------|-------------------|----------------------------------|
| VQ-VAE (our planned approach) | Yes | Yes | Yes | NO (gap) |
| Post-hoc VQ on HuBERT (Sarkar 2025) | Yes | No | Partially | Yes (marmosets, dogs) |
| K-means tokens (STSG 2025) | Yes | No | No (fixed clustering) | Yes (birds, insects) |
| Convolutional AE (Best 2023) | No | Yes | N/A | Yes (6 species) |
| VAE/AVA (Goffinet 2021) | No | Yes | N/A | Yes (mice, finches) |
| AVES (Hagiwara 2023) | No | Yes | N/A | Yes (multiple species) |
| FSQ (Mentzer 2024) | Yes | Yes | No (fixed grid) | No |

The unique combination is: **end-to-end trained discrete codebook** applied to **animal vocalizations**. Nobody has published this yet.

## Implications for Our VQ-VAE Pipeline

1. **Novelty confirmed** — The research gap is real and still open
2. **Post-hoc VQ underperforms** — Sarkar 2025 shows that slapping VQ on frozen representations loses information (35% vs 49% UAR). End-to-end VQ-VAE training should learn better discrete representations.
3. **Single codebook may be insufficient** — Sarkar found V=50 wasn't enough. Our K=64 is similar. May need to consider RVQ (residual) or larger K.
4. **FSQ deserves serious consideration** — Eliminates codebook collapse by design, simpler than VQ-VAE, competitive performance. The vault already has [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] as an open question.
5. **Gumbel-softmax VQ collapses** — Sarkar's negative result with GVQ validates our choice of standard VQ-VAE approach
6. **Speech SSL models transfer well** — If we hit issues with training data volume, fine-tuning AVES/HuBERT + VQ-VAE could bootstrap representations
7. **The continuum finding stands** — No new work contradicts Goffinet 2021's USV continuum. VQ-VAE discretization remains scientifically motivated.

## Processing Notes
Extract at minimum:
- Update to the gap claim note with 2025 evidence (Sarkar narrowed but didn't close it)
- New note on Sarkar 2025 post-hoc VQ results
- New note on Best 2023 convolutional AE for repertoire clustering
- New note on AVES as potential backbone for VQ-VAE
- New note on FSQ as alternative discretization (with 2024 ICLR results)
- New note on STSG discrete token approach and its limitations
- New note on speech-to-bioacoustics transfer learning findings
- Update the gap analysis table as a synthesis note
