---
description: "Investigates whether discrete acoustic token sequences from VQ and GVQ of HuBERT representations capture temporal structure in animal calls"
source_type: paper
url: "https://arxiv.org/abs/2511.10190"
author: "Eklavya Sarkar, Mathew Magimai-Doss"
date_accessed: "2026-02-27"
status: unprocessed
---

# Towards Leveraging Sequential Structure in Animal Vocalizations (Sarkar & Magimai-Doss, 2025)

Paper from AI for Non-Human Animal Communication workshop. Idiap Research Institute / EPFL.
Funded by Swiss National Science Foundation NCCR Evolving Language (grant 51NF40_180888).

## Core Problem

Existing computational bioacoustics typically averages extracted frame-level features across the temporal axis, discarding the order of sub-units within vocalizations. This throws away potentially meaningful sequential information. The question: can discrete token sequences preserve this temporal structure?

## Methodology

1. Extract frame-level embeddings from raw audio using HuBERT (self-supervised speech model pre-trained on LibriSpeech)
2. Discretize embeddings into token sequences using two methods:
   - **VQ (Vector Quantization)**: k-means clustering of frame embeddings, each frame mapped to nearest centroid index
   - **GVQ (Gumbel-Softmax Vector Quantization)**: differentiable relaxation of argmax, allows gradient-based training
3. Codebook adapted to distribution of bioacoustic embeddings — yields a discrete vocabulary capturing statistical structure
4. Measure pairwise distances between token sequences using Levenshtein (edit) distance
5. Classify using k-NN with Levenshtein distance metric

## Datasets

- **InfantMarmosetsVox (IMV)**: Marmoset call-types, infant vocalizations
- **Bosshard**: Marmoset dataset
- **Wierucka**: Marmoset dataset
- **Abzaliev**: 8,034 dog vocalizations across 14 call-types

Four distance categories tested: same-caller same-call-type, same-caller different-call-type, different-caller same-call-type, different-caller different-call-type.

## Key Results

### Distance Analysis
VQ tokens demonstrated expected hierarchical distance patterns:
- Smallest distances: same-caller, same-call-type vocalizations
- Largest distances: different-caller, different-call-type
- Critical finding: "Two vocalizations produced by a caller vocalizing different call-types are more likely to be acoustically distinct" than those from different callers producing identical calls
- This means caller identity is embedded MORE strongly than call-type identity in the acoustic signal

### Classification Performance
- Call-type identification (CTID): VQ showed 26-39% performance drop vs. linear baseline
- Caller identification (CLID): VQ showed 15-71% performance drop
- GVQ underperformed significantly — suggests codebook collapse problem
- Linear baseline consistently outperformed token-based approaches across all scenarios

### Codebook Collapse
VQ approaches suffered from codebook collapse: codebook usage is highly imbalanced, with most input embeddings mapped to one or two centroids while the rest remain idle. This drastically reduces effective representation capacity. GVQ was worse than VQ despite theoretical advantages.

## Key Tensions and Open Questions

1. Token sequences capture sufficient information for distance-based discrimination but fail at classification — suggests the Levenshtein + k-NN pipeline loses information that a more sophisticated sequence model could preserve
2. Single-codebook quantization proved insufficient for preserving caller-specific nuances — multi-codebook architectures (like residual VQ) may be necessary
3. Caller identity is encoded more strongly than call-type in the acoustic signal — this has implications for any classification system that tries to separate the two

## Recommended Future Directions (from paper)
- Multi-codebook architectures (residual vector quantization)
- Sequence post-processing: deduplication or acoustic byte-pair encoding
- More sophisticated sequence modeling beyond k-NN
- Cross-species evaluation of token-based representations

## Raw Observations

This paper is directly relevant to our VQ-VAE hidden state work. The codebook collapse finding validates our concern about effective codebook utilization. The hierarchy (caller > call-type in acoustic distinctiveness) could inform our probing experiment design — we should test whether our VQ-VAE codes similarly preserve caller identity more than syllable type.

The Levenshtein distance approach is interesting as a complement to our information-theoretic measures (entropy, mutual information). It captures sequence-level similarity without requiring a statistical model of the token distribution.

The failure of GVQ despite theoretical advantages (differentiable, end-to-end trainable) vs. plain VQ (non-differentiable but robust) mirrors a broader pattern in representation learning: simpler discretization methods often outperform more sophisticated ones.
