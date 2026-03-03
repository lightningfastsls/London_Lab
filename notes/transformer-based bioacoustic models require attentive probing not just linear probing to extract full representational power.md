---
description: "Foundation Models comparative review found linear probes underestimate transformer capacity — attentive probing unlocks additional task-relevant features frozen in intermediate layers"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# Transformer-based bioacoustic models require attentive probing not just linear probing to extract full representational power

A comprehensive review of foundation models for bioacoustics (arXiv 2508.01277) found that transformer-based models require attentive probing — not just linear probing — to extract their full representational power. BirdMAE, a masked autoencoder trained on large-scale birdsong data, achieved the best performance on the BirdSet benchmark, but only when properly probed.

Linear probing (training only a single linear layer on frozen embeddings) systematically underestimates transformer capacity because the features most relevant for downstream tasks may be distributed across layers and entangled in ways that a linear transformation cannot disentangle. Attentive probing adds a small attention mechanism that can selectively combine information from different positions and layers.

This is relevant for our VQ-VAE work: since [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]], the choice of which layer to quantize AND how to probe it both matter. Using frozen embeddings with only linear probes may give an unfairly pessimistic view of what the model has learned.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- probing method matters alongside layer choice
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- attentive probing could improve layer comparison

Topics:
- [[bioacoustic-ssl]]
