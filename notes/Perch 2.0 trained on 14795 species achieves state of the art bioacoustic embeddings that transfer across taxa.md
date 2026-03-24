---
description: "EfficientNet-B3 backbone (12M params) trained on 1.5M+ recordings; AUROC 0.908 on BirdSet; outperforms specialized marine models on marine tasks despite minimal marine training data"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# Perch 2.0 trained on 14795 species achieves state of the art bioacoustic embeddings that transfer across taxa

Perch 2.0 (Google, 2025) represents the current state of the art for general bioacoustic embeddings. Its EfficientNet-B3 backbone (12M parameters) was trained on over 1.5 million labeled recordings covering 14,795 species across birds, amphibians, insects, and mammals. It achieves AUROC 0.908 on BirdSet and 0.840 classification accuracy on BEANS.

The most remarkable finding is cross-taxa transfer: Perch 2.0 outperforms specialized marine models on marine transfer tasks despite having almost no marine training data. This suggests its embeddings capture fundamental acoustic structure that transfers across taxonomic groups — pitch contours, temporal modulations, harmonic structure, and amplitude envelopes appear to share universal properties across vocal production systems.

For our USV pipeline, Perch 2.0 is relevant as a potential off-the-shelf embedding extractor for clustering. However, mouse USVs at 20-120 kHz may fall outside the frequency range Perch was trained on (primarily audible-range bird and frog calls). The 12M parameter model is also practical for embedding extraction, unlike larger language-model-based approaches. Whether Perch embeddings capture the fine-grained within-species variation needed for USV syllable clustering (rather than just species-level separation) remains an open question.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[supervised bioacoustic foundation models vastly outperform self-supervised for species-level clustering]] -- Perch is the top-performing model in this comparison
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] -- Perch embeddings feed into this pipeline
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- could efficiently adapt Perch's 12M-param EfficientNet-B3 to USV-specific classification when frozen embeddings are insufficient

Topics:
- [[bioacoustic-ssl]]
