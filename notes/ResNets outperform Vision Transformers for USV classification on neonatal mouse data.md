---
description: "2024 systematic comparison of FC, CNN, ResNet, EfficientNet, ViT on neonatal USVs — adapted ResNets achieved 86.79% accuracy; ViT did NOT outperform convolutional architectures"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[classification]]"
---

# ResNets outperform Vision Transformers for USV classification on neonatal mouse data

A 2024 study in JASA systematically compared five architectures for neonatal mouse USV classification: fully-connected networks, CNNs, ResNets, EfficientNet, and Vision Transformers (ViT). Adapted ResNets achieved the best classification accuracy at 86.79%, while ViT did NOT outperform convolutional approaches.

This finding is significant for two reasons. First, it validates our CNN-based classifier architecture — since [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]], convolutional approaches remain competitive even against transformer architectures. Second, it suggests that the success of transformers in NLP and general vision does not automatically transfer to the small, structured spectrogram patches typical of USV classification. USV spectrograms are relatively simple visual patterns (frequency sweeps, harmonics) where local convolutional features may capture the relevant structure more efficiently than global self-attention.

The same study used entropy-based detection achieving 94.9% recall and 99.3% precision, suggesting signal-processing-based detection with ResNet classification as a strong pipeline alternative.

---

Source: usv-detection-methods-landscape-2024-2026-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- our CNN architecture is validated by this comparison
- [[transformer-based bioacoustic models require attentive probing not just linear probing to extract full representational power]] -- ViT underperformance may be partially due to probing/evaluation method

Topics:
- [[classification-methodology]]
