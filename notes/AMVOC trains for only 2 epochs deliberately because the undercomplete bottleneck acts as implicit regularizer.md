---
description: "loss plateaus at ~0.095 after 2 epochs from 0.125 — the 8x compression bottleneck constrains capacity enough that overfitting is unlikely even without dropout or early stopping"
type: decision
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
---

# AMVOC trains for only 2 epochs deliberately because the undercomplete bottleneck acts as implicit regularizer

AMVOC's autoencoder (Stoumpou et al. 2022) trains for only 2 epochs on 22,409 syllables (batch size 32, ~1,400 iterations total). The BCE loss drops rapidly from ~0.125 to ~0.095 in those 2 epochs, then shows minimal improvement. The authors deliberately chose this short training regimen — not because of computational constraints (training runs on a single Colab GPU in minutes) but as a design decision.

The reasoning is that the undercomplete bottleneck (8× compression, from 10,240 input values to 1,280) is itself a powerful regularizer. Because the network must compress the input through a narrow information bottleneck, it is forced to learn only the most salient features even in a single pass through the data. Dropout, batch normalization, and early stopping become less critical when the bottleneck already constrains what the network can memorize.

This reframes how to think about autoencoder training depth: the reconstruction is *intentionally lossy*. The goal is not perfect pixel reconstruction but extraction of features that discriminate between USV types. A "perfect" autoencoder that memorizes input spectrograms would actually be worse for downstream clustering because it would encode noise alongside signal.

However, this rationale has limits. With more variable data (such as our wild-mouse recordings, where since [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] suggests greater within-species variability), the autoencoder may need more capacity and thus more training to capture the broader feature space. The paper's 2-epoch finding may not transfer directly to our pipeline.

Training details: Adam optimizer, LR 0.001, no data augmentation, shuffle=True. The dataset was 80/20 train/test split from 26 recordings across 9 male mice (B6D2F1/J and C57BL/6J lab strains).

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression]] — the architecture whose bottleneck provides the regularization
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] — parallel finding that simpler architectures work for small USV datasets
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] — wild-mouse variability may require longer training
- [[spectrogram SpecAugment-style augmentation with frequency and time masking improves transformer generalization]] — AMVOC needs no augmentation because the bottleneck is regularizer enough; our deeper transformer architecture requires explicit augmentation (SpecAugment) to prevent overfitting, illustrating how regularization needs scale with model capacity

Topics:
- [[unsupervised-usv-discovery]]
