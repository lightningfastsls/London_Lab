---
description: Training identical VQ-VAE models on hidden states from layers 2, 4, 6, and 8 separately identifies which abstraction level best supports an interpretable discrete vocabulary.
type: method
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook

Train an identical VQ-VAE (same architecture, same hyperparameters, same training procedure) on the hidden states extracted from each of layers 2, 4, 6, and 8 of the trained transformer. The layer is the only variable. This controlled comparison isolates the effect of abstraction level on codebook quality, avoiding confounds from different VQ-VAE capacity or training dynamics.

Three metrics drive the comparison. Codebook perplexity (interpretability proxy) measures how uniformly the codebook is utilized — perplexity close to K means all codes are used equally, suggesting rich diversity; target is perplexity > 0.5×K. Codebook utilization (stability proxy) measures what fraction of codebook entries are used at all; target > 90% indicates no dead codes. Reconstruction loss measures how much information the layer's hidden states carry about the original spectrogram — lower is better for preserving acoustic content. Critically, the comparison emphasizes perplexity over reconstruction loss, because the goal is an interpretable discrete vocabulary, not maximal reconstruction fidelity.

The default extraction point is layer 4 (middle of the 8-block model) per [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]], but this experiment validates or overrides that default with empirical evidence. If VQ-VAE codebook collapse persists across all layers despite prevention mechanisms, [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] becomes the fallback -- the same layer comparison could be repeated with FSQ to determine if the collapse is layer-dependent or mechanism-dependent. Early layers (layer 2) likely encode low-level frequency patterns; late layers (layer 8) likely encode high-level predictive context; middle layers balance both. The chosen layer's codebook then feeds downstream analysis including [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]], where the choice of K must match the actual diversity of concepts the layer encodes.

Recent findings on codebook stability are directly relevant to this comparison: [[Gumbel-softmax VQ suffered severe codebook collapse in bioacoustic token experiments]], validating that standard VQ-VAE with explicit anti-collapse mechanisms is the right choice for this layer comparison. Additionally, [[FSQ eliminates codebook collapse by construction achieving 100 percent utilization through fixed scalar quantization]] offers a codebook-collapse-free alternative — the layer comparison could be repeated with FSQ to determine if collapse patterns are layer-dependent. The broader landscape of quantization methods is surveyed in [[discrete audio token taxonomy from 2025 survey covers quantization methods beyond simple VQ]].

---

Source: [ROADMAP](../ROADMAP.md), Phase 8
