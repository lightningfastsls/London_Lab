---
description: "Pure attention treats all positions equally regardless of distance — effective for language but insufficient for audio/vision where local patterns dominate, driving Conformer and CoAtNet designs"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# self-attention lacks inductive bias for local structure leading to hybrid architectures for domains where locality matters

Self-attention is permutation-equivariant by default — it treats all positions equally regardless of distance. While positional encoding partially addresses this, attention still lacks the strong locality bias that convolutions provide. A convolutional kernel inherently prioritizes nearby elements, which is a powerful inductive bias for domains where local patterns are fundamental (pixel neighborhoods in images, short-timescale acoustic features in audio).

This gap has led to hybrid architectures that combine attention's global reach with convolution's local bias. Conformer (Gulati et al., 2020) interleaves attention and convolution modules for speech/audio processing. CoAtNet (Dai et al., 2021) does the same for vision. These hybrids consistently outperform pure-attention models in domains where local structure is important.

For NLP, self-attention-only architectures have been highly successful because language understanding requires long-range dependency modeling from the start — a pronoun might reference a noun from many sentences prior. However, even modern NLP models increasingly incorporate local attention windows. Gemma 3, for example, uses a 5:1 ratio of local-to-global attention layers, suggesting that pure global attention may be suboptimal even for language.

This is directly relevant to the USV project: since [[ResNets outperform Vision Transformers for USV classification on neonatal mouse data]], the locality bias of convolutions clearly helps for spectrogram classification. The autoregressive transformer in our pipeline uses global attention because it needs to capture bout-level temporal dependencies, but a hybrid approach might improve performance for the classification task.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[ResNets outperform Vision Transformers for USV classification on neonatal mouse data]] -- empirical evidence for locality bias advantage in USV spectrograms
- [[self-attention provides O(1)-path global context from layer 1 while CNNs require many stacked layers to aggregate distant information]] -- the complementary advantage attention offers

Topics:
- [[transformer-architecture]]
