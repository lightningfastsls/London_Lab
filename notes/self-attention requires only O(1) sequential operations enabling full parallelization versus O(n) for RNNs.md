---
description: "All pairwise interactions computed simultaneously, making transformer training dramatically faster than recurrent models — one of three key motivations in Vaswani et al 2017"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# self-attention requires only O(1) sequential operations enabling full parallelization versus O(n) for RNNs

Self-attention computes all pairwise interactions simultaneously in a single matrix multiplication, requiring only O(1) sequential operations regardless of sequence length. RNNs, by contrast, require O(n) sequential steps — each hidden state depends on the previous one, creating an inherently serial computation that cannot be parallelized across timesteps.

This parallelization advantage was one of three key motivations in Vaswani et al. (2017), alongside reduced maximum path length and competitive per-layer complexity. The paper states the Transformer "only needs a constant number of sequential operations" — meaning the depth of the computation graph does not grow with sequence length.

The practical consequence was transformative for training at scale. An RNN processing a 1000-token sequence must execute 1000 serial steps; a transformer processes the same sequence in constant sequential depth (though with more total computation). On modern GPU hardware optimized for parallel matrix operations, this translates to dramatically faster wall-clock training times. This is why transformers enabled the scaling revolution — they could effectively utilize increasing GPU parallelism in a way that RNNs fundamentally could not.

However, since [[self-attention has O(n²d) time complexity while recurrence has O(nd²) making attention faster when sequence length is shorter than model dimension]], the parallelization advantage comes with a quadratic cost that becomes the bottleneck for very long sequences, driving research into efficient attention variants.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[self-attention provides O(1)-path global context from layer 1 while CNNs require many stacked layers to aggregate distant information]] -- the global context this parallelization enables
- [[self-attention has O(n²d) time complexity while recurrence has O(nd²) making attention faster when sequence length is shorter than model dimension]] -- the cost tradeoff

Topics:
- [[transformer-architecture]]
