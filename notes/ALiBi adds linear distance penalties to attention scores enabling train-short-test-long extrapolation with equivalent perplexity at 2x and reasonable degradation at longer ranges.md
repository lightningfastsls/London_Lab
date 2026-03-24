---
description: "Press et al 2022 eliminates positional embeddings entirely — head-specific fixed slopes penalize distant tokens in attention scores, with BLOOM and MPT as key adopters"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# ALiBi adds linear distance penalties to attention scores enabling train-short-test-long extrapolation with equivalent perplexity at 2x and reasonable degradation at longer ranges

Attention with Linear Biases (ALiBi, Press et al., 2022) takes the most radical departure from prior positional encoding: it eliminates positional embeddings entirely. Instead, a position-dependent bias is added directly to the attention scores before softmax: score(i,j) = Q_i · K_j^T - m · |i-j|, where m is a head-specific slope and |i-j| is the distance between positions.

The slopes m are fixed before training (not learned), set as geometric sequences. Different heads get different slopes, creating a spectrum from heads that attend broadly (small m, gentle distance penalty) to heads that focus locally (large m, steep penalty). This naturally implements a multi-scale attention pattern without any learned parameters.

ALiBi's defining advantage is length extrapolation. The paper's main result: a model trained on 1024-length sequences achieves equivalent perplexity when evaluated on 2048-length sequences (2x), matching models explicitly trained on the longer length while using 11% less memory and training 11% faster. At longer ranges (5-10x), performance remains reasonable but degrades gradually — the specific extrapolation quality depends on model size, dataset, and evaluation metric. The often-cited "8x" figure appears in informal descriptions rather than as a precise verified benchmark.

Despite strong extrapolation results, ALiBi has seen less widespread adoption than RoPE, partly due to RoPE's earlier momentum, existing infrastructure support, and RoPE's compatibility with length extension techniques (position interpolation, YaRN). ALiBi's main adopters include BLOOM, BloombergGPT, MPT-7B, and MPT-30B.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[RoPE encodes relative position through rotation of Q and K vectors where the dot product naturally incorporates the position difference between tokens]] -- the dominant alternative that won in practice
- [[sinusoidal positional encoding is added not concatenated to token embeddings preserving dimension while forcing position-content interaction]] -- the original approach ALiBi replaces

Topics:
- [[transformer-architecture]]
