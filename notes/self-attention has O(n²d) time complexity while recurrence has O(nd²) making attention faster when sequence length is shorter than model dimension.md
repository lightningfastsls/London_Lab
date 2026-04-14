---
description: "Crossover at n=d — typical GPT (d=4096, n=2048) favors attention, but long-context (n=100K+) requires efficient variants like sparse/linear/FlashAttention"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# self-attention has O(n²d) time complexity while recurrence has O(nd²) making attention faster when sequence length is shorter than model dimension

The computational complexity of self-attention is O(n² · d) per layer, where n is the sequence length and d is the model dimension. Recurrent layers have O(n · d²) complexity. The crossover point occurs at n = d — when sequence length exceeds model dimension, attention becomes more expensive.

This matters because it directly governs which architecture is computationally favorable. From Vaswani et al. (2017) Table 1: for most practical NLP settings, n < d holds. A typical GPT model might have d = 4096 processing sequences of n = 2048 tokens — well below the crossover point. But for very long sequences (n = 100K+), the quadratic cost dominates, which is why efficient attention variants (sparse attention, linear attention, FlashAttention) have become essential for long-context models.

The n² term in self-attention arises from computing all pairwise dot products between n positions — the same all-pairs computation that gives [[self-attention provides O(1)-path global context from layer 1 while CNNs require many stacked layers to aggregate distant information]], so the mechanism responsible for quadratic cost is also what makes attention representationally powerful. The d term is the per-pair computation cost. Within each pair, [[dividing by sqrt(d_k) prevents softmax saturation by rescaling dot products to unit variance regardless of dimension]] — the scaling denominator is derived from the same dimension parameter d that appears in the complexity formula. Notably, [[multi-head attention splits computation into parallel specialized subspaces without increasing total computation]] — h heads each compute O(n² · d/h), summing to the same O(n²d) total, so the multi-head design adds specialization without changing the complexity class. For recurrence, the d² term comes from the hidden-state-to-hidden-state weight matrix multiplication at each of n steps.

This complexity analysis also explains why [[KV cache compression techniques extend effective context by 3-32x with trade-offs between memory reduction and information preservation]] — at inference time, the KV cache grows linearly with sequence length, and the attention computation over it grows quadratically, making compression essential for long-context deployment. The quadratic wall is also the primary motivation behind [[infinite context architectures combine compressive memory with standard attention to handle arbitrarily long sequences]] — Infini-Attention, Ring Attention, and StreamingLLM all exist to circumvent the O(n²) cost when n grows unbounded.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[self-attention requires only O(1) sequential operations enabling full parallelization versus O(n) for RNNs]] -- parallelization advantage despite higher total compute
- [[self-attention provides O(1)-path global context from layer 1 while CNNs require many stacked layers to aggregate distant information]] -- the n² pairwise computation that causes quadratic cost is the same mechanism enabling O(1) global reach
- [[dividing by sqrt(d_k) prevents softmax saturation by rescaling dot products to unit variance regardless of dimension]] -- scaling within each pairwise computation, derived from the same d parameter in O(n²d)
- [[self-attention lacks inductive bias for local structure leading to hybrid architectures for domains where locality matters]] -- a separate limitation of attention beyond computational cost; hybrid architectures address both cost and locality
- [[KV cache compression techniques extend effective context by 3-32x with trade-offs between memory reduction and information preservation]] -- practical consequence of quadratic cost
- [[multi-head attention splits computation into parallel specialized subspaces without increasing total computation]] -- h heads preserve O(n²d) total cost while adding specialization
- [[infinite context architectures combine compressive memory with standard attention to handle arbitrarily long sequences]] -- architectural responses to the quadratic cost wall

Topics:
- [[transformer-architecture]]
