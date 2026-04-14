---
description: "Each head operates on d_model/h dimensions with its own Q/K/V projections — total cost O(n²·d_model) same as single-head, but enables diverse relationship detection"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# multi-head attention splits computation into parallel specialized subspaces without increasing total computation

Rather than computing a single attention function over the full d_model-dimensional space, multi-head attention splits the computation into h parallel heads, each operating on a d_k = d_model/h dimensional subspace. Each head has its own learned W_Q, W_K, W_V projection matrices, computes attention independently, and the h outputs are concatenated and projected back through a final W_O matrix.

The computational cost is h × O(n² × d_model/h) = O(n² × d_model) — identical to single-head attention. As Vaswani et al. (2017) note: "due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality." The multi-head design provides specialization for free.

Why this works: each head operates in its own low-dimensional subspace, giving it the opportunity to specialize in different types of relationships. Different heads can develop different matching criteria through their independent Q/K/V projections. This turns out to be more powerful than a single high-dimensional attention because it allows the model to simultaneously attend to information from different representation subspaces at different positions — something a single attention function cannot do.

This connects to why [[Q-K-V separation enables asymmetric context-dependent relevance matching through three independently specialized projections]] is so powerful — multiply that specialization by h heads, each learning its own notion of "what to look for" and "what to contribute," and the combined output captures far richer relational structure than any single attention pass.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[Q-K-V separation enables asymmetric context-dependent relevance matching through three independently specialized projections]] -- per-head projections enable further specialization
- [[attention heads empirically specialize into positional syntactic semantic and rare-word roles with most encoder information concentrated in few heads]] -- empirical evidence this specialization occurs
- [[dividing by sqrt(d_k) prevents softmax saturation by rescaling dot products to unit variance regardless of dimension]] -- the d_k = d_model/h from multi-head splitting determines the scaling denominator

Topics:
- [[transformer-architecture]]
