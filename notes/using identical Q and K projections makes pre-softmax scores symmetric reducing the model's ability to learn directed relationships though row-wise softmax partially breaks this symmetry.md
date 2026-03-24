---
description: "With Q=K=XW, the score matrix XWW'X' is symmetric before softmax — but row-wise softmax normalization partially breaks this, so the constraint reduces rather than prevents directional attention"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# using identical Q and K projections makes pre-softmax scores symmetric reducing the model's ability to learn directed relationships though row-wise softmax partially breaks this symmetry

If Q and K use the same projection matrix (Q = K = XW), the pre-softmax attention score matrix becomes Q·K^T = XWW^TX^T, which is mathematically symmetric. This means the raw compatibility score between positions i and j is the same as between j and i.

However, the picture is more nuanced than "symmetric projections prevent directed attention." After applying row-wise softmax, the attention weights are generally NOT symmetric even with symmetric scores, because softmax normalizes each row independently — the denominator (sum of all scores in that row) differs between rows. Position i's attention to position j and position j's attention to position i will typically differ because each row sums over different context.

That said, separate Q and K projections give the model strictly more representational freedom. With separate projections, the model can learn that "sat" should strongly attend to "cat" (finding its subject) while "cat" attends weakly to "sat" (less interested in its predicate). This asymmetry is natural in language where relationships are inherently directed.

In practice, some efficient attention variants (Linformer, shared-projection approaches) have shown competitive performance with tied Q/K projections, suggesting the asymmetry primarily helps rather than being strictly required. But for maximum expressiveness, separate projections remain the standard choice in all major LLM architectures.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[Q-K-V separation enables asymmetric context-dependent relevance matching through three independently specialized projections]] -- the full three-projection design this simplification reduces
- [[attention heads empirically specialize into positional syntactic semantic and rare-word roles with most encoder information concentrated in few heads]] -- specialization requires asymmetric flexibility

Topics:
- [[transformer-architecture]]
