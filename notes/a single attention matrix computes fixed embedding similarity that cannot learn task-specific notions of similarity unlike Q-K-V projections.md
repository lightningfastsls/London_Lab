---
description: "Without learned Q/K/V projections, matching is limited to raw embedding dot products — same notion of similarity for every relationship type regardless of context"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# a single attention matrix computes fixed embedding similarity that cannot learn task-specific notions of similarity unlike Q-K-V projections

Without separate Q, K, and V projections, attention would compute raw dot products between token embeddings. This is a fixed computation — the same notion of "similarity" applies everywhere, regardless of what type of relationship the model needs to capture. Two tokens that are close in embedding space will always attend to each other, even when the task requires attending to distant or contrasting tokens.

The three-matrix design of Q/K/V enables the model to learn what to look for (Q), what to advertise as matchable (K), and what to contribute when matched (V) — all independently and all learned from data. This means the model can learn entirely different notions of similarity for different heads and layers. One head might match tokens by syntactic role while another matches by semantic topic, even though both operate on the same underlying embeddings.

This is distinct from simply adding more capacity. A single wider projection could capture more information, but it would still produce one notion of similarity. The Q/K separation specifically enables the model to learn that the criteria for "what am I looking for" can be completely different from "how should I be found" — which since [[Q-K-V separation enables asymmetric context-dependent relevance matching through three independently specialized projections]], is the key insight of the transformer's attention mechanism.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[Q-K-V separation enables asymmetric context-dependent relevance matching through three independently specialized projections]] -- the design that solves this limitation

Topics:
- [[transformer-architecture]]
