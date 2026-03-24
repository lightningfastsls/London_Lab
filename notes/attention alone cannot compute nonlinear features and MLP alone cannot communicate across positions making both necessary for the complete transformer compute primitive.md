---
description: "Attention produces linear combinations of value vectors (weighted average) — it needs MLP's nonlinearity for feature creation, while MLP needs attention to break position independence"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# attention alone cannot compute nonlinear features and MLP alone cannot communicate across positions making both necessary for the complete transformer compute primitive

Neither attention nor MLP is sufficient alone for general computation — they have complementary limitations that together form the complete transformer primitive.

**Attention's limitation**: Attention is fundamentally a weighted average — it produces linear combinations of value vectors. Even though the attention weights themselves are computed through a nonlinear process (softmax), the output is a weighted sum of V vectors. This means attention can mix and route existing information across positions but cannot create genuinely new features through nonlinear combination. It cannot implement functions like XOR on features. Both attention (via softmax) and MLP (via GeLU/SiLU) are "mostly linear with a single nonlinearity" as Elhage et al. (2021) note, but attention's nonlinearity is in the weight computation, not in the output transformation.

**MLP's limitation**: The feed-forward network processes each position independently — there is no cross-position communication. Without attention, a token at position 10 has zero access to information from position 5. This is why pure MLP architectures (like gMLP, Tolstikhin et al., 2021) require special gating mechanisms to approximate cross-position interaction, and even then they underperform transformers.

**Together**: attention gathers relevant context from across the sequence (communication between positions), and MLP transforms that gathered context into useful features (computation at each position). The gather-then-transform cycle since [[transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently]] repeats at each layer, building increasingly abstract representations that require both communication and computation.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently]] -- the pattern that combines both
- [[MLP layers store factual associations as distributed key-value memories where first-layer weights match patterns and second-layer weights output associated information]] -- what MLP specifically contributes

Topics:
- [[transformer-architecture]]
