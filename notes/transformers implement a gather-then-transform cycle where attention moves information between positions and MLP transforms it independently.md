---
description: "Each transformer block has two complementary sub-blocks — attention aggregates context across positions (gather), then MLP processes each position's enriched representation (transform)"
type: pattern
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently

Every transformer layer consists of two sub-blocks — multi-head attention followed by a feed-forward network (MLP/FFN) — implementing a complementary gather-then-transform processing cycle. This is not an arbitrary pairing but a fundamental compute pattern.

Attention serves the **gather** role: it routes information between positions. Given what position i needs to predict, attention identifies which other positions have relevant information and aggregates it. This is a movement operation — attention copies and mixes existing representations across positions but does not create genuinely new features through nonlinear combination.

The MLP serves the **transform** role: it processes each position independently through a typically two-layer network with nonlinear activation (expanding from d_model to 4×d_model then back). The MLP transforms the gathered information into new features, applying the nonlinear computations that attention cannot perform.

The cycle repeats at each layer: gather → transform → gather → transform. With each repetition, the model builds increasingly abstract representations. Lower layers gather and transform surface-level patterns; higher layers gather and transform the outputs of prior gather-transform cycles, composing simple patterns into complex concepts.

This pattern directly informs how we interpret hidden states in the USV transformer: since [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]], we are essentially probing what information has been gathered and transformed at each stage of this cycle.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[attention alone cannot compute nonlinear features and MLP alone cannot communicate across positions making both necessary for the complete transformer compute primitive]] -- why both are required
- [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] -- our USV application of probing these representations
- [[the residual stream architecture lets transformer components read from and write to a shared information stream enabling additive accumulation]] -- the shared stream they read/write to

Topics:
- [[transformer-architecture]]
