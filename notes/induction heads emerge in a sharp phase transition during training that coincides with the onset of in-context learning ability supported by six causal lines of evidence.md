---
description: "Not gradual improvement — a discrete transition with visible loss curve bump, where one-layer models never develop induction heads and never develop substantial ICL, providing causal evidence"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# induction heads emerge in a sharp phase transition during training that coincides with the onset of in-context learning ability supported by six causal lines of evidence

Olsson et al. (2022) document that induction heads do not develop gradually during training — they emerge in a sharp phase transition. Before the transition, in-context learning ability is weak and loss does not decrease much with more context. During the transition, induction heads form suddenly, accompanied by a visible "bump" in the training loss curve. After the transition, ICL ability increases dramatically and loss decreases substantially with more context tokens.

Six complementary lines of evidence establish a causal link between induction head formation and ICL ability:

1. The phase change in training coincides precisely with induction head formation and ICL improvement
2. Architectural changes that prevent induction head formation correspondingly prevent ICL improvement
3. "Knocking out" induction heads at test time greatly reduces in-context learning
4. Induction heads appear in models of all sizes tested (one-layer models excepted — they cannot form the two-layer circuit)
5. The mechanism generalizes beyond exact token matching to fuzzy/semantic pattern completion
6. The timing and character of the phase change is consistent across different training runs

The one-layer control is particularly compelling: since [[induction heads implement pattern completion via a two-layer circuit where previous-token heads write context and induction heads read it to predict continuations]], the mechanism inherently requires two layers. One-layer models never develop induction heads and correspondingly never develop substantial in-context learning, providing a clean causal test.

This phase transition phenomenon has broader implications for understanding how capabilities emerge in neural networks — not through gradual accumulation but through sudden formation of specific computational circuits.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[induction heads implement pattern completion via a two-layer circuit where previous-token heads write context and induction heads read it to predict continuations]] -- the mechanism that emerges in this transition
- [[induction heads in larger models generalize from exact-match token copying to fuzzy pattern completion]] -- the more sophisticated behavior that builds on this foundation
- [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]] -- parallel emergence phenomenon: both ICL and reasoning-via-RL appear as phase transitions rather than gradual improvements during training

Topics:
- [[transformer-architecture]]
