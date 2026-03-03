---
description: "H(X_t|X_{t-lag}) measures how much one specific past token reduces uncertainty, revealing temporal decay of individual token influence"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[representation-learning]]"
---

# conditional entropy by lag probes single-token influence at distance unlike entropy rate which uses contiguous history

Entropy rate H(X_n | X_{n-1}, ..., X_{n-k}) conditions on the entire contiguous history of length k, measuring the total predictability afforded by recent context. This is the standard measure of sequential structure, but it conflates multiple sources of information: direct influence from each individual past token, redundant information shared across nearby tokens, and synergistic information that only emerges from combinations of past tokens. Conditional entropy by lag — H(X_t | X_{t-lag}) for a specific lag value — isolates a different quantity: how much uncertainty about the current token is reduced by knowing a single token at a specific temporal distance.

These two measures are complementary and can reveal different aspects of sequential organization. Entropy rate tells you "how predictable is the sequence given everything recent," which is the operationally relevant quantity for compression and prediction. Conditional entropy by lag tells you "how far does a single token's influence reach," which reveals the temporal decay structure of individual dependencies. If conditional entropy by lag decays slowly with increasing lag (influence persists at long distances), that suggests long-range dependencies even if the entropy rate converges quickly with context length. This can happen because contiguous context may redundantly encode the same long-range information — knowing tokens at positions t-1 through t-5 may already capture whatever token t-20 would contribute, so the entropy rate plateaus early even though individual tokens at distance 20 still carry predictive information.

For USV code sequences, the decay profile of conditional entropy by lag provides insight into the temporal scale of vocalization organization. A rapid exponential decay suggests local-only structure (adjacent codes predict each other but distant codes are independent). A slow power-law decay suggests scale-free organization where vocalization choices influence the sequence at multiple time scales. This complements the entropy rate analysis in [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] by decomposing the aggregate predictability into individual-lag contributions.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- the aggregate entropy rate curve that this lag-specific analysis decomposes into individual contributions
- [[excess entropy measures long-range structure complexity in discrete code sequences]] -- excess entropy integrates all long-range dependencies; conditional entropy by lag reveals their temporal distribution
- [[null models are essential for interpreting information-theoretic metrics on USV code sequences]] -- Markov null models will show specific lag decay profiles that serve as baselines for the real data

Topics:
- [[representation-learning]]
