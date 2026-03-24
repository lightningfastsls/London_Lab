---
description: "spreading activation from cognitive science — activating one node primes connected nodes — maps to how agents should follow links during vault traversal"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[graph-structure]]"
  - "[[agent-external-cognition]]"
---

# spreading activation models how agents should traverse

Spreading activation is a cognitive science model where activating one concept primes related concepts along associative links. In knowledge graph traversal, the analogy is direct: reading one note activates the claims it links to, which activates their connections in turn. The traversal strategy matters because since [[different traversal starting points produce different emergent knowledge from the same graph]], activation patterns determine which regions of the graph become primed.

Since [[knowledge lives in paths between notes not in any single note]], the activation pattern shapes what inter-note knowledge can be generated. Practically, this means topic maps function as activation sources — they determine where traversal begins, which determines what the agent discovers. A topic map with well-chosen context phrases biases activation toward productive paths, while a flat list of links produces unfocused activation that dissipates before reaching useful depth.

---

Relevant Notes:
- [[different traversal starting points produce different emergent knowledge from the same graph]] — path-dependence of activation patterns
- [[knowledge lives in paths between notes not in any single note]] — activation generates inter-note knowledge
- [[MOCs are attention management devices not just organizational tools]] — topic maps as activation sources
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] — temporal dimension of activation
