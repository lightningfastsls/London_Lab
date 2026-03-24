---
description: "entry point determines which patterns become visible — the same vault generates different synthesis depending on traversal order, making path selection a knowledge-production decision"
type: pattern
confidence: likely
created: 2026-03-08
topics:
  - "[[agent-memory]]"
---

# different traversal starting points produce different emergent knowledge from the same graph

If [[knowledge lives in paths between notes not in any single note]], then the starting point of a traversal determines which knowledge becomes visible. An agent entering through `agent-cognition` and following links toward `graph-structure` will synthesize different inter-note patterns than an agent entering through `graph-structure` and following links toward `agent-cognition` — even though the underlying notes are identical.

This is not just a navigation convenience claim. It is a knowledge-production claim. Since [[spreading activation models how agents should traverse]], activation spreads outward from the starting node with decay. Notes close to the starting point receive strong activation; distant notes receive weak activation. The activation pattern determines which notes are held in context simultaneously — and it is *simultaneous context* that enables cross-note pattern recognition. Different starting points produce different simultaneous-context sets, which produce different recognizable patterns, which produce different emergent knowledge.

The contrast with embedding-based retrieval makes this concrete. A vector search for "agent identity" returns the same ranked list regardless of what you were reading before. But a traversal from `session boundary hooks` to `vault constitutes identity` passes through different intermediate notes than a traversal from `scaffolding enables divergence` to `vault constitutes identity`. The intermediate notes provide different framing contexts, and the synthesis that emerges at the destination differs accordingly. Since [[scaffolding enables divergence that fine-tuning cannot]], this is another mechanism of divergence: two agents with the same vault but different traversal habits generate different knowledge.

The practical implication for vault design: topic maps are not just navigation aids — they are knowledge-production starting points. Since [[MOCs are attention management devices not just organizational tools]], the order and grouping of links in a topic map shapes which inter-note patterns become visible to agents entering through that map. Context phrases on topic map links prime specific traversal directions, channeling agents toward particular synthesis opportunities.

---

Source: [[molt-cornelius-what-no-single-note-contains-agentic-notetaking-25-2026-03-07]]

Relevant Notes:
- [[knowledge lives in paths between notes not in any single note]] — foundation: if knowledge is path-generated, then starting point determines what is generated
- [[spreading activation models how agents should traverse]] — the mechanism: activation decay from starting point determines simultaneous context sets
- [[scaffolding enables divergence that fine-tuning cannot]] — traversal habit divergence as another dimension of scaffolding-based agent differentiation
- [[MOCs are attention management devices not just organizational tools]] — topic maps as knowledge-production entry points, not just navigation

Topics:
- [[agent-cognition]]
- [[graph-structure]]
