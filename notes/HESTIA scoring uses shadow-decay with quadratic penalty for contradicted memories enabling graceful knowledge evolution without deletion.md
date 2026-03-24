---
description: "Ember MCP's scoring formula combines similarity, importance, and recency with a quadratic shadow penalty that fades contradicted memories gradually — preserving audit trail while reducing retrieval weight"
type: method
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# HESTIA scoring uses shadow-decay with quadratic penalty for contradicted memories enabling graceful knowledge evolution without deletion

Ember MCP (v3.0.0, 26 stars) implements the most sophisticated forgetting mechanism in the MCP memory server landscape. The HESTIA scoring formula:

```
score = similarity * shadow_penalty * (0.6 + 0.3 * importance + 0.1 * recency)
```

Where `shadow_penalty = (1.0 - shadow_load)^2.0`. When memories are contradicted by newer information, shadow_load increases and the quadratic penalty rapidly reduces retrieval weight without deleting the memory. This preserves the audit trail — you can still find contradicted memories if you look for them, but they naturally fade from relevance-ranked results.

The weighting (60% similarity, 30% importance, 10% recency) is notable: it prioritizes topical relevance over recency, countering the common failure mode of recency bias where recent-but-irrelevant memories crowd out older-but-relevant ones.

The shadow-decay mechanism works in conjunction with since [[Voronoi-based drift detection identifies when memory topic clusters have shifted signaling reorganization needs]] — drift detection identifies structural changes in knowledge, while shadow-decay handles individual memory evolution.

This approach has an analogue in knowledge management: the vault's `meta_state` field (current/outdated/superseded) serves a similar function — marking notes as superseded rather than deleting them preserves provenance while signaling reduced relevance.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[three forgetting strategies dominate agent memory temporal decay importance-based pruning and contradiction-based shadowing]] -- the taxonomy this method belongs to
- [[Voronoi-based drift detection identifies when memory topic clusters have shifted signaling reorganization needs]] -- the complementary structural detection mechanism
- [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]] -- the problem this solves
- [[each between-session processing cycle is a training step that does not touch the weights]] — shadow-decay scoring is a structural parameter that adjusts between sessions: as contradicting memories arrive, shadow_load increases and retrieval weights shift, analogous to weight updates through vault structure changes

Topics:
- [[agent-memory]]
