---
description: "search queries shift as agents discover new terms and concepts mid-search — checkpointing prevents losing the original intent while allowing productive drift"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[agent-memory]]"
---

# queries evolve during search so agents should checkpoint

When an agent searches the vault, the initial query reflects current understanding. But search results introduce new vocabulary, related concepts, and unexpected connections that shift what the agent is looking for. Without checkpointing, the agent may drift far from the original intent — finding interesting but irrelevant material.

Since [[FLARE uses the agent's intended action as a retrieval query enabling pre-modification knowledge checks]], anchoring retrieval to the original intent prevents unproductive drift. Checkpointing means: record the original query and intent, allow the search to evolve, then verify that the final results still serve the original purpose. This is the search-level equivalent of session orientation. The discipline is lightweight — a single sentence noting "I started looking for X" — but it prevents the common failure mode where an agent follows a chain of interesting links and forgets why it started searching.

---

Relevant Notes:
- [[FLARE uses the agent's intended action as a retrieval query enabling pre-modification knowledge checks]] — intent-anchored retrieval
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] — when retrieval happens affects its value
- [[CRAG's retrieval evaluator prevents noise-induced gate fatigue through relevance thresholds]] — filtering irrelevant results
