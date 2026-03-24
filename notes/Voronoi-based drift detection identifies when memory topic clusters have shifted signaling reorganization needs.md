---
description: "Ember MCP monitors embedding-space topology to detect when the centroid of a topic cluster has moved — flagging migrations like Redux to Zustand without auto-applying changes"
type: method
confidence: experimental
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# Voronoi-based drift detection identifies when memory topic clusters have shifted signaling reorganization needs

Ember MCP implements Voronoi-based drift detection that monitors the embedding-space topology of stored memories. When the centroid of a topic cluster moves significantly — for example, as a codebase migrates from Redux to Zustand — the system flags the drift as a signal that memory reorganization may be needed.

Critically, drift detection flags only; it does not auto-shadow or auto-delete. The design philosophy is that structural changes in knowledge require human or agent judgment — the system surfaces the signal but does not act on it. This is a deliberate separation of detection from enforcement.

The mechanism works by partitioning the embedding space into Voronoi cells around topic centroids and tracking centroid positions over time. When a centroid shifts beyond a threshold, it indicates that the cluster's semantic center has moved — new memories about a topic are systematically different from older ones.

This addresses a problem that simple temporal decay cannot: topic evolution. A temporal decay strategy might keep recent Zustand memories and discard old Redux memories, but it cannot detect that these represent a deliberate migration rather than topic drift. Voronoi detection identifies the structural change, enabling targeted shadowing since [[HESTIA scoring uses shadow-decay with quadratic penalty for contradicted memories enabling graceful knowledge evolution without deletion]].

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[HESTIA scoring uses shadow-decay with quadratic penalty for contradicted memories enabling graceful knowledge evolution without deletion]] -- the complementary scoring mechanism
- [[three forgetting strategies dominate agent memory temporal decay importance-based pruning and contradiction-based shadowing]] -- the broader design space
- [[between-session observation accumulation is directed dreaming that produces patterns no individual session contained]] — drift detection signals ARE accumulated observations: the vault's rethink cycle detects drift through observation patterns across sessions, serving the same function as Voronoi centroid monitoring but through human-readable friction signals rather than embedding space topology
- [[each between-session processing cycle is a training step that does not touch the weights]] — drift detection feeds structural updates: when Voronoi-detected drift triggers reorganization, the reorganization is a between-session training step that updates vault structure without model weight changes

Topics:
- [[agent-memory]]
