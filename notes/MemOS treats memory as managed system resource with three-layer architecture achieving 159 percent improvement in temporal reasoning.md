---
description: "Shanghai Jiao Tong and Zhejiang's MemOS (May 2025) introduces MemCube units encapsulating content plus metadata with provenance and versioning — 159 percent over OpenAI memory on LOCOMO temporal tasks and 60.95 percent token reduction"
type: finding
confidence: likely
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# MemOS treats memory as managed system resource with three-layer architecture achieving 159 percent improvement in temporal reasoning

MemOS (Shanghai Jiao Tong / Zhejiang, May 2025) reframes agent memory as a managed system resource — analogous to how an operating system manages files, processes, and I/O — rather than a simple key-value store or retrieval system.

The three-layer architecture: API layer (unified interface), scheduling/management layer (lifecycle, prioritization), storage/infrastructure layer (physical storage). MemCube units encapsulate content + metadata (provenance, versioning) that can be composed, migrated, and fused across systems.

Results on LOCOMO: 159% improvement in temporal reasoning over OpenAI memory, 38.97% accuracy gain, 60.95% token reduction. v2.0 (Dec 2025) added multi-modal memory, tool memory for planning, and cross-project knowledge base sharing.

The paradigm shift is from "memory as retrieval" to "memory as managed lifecycle." Since [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]], MemOS addresses this by making lifecycle management (creation, evolution, decay, archival) a first-class concern rather than an afterthought. The MemCube abstraction — memory with metadata about its own provenance and version history — enables operations like "show me what I knew about X at time T" that flat storage cannot support.

This connects to knowledge management: the vault's YAML frontmatter (description, type, confidence, meta_state, created) is a simpler version of MemCube metadata — provenance and lifecycle state attached to each knowledge unit.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]] -- the problem MemOS addresses architecturally
- [[dream-inspired consolidation cycles compress old memories on daily weekly monthly schedules to manage long-term growth]] -- one consolidation strategy within the broader lifecycle MemOS manages

Topics:
- [[agent-memory]]
