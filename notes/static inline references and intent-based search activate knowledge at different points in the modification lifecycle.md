---
description: "canary comments fire when the agent reads a file (reactive), while intent-based search fires when the agent plans a modification (proactive) — the earlier trigger catches cross-system constraints the file-level reference misses"
type: decision
confidence: likely
created: 2026-03-07
meta_state: current
---

# static inline references and intent-based search activate knowledge at different points in the modification lifecycle

Two knowledge activation mechanisms serve different timing points in the code modification lifecycle:

**Static inline references** (canary comments like `# VAULT: <note-title>`) activate when the agent *reads the file* it's about to modify. The reference is right there — zero search cost, zero latency. But activation happens late: the agent has already decided to modify this file, already has an approach in mind, and the canary can only course-correct, not redirect.

**Intent-based search** (describing planned work and searching the vault) activates when the agent is *planning* the modification, before any file is opened. Since [[FLARE uses the agent's intended action as a retrieval query enabling pre-modification knowledge checks]], the intent description catches constraints from *related* systems — a note about labeling pipeline assumptions might be critical when modifying detection overlays, but it wouldn't appear in the detection file's canary comment.

The two mechanisms are complementary, not competing:
- Canary comments are the safety net — they activate even when the agent skips /kcheck
- Intent-based search is the first line — it activates early enough to influence the approach
- Together they create defense-in-depth for knowledge activation

A third timing point — session start — provides the broadest activation. Goal-aware orient searches surface notes relevant to the session's overall threads, before any specific modification is planned. This creates three layers: session-level (broad), planning-level (focused), file-level (specific).

---

Source: knowledge-activation-architecture-phase1-2026-03-07 (archived inbox)

Relevant Notes:
- [[FLARE uses the agent's intended action as a retrieval query enabling pre-modification knowledge checks]] — the mechanism behind intent-based search
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] — the thesis this architecture serves
- [[session boundary hooks implement cognitive bookends for orientation and reflection]] — the session-start timing point

Topics:
- [[agent-memory]]
