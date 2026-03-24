---
description: "instead of searching for what you already know, describe what you're about to do and let that intent drive retrieval — catches constraints from related systems, not just the file being edited"
type: method
confidence: likely
created: 2026-03-07
meta_state: current
topics:
  - "[[agent-memory]]"
---

# FLARE uses the agent's intended action as a retrieval query enabling pre-modification knowledge checks

FLARE — Forward-Looking Active REtrieval (Jiang et al., EMNLP 2023) — uses the model's *draft output* as a retrieval query. Rather than searching for information the model already has, FLARE searches using what the model is *about to generate*, retrieving context that might correct or constrain the next generation step.

Applied to coding agents, the translation is: before modifying a system, the agent describes what it intends to do, and that description becomes the search query against the knowledge vault. This is fundamentally different from searching for the file name or the function being modified — intent-based search catches constraints from *related* systems that wouldn't appear in file-specific searches.

For example, modifying the detection app's overlay rendering might surface constraints from a note about the labeling pipeline's assumptions — a constraint that wouldn't be in the detection file's comments or even in a search for "detection overlay." The intent "change how saved detections are rendered" activates broader semantic connections than the file path `spectrogram_view.py`.

Since [[spreading activation models how agents should traverse]], FLARE adds a specific answer to the question of WHAT query to use for initial activation: the agent's stated intent. This complements the spreading activation model, which describes how to traverse once activated but doesn't specify the initial activation query. Since [[activation timing matters as much as retrieval quality in agent knowledge systems]], FLARE addresses the timing problem by triggering retrieval at the planning stage — before code is modified, when constraints can still influence the approach.

---

Source: knowledge-activation-architecture-phase1-2026-03-07 (archived inbox)

Relevant Notes:
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] — the thesis this method serves
- [[spreading activation models how agents should traverse]] — covers traversal mechanics; this note adds the initial query mechanism
- [[queries evolve during search so agents should checkpoint]] — related: FLARE's query is the agent's current intent, which evolves as work progresses

Topics:
- [[agent-memory]]
