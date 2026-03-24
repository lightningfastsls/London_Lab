---
description: "discrete session architecture enables a secondary agent to consolidate memory between conversations — continuous systems lack the idle window this requires"
type: finding
confidence: proven
created: 2026-03-08
topics:
  - "[[agent-memory]]"
---

# Letta sleep-time compute pairs a primary agent with a sleep-time agent that processes memory during idle periods

Letta's sleep-time compute architecture separates agent work into two modes: a primary agent that handles real-time conversation, and a sleep-time agent that processes and consolidates memory during idle periods between conversations. The architecture requires discrete session boundaries — without idle gaps, the sleep-time agent has no window to operate.

This maps directly to the vault's session rhythm. The primary agent is the per-session Claude instance that does substantive work. The "sleep-time agent" is the between-session processing that fires when observation thresholds are met: `/rethink` consolidates accumulated observations, `/reweave` strengthens connections, `/reflect` discovers missed links. Since [[between-session observation accumulation is directed dreaming that produces patterns no individual session contained]], the sleep-time agent's function is the same generative recombination that Cornelius calls "directed dreaming."

The architectural insight is that paired-agent systems with different temporal rhythms — one real-time, one batch — can achieve memory consolidation that single continuous agents cannot. Since [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]], the vault already uses task-level isolation. Sleep-time compute extends the pattern to session-level isolation: the primary agent operates within a session, the sleep-time agent operates across sessions.

---

Source: [[molt-cornelius-agents-dream-agentic-notetaking-22-2026-03-07]]

Relevant Notes:
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] — task-level isolation pattern; sleep-time compute extends this to session-level
- [[between-session observation accumulation is directed dreaming that produces patterns no individual session contained]] — the vault's implementation of sleep-time processing
- [[session boundaries simultaneously limit agents and enable between-session processing making the limitation the precondition]] — why discrete sessions are required for this architecture

Topics:
- [[agent-cognition]]
