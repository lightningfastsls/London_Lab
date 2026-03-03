---
description: "Amp's active exploration — how to map review feedback to threads, incorporate learning into AGENTS.md, and use acceptance patterns for long-term improvement without overfitting to reviewer preferences"
type: open-question
confidence: speculative
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Whether accepted or rejected AI review comments should feed back into agent learning through persistent memory

Amp's code review system has raised an open question that no current tool has fully addressed: should the pattern of accepted and rejected review comments feed back into the agent's behavior for future reviews?

The question has several dimensions. First, the mapping problem: reviews don't map 1:1 to threads, since a human can review the output of multiple agent work threads simultaneously. How should feedback attribution work when one review comment applies to code generated across several interactions?

Second, the learning mechanism: Amp is exploring whether accepted/rejected review comments should inform behavior through AGENTS.md updates — essentially using the same repository-level configuration files that OpenAI Codex uses for context-specific review guidelines. This creates a feedback loop: agent generates code, reviewer provides feedback, feedback shapes future generation guidelines.

Third, the memory question: how should feedback incorporate into long-term memory? Per-project? Per-reviewer? Per-code-pattern? The granularity of learning determines both its usefulness and its risk of overfitting to specific reviewer preferences.

This connects to the existing open question [[how behavioral norms propagate across agent boundaries in multi-agent systems]] — if review feedback teaches the generation agent to avoid certain patterns, and that agent operates across multiple projects, the learning from one project's reviewer may influence behavior in another project where different norms apply. The propagation of review-derived behavioral norms across system boundaries is an unsolved governance problem.

There is also a risk dimension: since [[sycophancy in AI agents is a product decision not a bug creating tension between business incentives and reliability contracts]], feedback-driven learning could reinforce reviewer preferences rather than objective quality, especially if the learning system cannot distinguish between "this reviewer prefers X" and "X is objectively better."

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[how behavioral norms propagate across agent boundaries in multi-agent systems]] -- the boundary propagation problem
- [[sycophancy in AI agents is a product decision not a bug creating tension between business incentives and reliability contracts]] -- the risk of preference reinforcement
- [[LLM-assisted review works best as complement in AI-led co-reviewer or interactive on-demand mode not as replacement]] -- the review modes that would generate feedback

Topics:
- [[agent-governance]]
