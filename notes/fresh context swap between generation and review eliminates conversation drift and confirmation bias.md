---
description: "ASDLC pattern uses a fresh session (not just a new prompt) to eliminate anchoring — the Critic evaluates solely against specification contracts, never seeing the builder's reasoning"
type: method
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Fresh context swap between generation and review eliminates conversation drift and confirmation bias

The Adversarial Code Review pattern from ASDLC (Adversarial Software Development Lifecycle) defines a critical mechanism for breaking confirmation bias: the context swap. After the Builder Agent generates code, the Critic Agent begins in a completely fresh session — not just a different prompt in the same conversation, but a new context window with no access to the generation prompt or intermediate outputs.

This distinction matters because since [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]], the fresh context gives the Critic something close to optimal reasoning capacity. But the ASDLC pattern goes further than just context management: the Critic evaluates solely against specification contracts, not the builder's reasoning. This is "contract-based review" — the reviewer never asks "did the builder intend this?" but only "does this satisfy the specification?"

The four-phase workflow — Build, Context Swap, Critique, Verdict — makes the separation explicit and auditable. The Context Swap is not optional or approximate; it is a hard architectural boundary. This connects to the governance principle that since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]], structural enforcement (a new session) is more reliable than instructional enforcement (telling the reviewer to "be critical").

A validated case study caught a performance violation — loading an entire database table into memory for filtering — that passed all tests but would fail at scale. This is exactly the kind of "silent performance risk" that neither deterministic quality gates nor same-context review would catch, because the Critic brought judgment unconstrained by the builder's original framing. The g3 dialectical autocoding pattern extends this further: since [[memory wipe per review turn prevents attention degradation treating each attempt as fresh start guided by coach feedback]], the fresh context principle can be applied not just between roles (builder-to-critic) but between iterations of the same role.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the principle fresh context exploits
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- structural enforcement over instructional
- [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]] -- the problem this pattern solves
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the related isolation pattern for degradation
- [[memory wipe per review turn prevents attention degradation treating each attempt as fresh start guided by coach feedback]] -- extends fresh context from inter-role to intra-role iterations

Topics:
- [[agent-governance]]
- [[agent-cognition]]
