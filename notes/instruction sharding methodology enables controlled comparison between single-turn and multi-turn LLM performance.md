---
description: "Semi-automatic process transforming single-turn instructions into multi-turn conversations with 5 enforced properties — ~3 hours per 100 instructions"
type: method
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# Instruction sharding methodology enables controlled comparison between single-turn and multi-turn LLM performance

Laban et al. introduce "instruction sharding" — a semi-automatic methodology for transforming fully-specified single-turn instructions into sets of smaller informational units (shards) that collectively preserve the original meaning but are distributed across conversational turns. This enables the first controlled comparison between single-turn and multi-turn performance on identical tasks.

The sharding process enforces five properties: (P1) information preservation — all original information exists across shards, (P2) clear initial intent — the first shard establishes the task domain, (P3) order-insensitive information — shards can be revealed in any order, (P4) maximal granularity — information is split as finely as possible, (P5) minimal transformation — shards stay close to the original wording.

The process is semi-automatic: an LLM identifies logical units, rephrases them conversationally, and verifies completeness, then human authors manually inspect and edit. This takes approximately 3 hours of manual work per 100 sharded instructions. A GPT-4o-mini user simulator manages the conversation, classifying assistant responses into 7 categories (clarification, refusal, hedging, interrogation, discussion, missing, or answer attempt).

The methodology produced 600 sharded instructions across 6 benchmark tasks, with N=10 simulations per condition, totaling 200,000+ conversations. Code and data are released at github.com/microsoft/lost_in_conversation (217 stars) and on HuggingFace.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns]] -- the headline result enabled by this methodology
- [[most existing multi-turn benchmarks are episodic and overestimate real multi-turn performance]] -- the evaluation gap this methodology addresses
- [[the 39 percent degradation figure may overstate the problem for well-designed systems while understating it for messy real-world interactions]] -- critique of the methodology

Topics:
- [[agent-cognition]]
