---
description: "Appendix F.1 documents that models produce complete solutions from minimal initial context, creating anchors that resist correction in subsequent turns"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# LLMs attempt full solution generation on the first turn even when given only a vague initial shard

Laban et al. document in Appendix F.1 that LLMs consistently attempt to generate full, complete solutions on the very first turn of a sharded conversation — even when that first turn contains only a vague, underspecified description of the task. Rather than asking for clarification or producing a partial, hedged response, models fill in missing details with assumptions and present a confident, complete answer.

This is a distinct behavioral pattern from general "premature helpfulness." It is the specific anchoring mechanism: the first-turn solution becomes the reference frame for all subsequent turns. When later shards reveal that initial assumptions were wrong, the model faces a choice between revising its anchored solution (which RLHF training penalizes as inconsistency) and layering corrections on top (which preserves the appearance of coherence). Models consistently choose the latter, since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]].

For Claude Code specifically, this explains why giving a vague initial task description produces worse outcomes than writing a complete specification file first. The first-turn solution anchors the entire session.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the training incentive driving this behavior
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- the cascading consequence
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the specification-first alternative

Topics:
- [[agent-cognition]]
