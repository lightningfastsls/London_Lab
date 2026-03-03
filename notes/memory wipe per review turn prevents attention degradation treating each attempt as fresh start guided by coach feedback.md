---
description: "Block g3 dialectical autocoding wipes the player's memory every turn — a new instance sees only original requirements plus the coach's latest feedback, preventing cumulative drift"
type: method
confidence: experimental
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Memory wipe per review turn prevents attention degradation treating each attempt as fresh start guided by coach feedback

Block's research on dialectical autocoding (December 2025) introduces a radical approach to attention degradation in iterative code generation: rather than accumulating context across iterations, wipe the agent's memory completely on every turn. A new instance sees only two things: the original requirements and the coach's latest feedback. Every attempt is a genuinely fresh start guided by specific critique.

This is architecturally distinct from both subagent isolation and context swap. Since [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]], subagents use fresh context but within a task they accumulate state. The ASDLC context swap creates a fresh session between generation and review roles. G3's memory wipe creates a fresh session between iterations of the SAME role. The player never sees its own previous attempts — only the coach's assessment of what was wrong.

The ablation study provides the key evidence: removing coach feedback resulted in non-functional code after 4 rounds despite spontaneous improvements each round. This proves two things: (1) fresh context alone is not sufficient — guidance is necessary to prevent random walk; (2) the coach's feedback is the compression mechanism — it distills the essential learning from each round into a directive that guides the next fresh attempt.

The approach draws explicitly on Hegelian dialectics: thesis (implementation), antithesis (critique), synthesis (improved implementation). Each round is a full thesis-antithesis-synthesis cycle, with the memory wipe ensuring that synthesis begins from first principles rather than from accumulated assumptions. This is a particularly interesting contrast to the typical context management strategy of preserving and compressing information -- g3 argues that for iterative refinement, forgetting and redirecting is superior to remembering and adjusting.

The memory wipe is the most radical form of the recovery mechanism described in ABC: since [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]], g3 achieves the strongest possible recovery -- resetting drift to zero every turn rather than merely bounding it. Where the ASDLC pattern uses a [[fresh context swap between generation and review eliminates conversation drift and confirmation bias]] between roles, g3 extends the same principle to iterations within the same role.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the related but distinct isolation pattern
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the fresh context principle g3 exploits per iteration
- [[context compaction quality degrades cumulatively with multiple compressions regardless of implementation]] -- the problem memory wipe sidesteps entirely
- [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]] -- memory wipe is the most radical recovery: drift reset to zero every turn
- [[fresh context swap between generation and review eliminates conversation drift and confirmation bias]] -- inter-role fresh context; g3 extends to intra-role iterations

Topics:
- [[agent-governance]]
- [[agent-cognition]]
