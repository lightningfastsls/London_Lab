---
description: "Claude Code, Codex CLI, OpenCode, and Amp all acknowledge compaction loss — each compression discards nuance that subsequent compressions cannot recover"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Context compaction quality degrades cumulatively with multiple compressions regardless of implementation

Summarizing conversation history near context limits while preserving key decisions and unresolved issues — context compaction — is universally implemented but universally acknowledged to degrade cumulatively. Claude Code, Codex CLI, OpenCode, and Amp all implement variants, and all acknowledge quality loss with repeated compressions.

Implementation approaches differ but share the same fundamental limitation:
- **Claude Code**: Auto-compact at ~95% capacity; preserves accomplishments, WIP, files, next steps. Context editing (September 2025) reduced token consumption by 84%.
- **Codex CLI**: Token-based thresholds (180K-244K), preserves recent ~20K tokens plus summary of older content.
- **OpenCode**: Separate pruning mechanism that protects last 40K tokens of tool output.
- **Amp (Sourcegraph)**: Manual "Handoff" only; deliberately avoids automatic summarization, emphasizing user discipline instead.

Each compression discards nuance that subsequent compressions cannot recover. The first compaction loses detailed reasoning paths; the second loses the summaries of those paths; the third may lose the fact that those paths existed at all. This is why Amp's manual-only approach is a deliberate design choice — since [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]], the better strategy is preventing the need for compaction rather than optimizing the compaction itself.

For agent workflows, the implication is clear: design sessions to complete before compaction becomes necessary. The orient→work→persist rhythm, /clear between phases, and Fresh Context Pattern are all compaction-avoidance strategies.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the prevention strategy
- [[Claude Sonnet exhibits a qualitative performance cliff at 147K-152K tokens which is 73-76 percent of its 200K window]] -- what compaction is trying to prevent
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- an alternative to compaction
- [[dream-inspired consolidation cycles compress old memories on daily weekly monthly schedules to manage long-term growth]] -- memory consolidation is scheduled compaction; same lossy compression pattern applied cross-session
- [[semantic compression pipeline achieves 30x token reduction through structured compression online synthesis and intent-aware retrieval]] -- structured compression achieving 30x reduction, but cumulative loss principle likely applies

Topics:
- [[agent-cognition]]
