---
description: "Morphic's search subagent runs in its own context window issuing up to 8 parallel tool calls per turn — Claude Opus 4.5 goes from 45.9 to 57.5 percent on SWE-Bench Pro, demonstrating search as orthogonal capability multiplier"
type: baseline
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# WarpGrep RL-trained search subagent lifts SWE-Bench Pro by 11.6 percentage points while reducing cost 15.6 percent and improving speed 28 percent

WarpGrep is an RL-trained agentic code search subagent that runs in its own context window, issuing up to 8 parallel tool calls per turn. SWE-Bench Pro results: Claude Opus 4.5 goes from 45.9% to 57.5% with WarpGrep v2 (+11.6pp). On Opus 4.6: 15.6% cheaper and 28% faster. Works as MCP server inside Claude Code, Cursor, Windsurf, and Codex.

The key pattern is search as an orthogonal capability multiplier rather than a replacement. WarpGrep does not change how the agent reasons — it changes how effectively the agent finds relevant code. By isolating search in a dedicated subagent with its own context window, the main agent's context stays clean for reasoning.

This validates the subagent isolation pattern. Since [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]], WarpGrep demonstrates that specialized subagents for search can improve both quality and cost simultaneously — a rare case where the efficiency frontier moves in both dimensions.

The RL training is notable: rather than using rule-based search heuristics, WarpGrep learns search strategies that adapt to codebase structure. This is a concrete example of since [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] taken further — the tool itself is learned rather than hand-coded.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the isolation pattern WarpGrep implements
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- tools in agentic loops, which WarpGrep extends with RL
- [[vendor self-evaluation bias means every AI code review benchmark vendor wins their own evaluation]] -- WarpGrep's SWE-Bench results are vendor-published; independent replication needed
- [[Search-R1 found REINFORCE outperformed both PPO and GRPO for agentic deep research tasks with the highest accuracy and most efficient search strategies]] -- parallel finding that RL-trained search outperforms alternatives; both demonstrate RL-for-search as a productive paradigm

Topics:
- [[agent-governance]]
