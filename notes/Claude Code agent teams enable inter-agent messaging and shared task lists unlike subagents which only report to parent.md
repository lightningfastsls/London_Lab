---
description: "Experimental first-party multi-agent feature where teammates are independent Claude instances with mailbox messaging, shared tasks, and dependency tracking — architecturally distinct from lightweight subagents"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# Claude Code agent teams enable inter-agent messaging and shared task lists unlike subagents which only report to parent

Claude Code's Agent Teams (experimental research preview, disabled by default) introduce a fundamentally different parallelization model from subagents. The architecture: one session acts as team lead, spawning independent teammates that each get their own full context window. Unlike subagents which can only report results back to a parent, teammates can message each other directly via inbox-based mailbox messaging.

The coordination surface includes shared task lists with three states (pending, in_progress, completed), dependency tracking between tasks, and file-locking for race-condition-safe task claiming. Two display modes: in-process (single terminal, Shift+Down to cycle) and split-pane (requires tmux or iTerm2 — does not work on Windows Terminal, VS Code integrated terminal, or Ghostty).

Best use cases: parallel code review with specialized lenses, competing hypothesis debugging, cross-layer coordination (frontend/backend/tests), research with peer debate. Anti-patterns: sequential tasks, same-file edits, "build me an app" without clear boundaries.

Quality gates via hooks: `TeammateIdle` (exit code 2 sends feedback to keep working) and `TaskCompleted` (exit code 2 prevents completion) provide programmatic enforcement. Since [[hook-based governance through 16 lifecycle events creates a programmable enforcement surface between prompt contracts and infrastructure gateways]], these team-specific hooks extend governance to multi-agent workflows.

Key limitations: no session resumption for teammates, task status can lag, one team per session, no nested teams, permissions inherited from lead at spawn time.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]] -- the comparison framework
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the subagent pattern teams extend

Topics:
- [[agent-memory]]
