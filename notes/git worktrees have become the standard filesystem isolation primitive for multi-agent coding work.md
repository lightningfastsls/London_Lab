---
description: "Claude Squad, ccswarm, CCPM, Agent Deck, and Claude Code built-in all converge on git worktrees for agent filesystem isolation — mirroring containerization patterns from DevOps"
type: pattern
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# git worktrees have become the standard filesystem isolation primitive for multi-agent coding work

Git worktrees have emerged as the convergent filesystem isolation primitive across the multi-agent coding ecosystem. Five independent implementations all adopt worktrees:

- **Claude Code built-in** — native worktree support for subagents and teams (announced by Boris Cherny)
- **Claude Squad** (6.2K stars) — tmux + git worktrees for codebase isolation
- **ccswarm** — template-based scaffolding with git worktree isolation
- **CCPM** — GitHub Issues + git worktrees for parallel agent execution
- **Agent Deck** — TUI mission control using worktrees

The pattern mirrors containerization from DevOps: each agent gets an isolated copy of the codebase so file modifications cannot conflict. Unlike full containers, worktrees share the same git history and can merge results back cleanly.

Worktrees solve a different problem than context isolation. Since [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]], subagents isolate context windows. Worktrees isolate filesystems. Since [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]], the two are composable — an agent team can use worktrees for filesystem safety while subagents handle focused subtasks.

The convergence on worktrees rather than alternatives (Docker containers, separate clones, branch switching) reflects that git worktrees are lightweight (~instant creation), share object storage (no disk duplication), and produce clean merges via standard git workflows.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]] -- the three-layer framework
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- context isolation (complementary)

Topics:
- [[agent-memory]]
