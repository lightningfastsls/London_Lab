---
description: "Claude Squad (6.2K stars, 6 tools) uses tmux+worktrees, Claude Flow/Ruflo (12.9K stars, 60+ agents) adds swarm topologies and self-learning — both support tools beyond Claude Code that official teams cannot"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# Claude Squad and Claude Flow provide community multi-agent orchestration with broader tool support than official agent teams

Two community orchestration tools have achieved significant adoption as alternatives to Claude Code's official agent teams:

**Claude Squad** (6.2K stars, v1.0.16, AGPL-3.0) — Terminal-based TUI using tmux for isolated terminal sessions and git worktrees for codebase isolation. Supports Claude Code, Aider, Codex, Gemini, OpenCode, and Amp. Key feature: background task completion with auto-accept mode. Install via Homebrew or curl.

**Claude Flow / Ruflo** (12.9K stars) — Enterprise-grade orchestration by Ruv. 60+ specialized agents with self-learning capabilities. Multiple swarm topologies (mesh, hierarchical, ring, star), consensus protocols (Raft, BFT, Gossip, CRDT), Q-Learning router, Mixture of Experts (8), 42+ skills. Now integrates with Claude Code's Agent Teams for native multi-instance coordination.

The key differentiator from official agent teams: both support tools beyond Claude Code. Official agent teams only orchestrate Claude Code instances. Community tools orchestrate across AI coding platforms, enabling heterogeneous agent teams that use the best tool for each subtask.

Since [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]], community tools add a fourth dimension: cross-tool orchestration.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]] -- the official parallelization framework these extend
- [[git worktrees have become the standard filesystem isolation primitive for multi-agent coding work]] -- the convergent isolation pattern both tools use

Topics:
- [[agent-memory]]
