---
description: Subagents provide context isolation, agent teams enable collaborative messaging, and worktrees ensure filesystem safety — each serving distinct architectural needs
type: finding
confidence: likely
topics: "[[agent-memory]]"
---

# three parallelization strategies in Claude Code serve different isolation needs: subagents for focused results, teams for collaborative discussion, worktrees for filesystem safety

Claude Code provides three distinct parallelization mechanisms, each optimized for a different isolation property:

**Subagents** run in focused context windows and report results back to a parent agent. They provide context isolation — the parent's context window stays clean while the subagent explores a specific question. Communication is unidirectional (child → parent). Best for delegating focused research or validation tasks.

**Agent teams** are independent Claude instances with bidirectional messaging via mailboxes and shared task lists. [[Claude Code agent teams enable inter-agent messaging and shared task lists unlike subagents which only report to parent]]. They provide communication isolation — each agent maintains its own context while collaborating on shared goals. Best for collaborative workflows where agents need to coordinate without merging contexts.

**Worktrees** use git worktrees to give each agent an isolated copy of the filesystem. They provide filesystem isolation — concurrent file modifications cannot conflict. Best for multi-branch development or when agents need to make incompatible changes simultaneously.

The three strategies compose: an agent team member might spawn subagents for focused tasks while working in a worktree for filesystem safety. [[Claude Code as MCP server enables agent-to-agent orchestration where other tools invoke Claude's file editing and command execution remotely]] extends this further by enabling cross-tool orchestration.

Choosing the wrong isolation type creates predictable failures: subagents for collaborative work leads to information bottlenecks at the parent; teams for focused queries wastes orchestration overhead; direct file access without worktrees risks merge conflicts.
