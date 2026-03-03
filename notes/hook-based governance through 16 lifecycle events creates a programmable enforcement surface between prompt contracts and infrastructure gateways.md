---
description: "Claude Code hooks fire at 16 lifecycle points from SessionStart through PreCompact to SessionEnd — three handler types (shell, HTTP, LLM prompt) with PreToolUse supporting block, modify, and escalate decisions"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
---

# hook-based governance through 16 lifecycle events creates a programmable enforcement surface between prompt contracts and infrastructure gateways

Claude Code's hook system provides 16 lifecycle events as of early 2026, spanning the full agent lifecycle: SessionStart, UserPromptSubmit, PreToolUse, PermissionRequest, PostToolUse, PostToolUseFailure, Notification, SubagentStart, SubagentStop, Stop, TeammateIdle, TaskCompleted, ConfigChange, WorktreeCreate, WorktreeRemove, PreCompact, and SessionEnd.

Three handler types: shell commands, HTTP endpoints, and LLM prompts. PreToolUse is the most governance-relevant — it supports `hookSpecificOutput` with `permissionDecision` (allow/deny/escalate) and can modify `tool_input` before execution. Session ID available via `${CLAUDE_SESSION_ID}` (v2.1.9+).

The hook layer sits between prompt-level contracts and infrastructure gateways in since [[three-level tool governance layers gateway enforcement hook enforcement and contract enforcement in decreasing reliability but increasing flexibility]]. Hooks are more reliable than contracts because they execute as code outside the agent's context window — immune to the attention degradation that affects prompt-based instructions. But they are less reliable than gateways because they depend on the Claude Code runtime rather than external infrastructure.

For agent teams, TeammateIdle (exit code 2 sends feedback to keep working) and TaskCompleted (exit code 2 prevents completion) provide team-specific quality gates that extend governance to multi-agent workflows.

This vault uses hooks extensively: session-orient.ps1 (SessionStart), session-capture.ps1 (Stop), check_agents_tag.cmd (Stop), check_plan_mode.cmd (PreToolUse), validate-note.cmd and auto-commit.cmd (PostToolUse:Write). The hook system is the infrastructure layer that makes the behavioral contract in CLAUDE.md enforceable rather than advisory.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[three-level tool governance layers gateway enforcement hook enforcement and contract enforcement in decreasing reliability but increasing flexibility]] -- the governance hierarchy
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- the theoretical foundation for hook-based enforcement

Topics:
- [[agent-governance]]
