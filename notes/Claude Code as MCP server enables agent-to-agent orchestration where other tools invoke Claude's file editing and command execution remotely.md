---
description: "claude mcp serve exposes Claude Code's capabilities via MCP protocol — other MCP clients (Claude Desktop, Cursor, Windsurf) can invoke Claude Code remotely, enabling recursive agent delegation and tool specialization"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# Claude Code as MCP server enables agent-to-agent orchestration where other tools invoke Claude's file editing and command execution remotely

`claude mcp serve` exposes Claude Code's file editing and command execution tools via MCP protocol. This inverts the typical relationship: instead of Claude Code consuming MCP servers, other MCP clients (Claude Desktop, Cursor, Windsurf) can invoke Claude Code as a tool.

This enables an "agent-to-agent" orchestration pattern: specialization across tools where each agent handles what it does best. Greptile for code review, WarpGrep for search, Nia for documentation context, Claude Code for file editing and execution. Rather than building a monolithic agent with all capabilities, the pattern composes specialized agents via MCP.

Since [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]], agent-to-agent orchestration via MCP adds a fourth pattern: cross-tool delegation where the "subagent" is an entirely different AI coding tool running its own context and reasoning.

Setup complexity is high for first-time configuration, and the pattern is still emerging. However, it represents a architectural direction where MCP becomes the standard interface for agent interoperability — not just tool access but agent-to-agent communication.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]] -- the parallelization framework this extends
- [[cross-agent memory bridges enable tool-agnostic knowledge persistence across multiple IDE platforms through shared storage]] -- the memory-layer analogue of cross-tool interop

Topics:
- [[agent-memory]]
