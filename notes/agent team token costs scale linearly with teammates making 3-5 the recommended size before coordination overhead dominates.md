---
description: "Each teammate is a separate full Claude instance — beyond 5 teammates with 5-6 tasks each, coordination overhead (messaging, task management, conflict resolution) exceeds parallelization benefit"
type: finding
confidence: likely
created: 2026-03-02
meta_state: current
---

# agent team token costs scale linearly with teammates making 3-5 the recommended size before coordination overhead dominates

Each agent team teammate in Claude Code is a separate Claude instance with a full context window. Token costs scale linearly: 3 teammates cost 3x, 5 cost 5x. The recommended configuration is 3-5 teammates with 5-6 tasks per teammate.

Beyond this range, coordination overhead dominates. The overhead comes from: mailbox messaging between teammates consuming context, shared task list management requiring state synchronization, and file-locking coordination to prevent race conditions. This mirrors the Brooks's Law pattern from software engineering — adding more agents beyond a point slows rather than accelerates work.

Since [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]], the linear cost scaling of teams creates a direct budget trade-off: more teammates means less context budget per teammate. This is particularly relevant because since [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]], each teammate's effective reasoning capacity is already constrained.

The practical implication: agent teams are powerful for genuinely parallel work (code review with specialized lenses, frontend/backend/tests coordination) but expensive for tasks that could be handled sequentially by a single agent with subagent delegation.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]] -- the resource governance framework
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- effective context per teammate

Topics:
- [[agent-memory]]
