---
description: "Each subagent uses tens of thousands of tokens internally but returns 1-2K summaries — endorsed by Anthropic, Huntley/Sourcegraph, implemented in Claude Code and Amp"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward

Splitting work among subagents — each with their own focused context window — and compressing findings back to a lead agent is the primary architectural pattern for scaling AI agent work beyond single-context limits. Each subagent uses tens of thousands of tokens internally but returns only 1,000-2,000 token summaries. This exploits the principle that since [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]], each subagent can operate within its comfortable context range for its specific task type.

The pattern was originally proposed by Geoffrey Huntley at Sourcegraph and is now endorsed by Anthropic's context engineering guidance and implemented in Claude Code, Amp, and other agent tools. The effectiveness comes from two sources: it avoids the context window degradation caused by accumulating all information in one window (since [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]]), and it avoids multi-turn degradation by giving each subagent a fresh context (since [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]]).

A concrete production example: Claude Code's review command spawns 9 parallel subagents via the Task tool, each focused on a specific quality dimension — test runner, linter/static analysis, code reviewer, security reviewer, quality/style reviewer, test quality reviewer, performance reviewer, dependency/deployment safety reviewer, and simplification/maintainability reviewer. Results are aggregated into issues vs suggestions, ranked by severity. One practitioner reported suggestions are ~75% useful with this 9-agent decomposition versus <50% with simpler single-agent review. A known limitation: subagent Write tool may not persist files to disk, requiring the main session to handle file writing — read-only subagents that analyze without modifying are the recommended pattern.

The pattern does have a cost: compression loses information. The lead agent receives summaries, not full reasoning traces. If a subagent's summary omits a critical detail, the lead agent cannot recover it without re-running the subagent. This creates a design challenge around what to summarize and what to preserve.

## Blackboard Architecture as Variant

The Liza system (Vass, 2026) demonstrates an alternative coordination mechanism: instead of summary compression, agents read and write a shared state file (blackboard architecture). Since [[Liza blackboard architecture coordinates multi-agent work through shared state files without inter-agent conversation]], the blackboard replaces summary compression with explicit structured state — agents read only what they need rather than receiving compressed summaries. The Agent Contracts framework (Ye and Tan, 2026) provides formal grounding through since [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]], ensuring multi-agent delegation cannot cascade into unbounded resource consumption.

This vault uses the subagent pattern throughout its processing pipeline — /reduce, /reflect, /reweave each operate as focused tasks with fresh context, passing structured outputs (notes, enrichments, connections) rather than accumulated conversation.

As of March 2026, Claude Code now offers since [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]] — subagents remain the lightweight option (report-only, no inter-agent communication), while Agent Teams add full instances with shared task lists, mailbox messaging, and autonomous coordination. Subagents now also support worktree isolation for filesystem safety, making the original summary-compression pattern composable with filesystem isolation.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the principle subagent isolation exploits
- [[context window and multi-turn degradation have distinct root causes but compound when both occur in long multi-turn sessions]] -- both failure modes this pattern mitigates
- [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] -- the complementary pattern
- [[Liza blackboard architecture coordinates multi-agent work through shared state files without inter-agent conversation]] -- shared-state variant of subagent coordination
- [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]] -- formal resource bounds for delegation
- [[cross-agent knowledge transfer requires flattening graph-traversable constraints into self-contained plain text]] — the knowledge transfer problem at delegation boundaries: subagent summary compression preserves task findings, but constraint flattening preserves architectural invariants that the receiving agent cannot discover independently
- [[Letta sleep-time compute pairs a primary agent with a sleep-time agent that processes memory during idle periods]] — extends subagent isolation from task-level to session-level: the primary agent operates within a session, the sleep-time agent operates across sessions, applying the same focused-window principle to memory consolidation

Topics:
- [[agent-cognition]]
