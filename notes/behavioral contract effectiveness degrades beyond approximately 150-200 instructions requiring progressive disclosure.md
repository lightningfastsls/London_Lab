---
description: "Community convergence on CLAUDE.md size ceiling — WHAT/WHY/HOW framework, move domain rules to skills/files, use linters for style, lifecycle hooks for non-optional compliance"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure

The CLAUDE.md file has emerged as the primary vehicle for practitioner-level behavioral contracts in Claude Code ecosystems. Community best practices converge on keeping contracts under approximately 150-200 instructions for reasonable instruction-following quality. Beyond this threshold, agents begin missing or misinterpreting specific rules as the contract competes with the task for context window attention.

The recommended structure follows a WHAT/WHY/HOW framework: what (technology and codebase), why (purpose and constraints), how (workflow and approval process). Domain-specific rules should be moved to separate files or skills for progressive disclosure — loaded only when relevant rather than occupying permanent context. Style guidance should be offloaded to linters and formatters (deterministic tools) rather than consuming instruction budget. Lifecycle hooks enforce non-optional compliance: pre-write formatting, post-write testing, session-start orientation.

This finding connects directly to context window research: since [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]], contract instructions compete for the same limited effective capacity as task context. A 200-line contract at session start consumes effective context that the task needs later. Progressive disclosure solves this by loading contract sections just-in-time, which is essentially since [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] applied to behavioral specifications.

The design implication: contracts should be structured as layered specifications — a concise core in CLAUDE.md with detailed rules in referenced files, loaded on demand through skills or hooks. Structural constraints like since [[one-feature-per-session constraint prevents scope creep and enables clean validation in long-running agent harnesses]] reduce the burden on the contract itself — when scope is structurally bounded, fewer instructions are needed to govern it. Similarly, since [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]], offloading style enforcement to linters directly frees instruction budget for the behavioral and reasoning constraints that only prompt-level contracts can express.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the context constraint that limits contract size
- [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] -- the progressive disclosure principle
- [[tiered behavioral contracts must scale with project complexity because instruction-following degrades with instruction count]] -- Vass's tiering approach to the same problem
- [[one-feature-per-session constraint prevents scope creep and enables clean validation in long-running agent harnesses]] -- structural scope bounds reduce contract instruction burden
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- offloading style to tools frees instruction budget

Topics:
- [[agent-governance]]
