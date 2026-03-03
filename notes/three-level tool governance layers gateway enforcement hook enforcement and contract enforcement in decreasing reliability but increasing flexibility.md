---
description: "Gateways control what agents CAN access (deterministic), hooks enforce what agents SHOULD do (code-enforced), contracts guide what agents WILL do (prompt-dependent) — effective governance layers all three"
type: pattern
confidence: likely
created: 2026-03-02
meta_state: current
---

# three-level tool governance layers gateway enforcement hook enforcement and contract enforcement in decreasing reliability but increasing flexibility

The interaction between MCP tool selection and agent governance operates at three levels, each with a distinct reliability-flexibility trade-off:

**1. Gateway level (most reliable, least flexible)** — MCP gateways control what agents CAN access. Role-based or attribute-based access controls restrict tool availability. Deterministic: if a tool is not available, the agent cannot use it. No prompt engineering can bypass a gateway restriction.

**2. Hook level (code-enforced, moderate flexibility)** — Claude Code hooks enforce what agents SHOULD do at runtime. PreToolUse hooks can block, modify, or escalate tool calls. PostToolUse hooks validate results. This is boundary-level enforcement — code runs outside the agent's context window, immune to attention degradation.

**3. Contract level (least reliable, most flexible)** — CLAUDE.md behavioral contracts guide what agents WILL do through prompt-level instructions. Since [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]], contracts have a ceiling. But they benefit from the transparency effect where since [[contract visibility improves natural compliance even before enforcement the transparency effect]], even imperfectly enforced contracts provide value.

The convergent pattern: effective governance layers all three. Since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]], the reliability gradient (gateway > hooks > contracts) suggests that critical constraints should be enforced at the gateway or hook level, with contracts handling judgment-dependent guidance.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- the theoretical foundation for the reliability gradient
- [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]] -- why contracts are the least reliable layer
- [[contract visibility improves natural compliance even before enforcement the transparency effect]] -- why contracts still provide value despite lower reliability

Topics:
- [[agent-governance]]
