---
description: "Vass defines three tiers — full (~200 lines), medium (~50), minimal (~30) — acknowledging that contract comprehensiveness trades off against compliance quality"
type: pattern
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Tiered behavioral contracts must scale with project complexity because instruction-following degrades with instruction count

Vass describes three tiers of contract complexity: full (~200 lines) with complete state machine, approval format, struggle protocol, test integrity rules, and learning mode for substantial projects; medium (~50 lines) with core rules and approval gates for moderate projects; and minimal (~30 lines) with essential constraints only for experimental work.

The tiering acknowledges a fundamental tension: instruction-following quality degrades uniformly as instruction count increases. A 200-line contract provides comprehensive governance but may cause the agent to miss or misinterpret specific rules. A 30-line contract is almost fully followed but cannot cover edge cases. The community has converged on ~150-200 instructions as the practical ceiling for reasonable instruction-following quality in CLAUDE.md files, which means since [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]], progressive disclosure becomes essential — moving domain-specific rules to separate files or skills loaded on demand.

The deeper principle is that contracts face the same information overload challenge as context windows: since [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]], packing instructions to capacity is counterproductive. The contract must leave cognitive headroom for the actual task.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]] -- the empirical ceiling
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the analogous context window constraint
- [[contract comprehensiveness versus instruction-following quality creates a fundamental scaling tension]] -- the tension this tiering addresses

Topics:
- [[agent-governance]]
