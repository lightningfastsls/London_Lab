---
description: "More comprehensive contracts provide better governance but degrade instruction-following quality — no formal theory for optimal contract size exists"
type: tension
status: pending
created: 2026-03-01
topics:
  - "[[agent-governance]]"
---

# Contract comprehensiveness versus instruction-following quality creates a fundamental scaling tension

## The Tension

More comprehensive behavioral contracts provide better governance — more edge cases covered, more failure modes addressed, more nuanced behavior specified. But instruction-following quality degrades uniformly as instruction count increases. Since [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]], there is a practical ceiling beyond which adding more rules actually reduces overall compliance.

## Quick Test

Does adding this rule to the contract improve net compliance (including the degradation from increased contract size)?

## When Each Pole Wins

**Comprehensiveness wins** when: the failure mode being prevented is catastrophic (data loss, security breach), the rule is simple to evaluate, or the project complexity warrants the overhead.

**Simplicity wins** when: the project is small, the failure modes are recoverable, or the agent already demonstrates natural compliance through the transparency effect.

## Dissolution Attempts

Vass's tiered contracts (full ~200, medium ~50, minimal ~30) attempt dissolution by matching contract size to project complexity. Progressive disclosure (core rules in CLAUDE.md, domain rules in skills) attempts dissolution by deferring complexity. Neither fully resolves the tension — even progressive disclosure consumes context when the deferred rules are loaded.

## Practical Applications

This tension is live in our own CLAUDE.md. Every rule added to govern agent behavior competes for the same finite context window and instruction-following capacity that the task itself needs.

---

Source: [[behavioral-contracts-ai-coding-agents-research-2026-03-01]]

Topics:
- [[agent-governance]]
