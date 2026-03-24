---
description: "MIT Technology Review argues rules succeed at boundary not prompt — but prompt-level contracts address behavioral and reasoning constraints that type systems cannot express"
type: tension
status: pending
created: 2026-03-01
topics:
  - "[[agent-governance]]"
---

# Prompt-level versus boundary-level enforcement represents competing philosophies for constraining agent behavior

## The Tension

MIT Technology Review (2026) argues "rules fail at the prompt, succeed at the boundary" — external enforcement through type systems, linters, and test suites is structurally more robust than prompt-based behavioral instructions. This directly challenges the premise of prompt-level behavioral contracts (CLAUDE.md, ABC contracts, Vass state machine). If boundary enforcement is superior, why invest in prompt-level contracts at all?

## Quick Test

Can this specific constraint be expressed as a deterministic, boundary-level check?

## When Each Pole Wins

**Boundary enforcement wins** for: formatting rules, type constraints, import restrictions, structural patterns, test requirements, dependency limits — anything expressible as a deterministic check.

**Prompt enforcement wins** for: "explain your reasoning before executing," "one logical change per approval," "surface uncertainty rather than fabricate," "prioritize user learning" — behavioral and metacognitive constraints requiring judgment.

## Dissolution Attempts

Since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]], the pragmatic dissolution is a layered model: use boundary enforcement for everything expressible as a rule, and prompt enforcement for everything else. Since [[contract visibility improves natural compliance even before enforcement the transparency effect]], prompt-level contracts still provide value even for rules that cannot be enforced.

## Practical Applications

This vault's own contract uses both: lifecycle hooks (boundary enforcement for formatting, schema validation, session capture) combined with CLAUDE.md behavioral instructions (approval workflow, struggle protocol, learning priority). The question is whether the prompt-level behavioral instructions actually change behavior, or whether the boundary-level hooks do all the real work.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01

Topics:
- [[agent-governance]]
