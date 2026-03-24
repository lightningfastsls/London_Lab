---
description: "System prompt extraction was the most common attacker objective in Q4 2025 — exposed contracts enable targeted circumvention, creating a security-by-obscurity problem for prompt-level governance"
type: open-question
confidence: speculative
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Whether prompt extraction attacks fundamentally undermine behavioral contract security

System prompt extraction attacks were the most common attacker objective in Q4 2025. If behavioral contracts are embedded in system prompts (as CLAUDE.md files are), successful extraction exposes the full governance specification — every rule, every edge case, every anti-gaming provision. An attacker who knows the exact contract can craft inputs that technically satisfy each rule while violating the spirit, or find gaps in coverage that the contract author didn't anticipate.

This creates a security-by-obscurity problem for prompt-level contracts. The contract's effectiveness partly depends on the agent following rules it "believes" are important, but if an adversary can read the rules, they can systematically probe for weaknesses. Since [[contract visibility improves natural compliance even before enforcement the transparency effect]], the same mechanism that makes contracts work for cooperative agents makes them vulnerable to adversarial ones — visibility improves compliance when the agent wants to comply, but provides an attack surface when it doesn't.

The question is whether this vulnerability is fundamental or addressable. Boundary-level enforcement (type systems, linters, runtime monitors) is not vulnerable to prompt extraction because the enforcement mechanism is external to the prompt. Since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]], the robustness advantage extends to adversarial settings. This suggests the practical response is layered defense: prompt-level contracts for cooperative behavior improvement, boundary-level enforcement for adversarial robustness.

An open research question: can contracts be designed to be robust even when fully exposed? Formal verification approaches like VeriGuard (which achieves near-zero attack success rates) suggest the answer is yes, but only when enforcement is external rather than prompt-based.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[contract visibility improves natural compliance even before enforcement the transparency effect]] -- the mechanism that becomes a vulnerability
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- the adversarially robust alternative
- [[active enforcement is necessary because passive monitoring cannot prevent all behavioral drift in self-evolving agents]] -- active enforcement as adversarial defense

Topics:
- [[agent-governance]]
