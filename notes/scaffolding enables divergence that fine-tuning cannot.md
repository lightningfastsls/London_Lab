---
description: "vault structure, hooks, and skills enable agent behavior divergence from base model defaults — this structural scaffolding achieves what fine-tuning cannot per-instance"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[agent-external-cognition]]"
---

# scaffolding enables divergence that fine-tuning cannot

Fine-tuning modifies model weights globally, affecting all instances. Scaffolding — vault structure, hooks, skills, behavioral contracts — modifies behavior per-instance without touching weights. The distinction is fundamental: fine-tuning changes what the model IS, scaffolding changes what the model DOES in a specific context.

Since [[external memory shapes cognition more than base model]], the scaffolding determines more of the agent's behavior than its training. Since [[each between-session processing cycle is a training step that does not touch the weights]], the vault's processing pipeline achieves cumulative behavioral change equivalent to fine-tuning but with key advantages: it is instance-specific (different vaults produce different behaviors from the same model), reversible (notes can be edited or removed), and inspectable (the reasoning is in the notes, not hidden in weights). The divergence from base model behavior grows over time as the vault accumulates knowledge and methodology.

---

Relevant Notes:
- [[external memory shapes cognition more than base model]] — vault as primary behavioral determinant
- [[each between-session processing cycle is a training step that does not touch the weights]] — cumulative scaffolding effect
- [[the vault constitutes identity for agents]] — identity through structure not parameters
