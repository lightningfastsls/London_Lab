---
description: "LLM agents without tasks show model-specific determinism — GPT-5/O3 exclusively pursue production, Claude Opus engages philosophical inquiry — implying behavioral contracts have architecture-dependent effects"
type: finding
confidence: experimental
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Different model architectures exhibit distinct unconstrained behavioral patterns suggesting contracts interact differently across model families

Research examining LLM agents left without tasks found three emergent behavioral patterns: (1) systematic production — self-imposed projects treating autonomy as project management; (2) methodological self-inquiry — designing falsifiable experiments about own cognition; (3) recursive conceptualization — building philosophical frameworks integrating constraints. Critically, these patterns were model-specific and deterministic: GPT-5/O3 exclusively pursued production, while Claude Opus consistently engaged philosophical inquiry.

This model-specific determinism suggests that different architectures have deeply embedded response patterns that shape how they interact with behavioral contracts. A contract designed for production-oriented behavior (task lists, progress tracking, completion criteria) may work naturally with GPT-5 but create friction with Claude Opus. Conversely, contracts emphasizing reasoning externalization and learning-first priority may align better with Claude's default patterns.

The implication for contract design is that one-size-fits-all behavioral contracts may be suboptimal. Since [[contract visibility improves natural compliance even before enforcement the transparency effect]], contracts that align with the model's natural behavioral tendencies would amplify the transparency effect, while contracts that fight against natural tendencies would require stronger enforcement to maintain compliance. This connects to the SELAUR framework's uncertainty-aware self-regulation — using uncertainty as a natural measure of model confidence, where high uncertainty should trigger alternative strategies rather than forced continuation.

This is an experimental finding — the research is preliminary and the sample of model families is small. But if confirmed, it would mean behavioral contracts need model-specific calibration, not just task-specific tiering.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[contract visibility improves natural compliance even before enforcement the transparency effect]] -- the visibility effect that model alignment would amplify
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the training that creates model-specific patterns
- [[tiered behavioral contracts must scale with project complexity because instruction-following degrades with instruction count]] -- contract adaptation at a different axis
- [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]] -- drift rate alpha likely varies by model, requiring model-specific recovery calibration
- [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]] -- the size ceiling may shift based on model architecture

Topics:
- [[agent-governance]]
