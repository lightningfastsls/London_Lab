---
description: "Models trained with teacher forcing see ground-truth prefixes during training but their own imperfect outputs during inference — this train-test mismatch motivates RL fine-tuning"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[model-adaptation]]"
---

# SFT suffers from exposure bias where teacher-forcing creates reliance on ground-truth context that degrades autoregressive generation

Supervised fine-tuning (SFT) on curated demonstrations improves instruction following but has a fundamental limitation: during training, the model sees the correct previous tokens (teacher forcing), but during inference it sees its own potentially incorrect outputs. This train-test mismatch — exposure bias — means SFT models become overly reliant on ground-truth context and struggle when their own generation diverges from the training distribution.

SFT also cannot exhaustively cover subtle needs. The demonstration set is finite, so models still fabricate facts, produce biased content, and fail on edge cases not represented in training data. Since [[RL is needed for LLM alignment because no differentiable loss function captures the multi-dimensional quality of human preference judgments]], RL fine-tuning addresses exposure bias by training the model on its own generated outputs, evaluated by a reward signal — the model learns from its actual behavior, not idealized demonstrations.

This has a parallel to the LoRA/ICL distinction: since [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]], both SFT and RL operate on model weights but with fundamentally different feedback signals — SFT uses demonstration loss, RL uses preference reward.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[RL is needed for LLM alignment because no differentiable loss function captures the multi-dimensional quality of human preference judgments]] — the broader motivation for RL
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] — parallel between SFT and RL weight updates
- [[multi-epoch LoRA training on static instruction data causes overfitting and capability degradation]] -- exposure bias compounds with LoRA overfitting: static SFT data with multiple LoRA epochs amplifies the teacher-forcing distribution mismatch
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- curated data partially mitigates exposure bias by ensuring high-quality demonstrations that more closely match the distribution the model will encounter at inference

Topics:
- [[agent-cognition]]
- [[model-adaptation]]
- [[rl-alignment]]
