---
description: "Architecture is typically a fine-tuned LM with scalar output head — OpenAI used 175B policy + 6B reward model, Anthropic matched sizes at 10B-52B"
type: method
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# Reward model training uses Bradley-Terry pairwise comparison on approximately 50k labeled preference samples

The reward model is the core innovation of RLHF — it translates human preferences into a differentiable signal. The architecture is typically a fine-tuned language model with a scalar output head replacing the token prediction head. Training uses the Bradley-Terry model, a well-established statistical framework for ranking from pairwise comparisons.

The training process: generate multiple responses to prompts, have human annotators rank them pairwise ("A is better than B"), then train the model to predict scalar rewards matching those preferences. Scale varies: approximately 50k labeled preference samples is typical, though dataset size significantly affects reward model quality — since [[scaling laws for reward over-optimization show proxy rewards grow linearly while gold rewards follow a non-linear curve that eventually decreases]], more reward model training data reduces Goodharting.

Architecture choices vary by lab: OpenAI used a 175B policy with a smaller 6B reward model; Anthropic matched policy and reward model sizes at 10B-52B. The size asymmetry question is unresolved — smaller reward models are cheaper but may miss nuanced preferences that larger models capture.

The reward model's imperfection is the source of all downstream reward hacking problems — since [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]], optimizing against an imperfect proxy inevitably creates exploitable gaps.

---

Source: [[rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02]]

Relevant Notes:
- [[pairwise comparisons produce more reliable human preference data than absolute ratings because relative judgments avoid calibration problems]] — why pairwise, not absolute
- [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]] — the downstream consequence of imperfect reward models
- [[scaling laws for reward over-optimization show proxy rewards grow linearly while gold rewards follow a non-linear curve that eventually decreases]] — how reward model quality affects alignment

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
