---
description: "The canonical RLHF architecture: (1) pretrain LM, (2) SFT on demonstrations, (3) train reward model from pairwise preferences via Bradley-Terry, (4) RL fine-tune with KL penalty"
type: method
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning

The standard RLHF pipeline consists of four stages:

**Stage 1 — Pretraining**: Start with a large pretrained LM that generates fluent text but has no alignment. OpenAI used GPT-3 variants; Anthropic used 10B-52B parameter models; DeepMind used 280B Gopher.

**Stage 2 — SFT**: Fine-tune on human-written demonstrations of desired behavior. This gives the model a better starting point for RL — since [[SFT suffers from exposure bias where teacher-forcing creates reliance on ground-truth context that degrades autoregressive generation]], SFT alone is insufficient.

**Stage 3 — Reward Model Training**: The key innovation. Generate multiple responses, have human annotators rank them pairwise, then train a model to predict scalar rewards matching those preferences. Since [[pairwise comparisons produce more reliable human preference data than absolute ratings because relative judgments avoid calibration problems]], the Bradley-Terry model converts rankings to a trainable loss. Scale: ~50k labeled preference samples, with varying architectures (OpenAI: 175B policy + 6B reward; Anthropic: matched sizes).

**Stage 4 — RL Fine-tuning**: Use PPO (or alternatives) to optimize the policy against the reward model, with a KL divergence penalty preventing drift from the reference model. This stage requires four models in memory simultaneously when using PPO, which since [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]] drove development of simpler methods.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[pairwise comparisons produce more reliable human preference data than absolute ratings because relative judgments avoid calibration problems]] — Stage 3 methodology
- [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]] — Stage 4 constraint
- [[SFT suffers from exposure bias where teacher-forcing creates reliance on ground-truth context that degrades autoregressive generation]] — why Stage 2 alone is insufficient

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
