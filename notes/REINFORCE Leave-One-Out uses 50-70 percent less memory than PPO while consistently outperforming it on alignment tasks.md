---
description: "RLOO samples K completions per prompt, computes each baseline from other completions excluding itself — win-rate: +10.4% TL;DR, +14.5% HH, +32.1% Llama-7B (Ahmadian et al 2024)"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# REINFORCE Leave-One-Out uses 50-70 percent less memory than PPO while consistently outperforming it on alignment tasks

RLOO improves on standard REINFORCE through a variance reduction technique: sample K completions per prompt and compute each completion's baseline using rewards from the other completions (excluding itself): `baseline_i = mean(rewards_{j != i})`.

This "lowers variance relative to standard REINFORCE by using multiple samples per prompt" without requiring a learned value function. The leave-one-out baseline gives each completion an unbiased comparison point using naturally available information from the same batch.

Empirical results demonstrate RLOO's superiority over PPO:
- 50-70% less memory usage (3 models vs 4)
- 2-3x faster training
- Win-rate improvements: +10.4% on TL;DR summarization, +14.5% on HH (helpful/harmless) dataset, +32.1% on Llama-7B (Ahmadian et al., 2024)

Since [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]], RLOO achieves the critic-elimination goal while also improving quality. The combination of better results, lower memory, and faster training makes RLOO arguably the strongest simple baseline for RLHF.

Since [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]], RLOO succeeds because it addresses REINFORCE's main weakness (variance) without reintroducing PPO's complexity (learned critic, per-token advantages).

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]] — the theoretical foundation
- [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]] — the problem RLOO solves
- [[REINFORCE++ bridges REINFORCE simplicity with PPO stability via token-level KL penalty and ratio clipping achieving 30 percent training time reduction]] — the further refinement

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
