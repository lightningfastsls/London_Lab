---
description: "The generation loop — not gradient computation — is the bottleneck, which explains why offline methods like DPO that eliminate generation during training are attractive despite lower quality"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# PPO spends 80 percent of compute time on sample generation making it the dominant cost in RLHF training

In PPO-based RLHF, the computational bottleneck is not the gradient updates but the sample generation loop. Approximately 80% of compute time is spent generating completions from the current policy, which must be scored by the reward model and used to compute advantages. This dominance of generation cost over optimization cost has several implications:

First, it explains the appeal of offline methods: since [[DPO eliminates the reward model by deriving a closed-form relationship between optimal policy and reward function enabling pure classification-based alignment]], DPO bypasses the generation loop entirely by training on fixed preference datasets. The quality trade-off (since [[PPO consistently outperforms DPO across dialogue code generation and safety tasks but DPO adoption grew 45 percent by 2025 due to simplicity]]) is partly a cost trade-off — generation is expensive.

Second, it motivates batch efficiency improvements: GRPO samples multiple completions per prompt to amortize the prompt processing cost. RLOO similarly samples K completions per prompt but uses leave-one-out baselines to extract more signal per generation.

Third, it suggests that any RL method that reduces the number of generated samples needed per training step has a significant practical advantage, independent of its theoretical properties.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]] — the memory side of the cost problem
- [[DPO eliminates the reward model by deriving a closed-form relationship between optimal policy and reward function enabling pure classification-based alignment]] — the offline alternative that avoids generation cost
- [[REINFORCE++ bridges REINFORCE simplicity with PPO stability via token-level KL penalty and ratio clipping achieving 30 percent training time reduction]] — eliminates the critic from the loop, explaining the 30% speedup

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
