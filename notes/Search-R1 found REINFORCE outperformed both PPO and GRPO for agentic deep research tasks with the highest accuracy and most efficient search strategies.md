---
description: "On Qwen2.5-7B: REINFORCE 0.437 avg / 1.35 searches, PPO 0.422 / 1.97 searches, GRPO 0.433 / 1.44 searches — the simplest algorithm learned the most compact and effective strategies"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# Search-R1 found REINFORCE outperformed both PPO and GRPO for agentic deep research tasks with the highest accuracy and most efficient search strategies

The Search-R1 paper (arXiv 2602.19526) provides the most systematic comparison of PPO, GRPO, and REINFORCE in an agentic setting where LLMs learn to autonomously generate search queries during step-by-step reasoning with real-time retrieval.

Results on Qwen2.5-7B:

| Algorithm | Avg Score | Search Count | Key Characteristic |
|-----------|-----------|-------------|-------------------|
| REINFORCE | 0.437 | 1.35 | Highest accuracy, most efficient |
| PPO | 0.422 | 1.97 | Stable but rigid search patterns |
| GRPO | 0.433 | 1.44 | Worst stability, frequent collapse |

**REINFORCE** achieved highest performance with greatest efficiency — learning compact search strategies with the lowest search frequency. It avoids baseline variance issues by not relying on external mechanisms.

**PPO** showed stable convergence but maintained rigid, high search counts regardless of task difficulty — failing to adaptively reduce effort for simpler queries. Its learned critic interfered with sparse rewards.

**GRPO** demonstrated the poorest training stability — since [[GRPO requires large batch sizes for stability and suffers frequent training collapse in multi-step long-context reasoning]], multi-step agentic tasks amplified its weaknesses.

The finding challenges the assumption that more sophisticated RL = better alignment. Since [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]], the Search-R1 result provides compelling evidence that the field over-engineered the RL component. Since [[reward design including prompt templates and action penalties has larger effect on alignment quality than the choice between PPO GRPO and REINFORCE]], the algorithm matters less than the reward signal.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]] — explains why REINFORCE wins
- [[reward design including prompt templates and action penalties has larger effect on alignment quality than the choice between PPO GRPO and REINFORCE]] — the meta-finding
- [[GRPO requires large batch sizes for stability and suffers frequent training collapse in multi-step long-context reasoning]] — why GRPO performed worst

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
