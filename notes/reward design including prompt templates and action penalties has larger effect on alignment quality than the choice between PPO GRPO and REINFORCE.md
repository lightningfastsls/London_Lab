---
description: "Search-R1 showed Fast vs Slow Thinking templates, F1 vs F1+ reward functions, and action penalties each had bigger effects than switching algorithms — reward engineering dominates algorithm engineering"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
  - "[[agent-governance]]"
---

# Reward design including prompt templates and action penalties has larger effect on alignment quality than the choice between PPO GRPO and REINFORCE

The Search-R1 paper provides the most direct evidence that reward engineering dominates algorithm engineering in RLHF:

- **Prompt template effect** (Fast vs. Slow Thinking): 0.422 vs. 0.403 avg on Qwen2.5-7B — since [[Slow Thinking training templates are prone to self-reinforcing reasoning collapse where think tag frequency correlates with reward creating positive feedback loops]], the template choice caused training collapse, a qualitative failure no algorithm change could fix
- **Reward function effect** (F1 vs. F1+): 0.391 vs. 0.429 avg — since [[F1-based reward training causes answer avoidance where the policy learns never answering is safer than risking wrong answers]], two small penalty terms (alpha=beta=0.1) transformed a failing reward into one that surpassed the EM baseline
- **Algorithm effect** (REINFORCE vs. PPO vs. GRPO): 0.437 vs. 0.422 vs. 0.433 — a 1.5% spread

The reward function and prompt template each produced larger quality differences than the algorithm choice. The best configuration (Fast Thinking + REINFORCE + F1+) combined the right framing decisions.

This meta-finding has profound implications for alignment research: the field may have over-invested in algorithm innovation (PPO → GRPO → REINFORCE++) relative to reward design. The "simplicity premium" — since [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]] — suggests the algorithm is the wrong variable to optimize. The reward signal is the bottleneck.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[Search-R1 found REINFORCE outperformed both PPO and GRPO for agentic deep research tasks with the highest accuracy and most efficient search strategies]] — the specific algorithm comparison
- [[Slow Thinking training templates are prone to self-reinforcing reasoning collapse where think tag frequency correlates with reward creating positive feedback loops]] — the template effect
- [[F1-based reward training causes answer avoidance where the policy learns never answering is safer than risking wrong answers]] — the reward function effect

Topics:
- [[model-adaptation]]
- [[agent-governance]]
- [[rl-alignment]]
