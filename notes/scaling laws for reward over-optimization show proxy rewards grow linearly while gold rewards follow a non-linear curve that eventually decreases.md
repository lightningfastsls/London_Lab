---
description: "Gao et al 2022 quantified the proxy-gold divergence: best-of-n R*=d(a-b*d), RL R*=d(a-b*log(d)) — larger policies overoptimize less, more RM data reduces Goodharting"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
  - "[[agent-governance]]"
---

# Scaling laws for reward over-optimization show proxy rewards grow linearly while gold rewards follow a non-linear curve that eventually decreases

Gao et al. (2022) used synthetic oracle rewards (6B parameter reward model as "gold") versus proxy reward models (3M-3B parameters) to quantify the dynamics of reward over-optimization:

- Proxy rewards grow approximately linearly with KL divergence distance from the reference policy — the further you optimize, the higher the proxy score climbs
- Gold (true) rewards follow non-linear curves that *eventually decrease*:
  - Best-of-n: `R*_bon(d) = d(alpha - beta * d)` — parabolic, peaks then falls
  - RL: `R*_RL(d) = d(alpha - beta * log(d))` — slower decline than best-of-n

Key findings with practical implications:
- **Larger policies** see less benefit from optimization but also overoptimize less — scale provides natural regularization
- **More reward model training data** reduces Goodharting — the proxy becomes a better approximation
- **KL penalty** effect "resembles early stopping" but since [[KL divergence penalty prevents reward model exploitation but paradoxically increases the proxy-gold reward gap]], it may be counterproductive

The existence of a peak in the gold reward curve means there is an optimal amount of RL fine-tuning beyond which continued training actively degrades quality even as the proxy score continues to improve. This is the quantitative foundation for early stopping strategies in RLHF.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]] — the qualitative framework these curves quantify
- [[KL divergence penalty prevents reward model exploitation but paradoxically increases the proxy-gold reward gap]] — the counterintuitive interaction
- [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]] -- LoRA may provide a natural mechanism for staying near the gold reward peak: its low-rank constraint limits how far the policy can deviate from the reference point, functioning as implicit early stopping

Topics:
- [[model-adaptation]]
- [[agent-governance]]
- [[rl-alignment]]
