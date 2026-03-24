---
description: "Ahmadian et al 2024 showed modeling the full generation as a single action preserves performance and speeds learning — per-token MDP is unnecessary complexity for outcome rewards"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[model-adaptation]]"
---

# Sequence-level bandit formulation matches LLM outcome-reward settings better than per-token MDP modeling

Research demonstrates that "modeling the full generation as a single action preserves the LLM's performance and even speeds up learning, indicating that formulating each token as its own action is an unnecessary complexity in an outcome reward setting" (Ahmadian et al., 2024).

This is counterintuitive — PPO's per-token advantage estimation via a learned critic seems theoretically richer than REINFORCE's blunt "entire response = one action" formulation. But in practice, pretrained LLMs already have strong priors from billions of tokens of pretraining. The effective action space at any given step is small — probability mass is highly concentrated among a few plausible tokens. Large off-policy updates are rare and non-catastrophic in this regime.

The implication extends beyond algorithm choice: it suggests that the credit assignment problem in RLHF, while theoretically deep, is practically solved by the LLM's existing knowledge. The model already "knows" which tokens matter because of pretraining — RL just needs to nudge the overall distribution, not micromanage individual token choices. This explains why since [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]], the simplest algorithms achieve the best results.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[credit assignment over hundreds of tokens from a single scalar reward is the central bottleneck of RLHF]] — the problem this finding addresses
- [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]] — the practical consequence
- [[REINFORCE Leave-One-Out uses 50-70 percent less memory than PPO while consistently outperforming it on alignment tasks]] — the empirical validation

Topics:
- [[agent-cognition]]
- [[model-adaptation]]
- [[rl-alignment]]
