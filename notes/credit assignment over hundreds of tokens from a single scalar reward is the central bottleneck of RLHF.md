---
description: "The reward model provides one score for an entire generation — attributing that outcome to specific token-level decisions is the temporal credit assignment problem that shapes all RLHF algorithm design"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# Credit assignment over hundreds of tokens from a single scalar reward is the central bottleneck of RLHF

The reward model provides a single scalar score for an entire generated sequence, offering little insight into which token-level or span-level decisions were responsible for the outcome. A 200-token response gets one number — which tokens were responsible for the quality? This is the temporal credit assignment problem, one of the most fundamental challenges in RL.

Several approaches have emerged to address it, forming a spectrum from simple to complex:

- **Sequence-level (bandit) formulation** (REINFORCE, RLOO): Treats the entire response as one action. Avoids the problem entirely by not attempting per-token credit. Works well in practice because pretrained LLMs already have strong priors.
- **Token-level MDP** (PPO): Models each token as a separate action with its own advantage estimate via a learned value function. Theoretically richer but since [[sequence-level bandit formulation matches LLM outcome-reward settings better than per-token MDP modeling]], this complexity may be unnecessary.
- **Macro actions** (MA-RLHF): Groups tokens into higher-level constructs.
- **Shapley values** (SCAR): Distributes total reward among tokens based on marginal contributions from cooperative game theory.

The striking empirical finding is that the simplest approach — treating the whole response as one action — often works best, suggesting that pretrained LLMs carry enough structure that fine-grained credit assignment is redundant.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[sequence-level bandit formulation matches LLM outcome-reward settings better than per-token MDP modeling]] — the simplification that works
- [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]] — why simplicity wins

Topics:
- [[agent-cognition]]
- [[rl-alignment]]
