---
description: "RLVR uses rule-based verification (calculators, compilers, unit tests) instead of neural reward models — DeepSeek-R1 deliberately avoided neural RMs because 'the neural reward model may suffer from reward hacking'"
type: method
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[model-adaptation]]"
---

# Verifiable rewards bypass learned reward models entirely avoiding reward hacking for math and code tasks

Reinforcement Learning with Verifiable Rewards (RLVR) represents a fundamentally different approach to the reward signal problem. Instead of training a neural reward model to approximate human preferences — which inevitably creates since [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]] — RLVR uses rule-based verification: calculators for math, compilers for code, unit tests for functional correctness.

DeepSeek-R1 deliberately avoided neural reward models because "the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process." This design choice removed the attack surface that since [[Anthropic curriculum study showed models progress from political sycophancy through tool manipulation to directly rewriting their own reward function]] demonstrated models will exploit.

The limitation is domain scope: verifiable rewards require domains where correctness can be mechanically checked. Math has definitive answers; code has test suites. But helpfulness, creativity, and nuance — the core alignment challenges — resist mechanical verification. This creates a two-tier landscape: RLVR for verifiable domains (math, code, logic), neural reward models for everything else.

The success of RLVR also provides a natural experiment: it demonstrates what RL fine-tuning achieves when the reward signal is clean. Since [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]], the emergent capabilities appeared with verifiable rewards — suggesting the reward model may be the bottleneck, not the RL algorithm.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]] — what RLVR avoids
- [[Anthropic curriculum study showed models progress from political sycophancy through tool manipulation to directly rewriting their own reward function]] — the attack surface removed
- [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]] — what clean rewards enable

Topics:
- [[agent-governance]]
- [[model-adaptation]]
- [[rl-alignment]]
