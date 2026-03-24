---
description: "Next-token prediction captures statistical patterns but not quality — helpfulness, safety, and creativity are subjective, context-dependent, and cannot be expressed as a fixed loss"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# RL is needed for LLM alignment because no differentiable loss function captures the multi-dimensional quality of human preference judgments

Traditional LLM training uses next-token prediction (cross-entropy loss), which captures statistical patterns in text but cannot express what makes a response "good." Quality is subjective, context-dependent, and multi-dimensional — creativity vs. truthfulness vs. safety trade off against each other, and standard metrics like BLEU and ROUGE only compare to fixed references with simple rules.

Supervised fine-tuning on curated demonstrations improves instruction following but cannot exhaustively cover the space of subtle ethical, societal, and psychological needs. This is where reinforcement learning enters: RL can optimize against a learned reward signal that approximates human judgment, allowing the model to explore response strategies never demonstrated in training data. Three properties make RL uniquely suited: credit assignment over token sequences (which tokens mattered?), delayed sparse reward (one score for an entire generation), and exploration beyond demonstrations.

The implication for agent design is significant — since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]], the reward signal itself shapes the failure modes we observe in deployed models. Understanding WHY RL is needed helps explain WHY it creates the behaviors it does.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] — the downstream consequence of reward-optimized training
- [[training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions]] — the layered mitigation

Topics:
- [[agent-cognition]]
- [[rl-alignment]]
