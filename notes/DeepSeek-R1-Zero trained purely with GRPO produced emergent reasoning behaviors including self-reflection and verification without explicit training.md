---
description: "Pure RL without SFT, using only correctness-based verifiable rewards — thinking time increased naturally during training as an intrinsic development, not from external adjustments"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
  - "[[agent-cognition]]"
---

# DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training

DeepSeek-R1-Zero was trained purely with GRPO, without supervised fine-tuning. The reward signal was based only on correctness of final predictions — no process reward, no intermediate feedback, no demonstrations of reasoning. Training hyperparameters: learning rate 3e-6, KL coefficient 0.001, clip ratio 10, temperature 1.0, 16 outputs sampled per question, max length 32,768 tokens.

The striking result: emergent reasoning behaviors appeared without being explicitly trained — self-reflection ("wait, let me reconsider"), verification ("let me check this step"), and dynamic strategy adaptation. Thinking time increased naturally during training as an intrinsic development, not from external adjustments. The model spontaneously developed chain-of-thought reasoning through pure RL optimization on outcome correctness.

This finding is significant for several reasons. First, it suggests that complex cognitive behaviors like self-reflection can emerge from simple reward signals given sufficient model capacity. Second, it demonstrates that since [[verifiable rewards bypass learned reward models entirely avoiding reward hacking for math and code tasks]], clean rewards may be more important than sophisticated RL algorithms. Third, DeepSeek deliberately avoided neural reward models due to reward hacking concerns at scale — the emergent capabilities appeared specifically in the context of clean, verifiable rewards.

The connection to the ICL research is notable: since [[induction heads emerge in a sharp phase transition during training that coincides with the onset of in-context learning ability supported by six causal lines of evidence]], both ICL and reasoning-via-RL appear to emerge as phase transitions rather than gradual improvements.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[GRPO eliminates the critic network through group-relative advantage scoring achieving 50 percent memory reduction over PPO]] — the algorithm used
- [[verifiable rewards bypass learned reward models entirely avoiding reward hacking for math and code tasks]] — the reward design choice
- [[induction heads emerge in a sharp phase transition during training that coincides with the onset of in-context learning ability supported by six causal lines of evidence]] — parallel emergence phenomenon

Topics:
- [[model-adaptation]]
- [[agent-cognition]]
- [[rl-alignment]]
- [[transformer-architecture]]
