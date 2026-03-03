---
description: "DeepSeekMath used batch size 1024 (16 prompts x 64 completions) — small groups yield unreliable gradient estimates because advantages are purely relative with no learned baseline"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# GRPO requires large batch sizes for stability and suffers frequent training collapse in multi-step long-context reasoning

GRPO's group-relative advantage computation — `(reward_i - mean) / std` within each batch — creates a dependency on batch statistics that PPO's learned value function avoids. Small groups yield unreliable gradient estimates because the baseline (group mean) is noisy. Single policy updates per batch are common, contrasting with PPO's multiple epochs per batch.

The Search-R1 paper provided the most direct evidence of GRPO's instability: in the head-to-head comparison of PPO, GRPO, and REINFORCE for deep research tasks, GRPO showed "inferior robustness" and "frequently suffered from training collapse." In multi-step, long-context reasoning where the model must generate search queries and integrate results across steps, GRPO's group sampling creates noisy baselines because the variance between completions is high when tasks involve multiple sequential decisions.

Since [[Search-R1 found REINFORCE outperformed both PPO and GRPO for agentic deep research tasks with the highest accuracy and most efficient search strategies]], GRPO's instability was a significant practical limitation — not just slower convergence but actual collapse where training quality degrades irreversibly.

This suggests GRPO's sweet spot is well-defined tasks with verifiable outcomes (like math) where group statistics are meaningful, rather than open-ended tasks with many valid approaches where intra-group variance is inherently high. Since [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]], GRPO's successes have been specifically in verifiable-reward domains.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[GRPO eliminates the critic network through group-relative advantage scoring achieving 50 percent memory reduction over PPO]] — the mechanism that creates the instability
- [[Search-R1 found REINFORCE outperformed both PPO and GRPO for agentic deep research tasks with the highest accuracy and most efficient search strategies]] — the comparison evidence

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
