---
description: "For each prompt, generate multiple completions and normalize rewards within the group — advantage = (reward - mean) / std — eliminating the learned value function and saving ~16GB per billion parameters"
type: method
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# GRPO eliminates the critic network through group-relative advantage scoring achieving 50 percent memory reduction over PPO

Group Relative Policy Optimization (GRPO), introduced by DeepSeek for mathematical reasoning (DeepSeekMath, 2024), replaces PPO's learned critic with a simple statistical baseline.

**Core mechanism**: For each prompt, generate multiple completions (a "group"). Compute advantage as the normalized reward relative to the group: `Advantage_i = (reward_i - mean(group_rewards)) / (std(group_rewards) + epsilon)`. Each token within a completion receives identical advantage estimates — no per-token value function needed.

**Architectural simplification**: GRPO requires only 3 models (policy, reference, reward) versus PPO's 4 (adds critic). Since [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]], eliminating the critic saves approximately 16GB per billion parameters in training memory, achieving roughly 50% overall memory reduction.

The trade-off is stability: since [[GRPO requires large batch sizes for stability and suffers frequent training collapse in multi-step long-context reasoning]], the group-relative baseline is noisier than a learned value function. DeepSeekMath used batch size 1,024 (16 prompts x 64 completions) to compensate. Single policy updates per batch are common, contrasting with PPO's multiple epochs per batch.

Variants have emerged to address GRPO's biases: **Dr. GRPO** removes length and standard deviation normalization to eliminate biases favoring longer incorrect responses. **LCPO** adds explicit length penalties.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]] — the problem GRPO solves
- [[GRPO requires large batch sizes for stability and suffers frequent training collapse in multi-step long-context reasoning]] — the cost of simplification
- [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]] — the most dramatic GRPO result
- [[QLoRA 4-bit quantization enables 7B model fine-tuning on consumer GPUs with 33 percent memory savings at 39 percent runtime cost]] -- GRPO's 50% memory reduction stacks with QLoRA's quantization: 3 models (policy+reference+reward) with 4-bit quantization further reduces the GPU memory footprint for RL alignment training

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
