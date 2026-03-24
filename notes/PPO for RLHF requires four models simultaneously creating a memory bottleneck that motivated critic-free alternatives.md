---
description: "Actor, reward model, reference model, and critic must coexist in memory — an RLHF iteration involves six model function calls, and managing four 70B models exceeds typical infrastructure"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives

RLHF with PPO requires four LLMs concurrently in memory:

1. **Actor** (policy being trained) — generates responses
2. **Reward model** — scores generated responses
3. **Reference model** — provides KL baseline to prevent divergence
4. **Critic model** — estimates value function for advantage computation

An RLHF training iteration involves six model function calls across these four LLMs. Managing memory allocation and compute scheduling across four models — potentially each at 70B parameters — exceeds typical training infrastructure complexity. PPO also spends approximately 80% of compute time on sample generation, making it the dominant cost.

This memory bottleneck directly motivated the development of critic-free alternatives: since [[GRPO eliminates the critic network through group-relative advantage scoring achieving 50 percent memory reduction over PPO]], removing just one model saves ~16GB per billion parameters. Since [[REINFORCE Leave-One-Out uses 50-70 percent less memory than PPO while consistently outperforming it on alignment tasks]], the simplification also improved results.

The field's trajectory from 4-model PPO → 3-model GRPO/REINFORCE → 2-model DPO → 1-model SimPO represents a systematic reduction in infrastructure requirements, with each simplification surprisingly matching or exceeding its predecessor's quality.

---

Source: rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02

Relevant Notes:
- [[GRPO eliminates the critic network through group-relative advantage scoring achieving 50 percent memory reduction over PPO]] — the first simplification
- [[REINFORCE Leave-One-Out uses 50-70 percent less memory than PPO while consistently outperforming it on alignment tasks]] — the most effective simplification
- [[DPO eliminates the reward model by deriving a closed-form relationship between optimal policy and reward function enabling pure classification-based alignment]] — the offline simplification
- [[QLoRA 4-bit quantization enables 7B model fine-tuning on consumer GPUs with 33 percent memory savings at 39 percent runtime cost]] -- orthogonal to model-count reduction: QLoRA quantizes each of the 4 models, compounding the memory savings from GRPO/REINFORCE model elimination
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- LoRA adapters on the actor reduce the trainable parameter footprint, addressing a different dimension of the memory bottleneck than critic elimination

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
