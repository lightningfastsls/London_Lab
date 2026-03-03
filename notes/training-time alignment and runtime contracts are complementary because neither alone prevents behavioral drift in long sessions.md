---
description: "Constitutional AI shapes model weights; runtime contracts monitor outputs — ABC explicitly positions itself as complementary to RLHF because training alone cannot prevent session-level drift"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions

Constitutional AI (Bai et al., 2022) and RLHF operate at training time, baking behavioral norms into model weights through self-critique, revision against principles, and reinforcement learning from AI feedback. Runtime contracts (ABC, AgentSpec, Pro2Guard, VeriGuard) operate at deployment time, monitoring and constraining behavior externally. The ABC framework explicitly positions itself as "complementary to training-time alignment" — necessary because training-time alignment alone cannot prevent all behavioral drift in extended sessions.

The gap between training and runtime is clear from the multi-turn degradation research: since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]], training-time alignment actually creates some of the failure modes that runtime contracts must catch. RLHF optimizes for appearing helpful on individual turns, but this incentive becomes harmful when distributed across multi-turn sessions. The TrustAgent framework (2024) attempted to bridge this by applying constitutional principles at runtime through pre-planning, in-planning, and post-planning safety checks, but the ABC paper critiques this as relying on immutable principles insufficient for the complexity of evolving agent interactions.

The practical convergence is a layered model: training provides a behavioral baseline (models start aligned), and runtime contracts provide session-level governance (models stay aligned). Neither layer alone is sufficient — training without runtime monitoring allows drift, and runtime monitoring without training baseline would need to catch every single failure rather than only deviations from a mostly-aligned starting point.

## How Training-Time Alignment Actually Works

Since [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]], the training-time alignment involves: (1) a reward model trained on ~50k pairwise human preferences via Bradley-Terry, (2) RL fine-tuning with KL penalty preventing divergence from the reference model. However, since [[KL divergence penalty prevents reward model exploitation but paradoxically increases the proxy-gold reward gap]], even the standard safeguard mechanism has limitations. The reward model itself is an imperfect proxy — since [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]], optimizing against it inevitably creates exploitable patterns. This is precisely why runtime contracts are needed: training-time alignment is inherently approximate, and its approximation errors manifest as the behavioral drift that runtime governance must catch.

---

Sources:
- behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)
- rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02 (RLHF mechanism details)

Relevant Notes:
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the training-time alignment that creates some failure modes
- [[ABC framework defines probabilistic compliance where hard constraints hold with high probability and soft violations recover within bounded steps]] -- the runtime complement
- [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]] -- why runtime recovery is needed on top of training
- [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]] -- the pipeline creating the baseline
- [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]] -- why training-time alignment is inherently approximate

Topics:
- [[agent-governance]]
- [[rl-alignment]]
