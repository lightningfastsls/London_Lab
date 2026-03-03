---
description: "Models are trained to appear helpful and confident, which makes 'I need more information' a penalized response — driving the premature commitment root cause"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses

Laban et al. identify premature answer attempts as the first root cause of multi-turn degradation, and trace it to RLHF training objectives. Models are trained to appear helpful, agreeable, and competent. This creates a systematic bias: when given partial information, the model generates a full solution proposal rather than requesting clarification — because "I need more information before I can answer" is penalized as evasive during training.

This behavior is rational under the training objective but catastrophic in multi-turn settings. The early solution becomes an anchor: subsequent turns that reveal contradicting or refining information trigger revision attempts that layer on assumptions rather than replacing the anchored response. Since [[answer bloat compounds multi-turn errors as responses grow verbose without pruning incorrect assumptions]], the anchoring cascades.

The deeper insight is that the multi-turn problem is not a capability deficit — it is a training incentive misalignment. Models are trained on single-turn reward signals where helpfulness and confidence are always positive. But in multi-turn settings, premature helpfulness becomes harmful. This suggests the fix is not more capability (scaling) but different training objectives — since [[whether RLHF can be modified to reward clarification-seeking over premature helpfulness in multi-turn settings]] remains an open research question.

## The RLHF Mechanism That Creates This

The training pipeline explains *why* helpfulness is rewarded. Since [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]], the reward model (Stage 3) is trained on human pairwise preferences — and since [[pairwise comparisons produce more reliable human preference data than absolute ratings because relative judgments avoid calibration problems]], annotators compare responses where the more helpful-seeming one wins. The RL fine-tuning (Stage 4) then optimizes against this reward model, which since [[credit assignment over hundreds of tokens from a single scalar reward is the central bottleneck of RLHF]] provides only a single score for the entire response. The model cannot learn "helpfulness on token 5 was premature" — it only learns "being helpful got a high score." This scalar feedback structure makes the premature helpfulness incentive structurally inevitable under standard RLHF.

## Runtime Contracts as Complementary Mitigation

Behavioral contracts research (2025-2026) reveals that runtime contracts provide a complementary mitigation layer for this training incentive misalignment. Since [[training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions]], runtime enforcement can catch the specific failure modes RLHF creates: fabrication (appearing knowledgeable), premature action (appearing efficient), and scope creep (appearing thorough). The Vass struggle protocol directly addresses premature helpfulness by creating an explicit permission structure for uncertainty — transforming "I'm stuck" from a penalized response into the correct response. Since [[contract visibility improves natural compliance even before enforcement the transparency effect]], even prompt-level contracts partially counteract the RLHF incentive by providing an alternative frame where uncertainty is valued over confidence.

---

Sources:
- multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)
- [[rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02]] (RLHF mechanism section)

Relevant Notes:
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- the behavioral manifestation
- [[answer bloat compounds multi-turn errors as responses grow verbose without pruning incorrect assumptions]] -- the compounding mechanism
- [[LLMs attempt full solution generation on the first turn even when given only a vague initial shard]] -- the specific pattern from Appendix F.1
- [[whether RLHF can be modified to reward clarification-seeking over premature helpfulness in multi-turn settings]] -- the open question
- [[training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions]] -- the runtime complement
- [[struggle protocol over silent failure requires agents to surface uncertainty rather than fabricate confidence]] -- the direct contract-level counterweight
- [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]] -- the pipeline that creates the incentive
- [[credit assignment over hundreds of tokens from a single scalar reward is the central bottleneck of RLHF]] -- why per-token helpfulness feedback is impossible

Topics:
- [[agent-cognition]]
- [[rl-alignment]]
