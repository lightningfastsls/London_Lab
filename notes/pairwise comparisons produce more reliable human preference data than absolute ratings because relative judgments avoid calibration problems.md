---
description: "Humans cannot maintain consistent internal scoring scales — relative judgments ('which is better?') are inherently easier and more reliable, exhibiting better inter-annotator agreement"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics:
  - "[[model-adaptation]]"
---

# Pairwise comparisons produce more reliable human preference data than absolute ratings because relative judgments avoid calibration problems

Human preference data in RLHF is collected as pairwise comparisons ("A is better than B") rather than absolute scores for several well-established reasons:

- **Consistency**: Direct scalar scoring is uncalibrated and noisy due to differing human values. Asking "rate this 1-10" yields wildly different baselines across annotators. Asking "which is better?" eliminates this calibration problem entirely.
- **No reference point needed**: Absolute scores require an implicit reference standard that varies between annotators. Pairwise comparisons are self-referencing.
- **Better regularization**: Rankings from pairwise comparisons are much better regularized than raw scores, producing cleaner training signal.
- **Empirical validation**: Pairwise comparison exhibits better human correlations than traditional scoring-based evaluators.

The Bradley-Terry model converts these pairwise rankings into a trainable loss function: since [[reward model training uses Bradley-Terry pairwise comparison on approximately 50k labeled preference samples]], this mathematical framework is the bridge between human judgment and differentiable optimization.

Interestingly, some recent high-quality datasets (e.g., UltraFeedback) are curated with absolute ratings on multiple dimensions (instruction following, truthfulness, honesty, helpfulness) that are then *converted* to relative rankings for training — suggesting the field recognizes that even when richer data is available, the pairwise formulation is more effective for training.

---

Source: [[rl-alignment-rlhf-ppo-grpo-reinforce-dpo-research-2026-03-02]]

Relevant Notes:
- [[reward model training uses Bradley-Terry pairwise comparison on approximately 50k labeled preference samples]] — the mathematical framework
- [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]] — where pairwise data fits in the pipeline

Topics:
- [[model-adaptation]]
- [[rl-alignment]]
