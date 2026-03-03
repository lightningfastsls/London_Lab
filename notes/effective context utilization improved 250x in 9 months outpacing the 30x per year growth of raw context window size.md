---
description: "Epoch AI 2025 — the input length where top models reach 80 percent accuracy rose 250x (wide CI 200-20000x) suggesting architectural improvements will close the claimed-effective gap"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Effective context utilization improved 250x in 9 months outpacing the 30x per year growth of raw context window size

According to Epoch AI (2025), context windows have grown approximately 30x per year since mid-2023 (90% CI: 10x-50x). But more importantly, effective utilization — the input length where top models reach 80% accuracy — has risen by 250x in just 9 months (with a wide CI: 200x-20,000x).

This means architectural improvements are outpacing raw capacity growth. The gap between claimed and effective context that [[maximum effective context window can differ from claimed context by as much as 99 percent and shifts by problem type]] documents is shrinking. Since [[Claude Opus 4.6 achieves 76 percent on MRCR v2 8-needle at 1M tokens the strongest verified long-context result]] represents a 4x improvement over its predecessor in a single generation, the trend is already visible in flagship models.

However, the wide confidence interval (200x-20,000x) signals substantial uncertainty about the rate. And since [[left-skewed position frequency distributions during pretraining cause effective context to rarely exceed half of training context length]], the root cause is training methodology, not architecture — meaning sustained improvement requires changes to how models are trained, not just how they are deployed.

The practical implication is cautiously optimistic: the 60-70% rule and Fresh Context Pattern remain the safe architectural choices today, but the gap they compensate for is likely to narrow over the next 1-2 model generations. Designing systems that can take advantage of improved effective context (without depending on it) is the robust engineering approach.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[Claude Opus 4.6 achieves 76 percent on MRCR v2 8-needle at 1M tokens the strongest verified long-context result]] -- a concrete example of the improvement
- [[left-skewed position frequency distributions during pretraining cause effective context to rarely exceed half of training context length]] -- the root cause limiting further improvement
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the conservative rule that remains valid despite improvement

Topics:
- [[agent-cognition]]
