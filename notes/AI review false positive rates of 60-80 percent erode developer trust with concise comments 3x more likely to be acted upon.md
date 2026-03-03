---
description: "Category-specific FPR varies from 3 percent on security to 18 percent on style — aggregate metrics mislead, 33 hours per month per developer wasted filtering noise at scale"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon

The false positive problem in AI code review is both more severe and more nuanced than aggregate metrics suggest. Some tools generate 60-80% false positive rates, and research on 22,000+ comments found that concise comments are 3x more likely to be acted upon than verbose ones.

The crucial insight is that aggregate false positive rates (FPR) are misleading. A tool with 8% overall FPR might have 3% on security findings (highly reliable) but 18% on style suggestions (mostly noise). Developers who learn to distrust style suggestions may also begin ignoring security findings from the same tool — a trust erosion that spreads across categories regardless of category-specific reliability.

The business impact scales badly: 20 minutes per PR filtering noise x 5 PRs per day = approximately 33 hours per month wasted per developer. For a team of 10, that is 330 hours monthly spent on noise, not signal. This makes false positive management a critical engineering concern, not just a UX annoyance.

The 3x actionability of concise comments connects to a broader principle about agent output: since [[LLMs attempt full solution generation on the first turn even when given only a vague initial shard]], the default LLM behavior is to generate comprehensive but unfocused output. In the review context, this produces verbose comments that bury the actual finding in explanation. The research suggests that terse, specific findings (e.g., "potential race condition on line 47: counter read without lock") are more effective than detailed explanations of why race conditions are bad.

The practical response is category-specific thresholds: strict thresholds for security (where false positives are tolerated because the cost of missing a real issue is high) and aggressive filtering for style (where false positives are especially wasteful because style is subjective). This parallels the approach in USV detection where since [[recall versus precision tradeoff in two-stage USV detection]], a permissive first stage accepts false positives while a precision stage filters them.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs attempt full solution generation on the first turn even when given only a vague initial shard]] -- the output verbosity mechanism
- [[automated code review increases PR closure time by 42 percent despite 74 percent comment acceptance rate]] -- the downstream time cost of noise
- [[recall versus precision tradeoff in two-stage USV detection]] -- the same precision-recall design pattern in a different domain

Topics:
- [[agent-governance]]
