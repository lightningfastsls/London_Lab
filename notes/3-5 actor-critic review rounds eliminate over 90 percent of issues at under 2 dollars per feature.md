---
description: "8 critique dimensions (security, architecture, performance, testing, etc.) with stopping criteria at max rounds, excessive criticals, or improvement rate below 10 percent threshold"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# 3-5 actor-critic review rounds eliminate over 90 percent of issues at under 2 dollars per feature

The actor-critic pattern for code review provides one of the more concrete cost-effectiveness measurements in the multi-agent review literature. Running 3-5 rounds where the Actor generates code and the Critic performs adversarial review across 8 dimensions — security, architecture, performance, testing, error handling, documentation, accessibility, code quality — "eliminates 90%+ of issues that would otherwise reach code review."

The economics are straightforward: $0.50-1.50 per feature (5 LLM calls at $0.10-0.30 each), saving 25-55 minutes of human review per feature. Even at the high end, this is a 10-30x return on investment compared to human time at typical engineering rates. The cost is dominated by the number of rounds, not the model cost per call — since [[model cascading routes 70-90 percent of review to cheap models achieving 60-87 percent cost reduction]], the Actor can use cheaper models while the Critic uses more capable ones.

Three stopping criteria prevent wasteful iteration: (1) maximum rounds reached, (2) excessive critical issues suggesting fundamental rework needed rather than iterative improvement, (3) improvement rate below 10% threshold indicating diminishing returns. The 10% threshold formalizes the intuition that there is a point where additional rounds produce marginal gains. This connects to the circuit breaker concept in multi-agent debate, but expressed as a gradient (improvement rate) rather than a binary (stuck/not-stuck).

The 8-dimension decomposition is itself a design choice — it assumes that quality dimensions are separable enough that a single Critic can evaluate them distinctly. In contrast, Claude Code's approach uses 9 parallel specialized subagents, each owning one dimension entirely, suggesting an architectural spectrum from single-critic-multi-dimension to multi-agent-single-dimension.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[model cascading routes 70-90 percent of review to cheap models achieving 60-87 percent cost reduction]] -- the cost optimization that enables cheap actor rounds
- [[multi-agent debate with circuit breaker prevents infinite review loops while 3-7 agents achieves optimal accuracy-to-cost ratio]] -- the related multi-agent pattern with different stopping criteria
- [[adversarial builder-critic separation catches silent performance risks that pass all tests]] -- the type of issue this pattern catches

Topics:
- [[agent-governance]]
