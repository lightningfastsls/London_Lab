---
description: "ABC's Drift Bounds Theorem — without recovery, compliance decays exponentially; with recovery rate gamma exceeding drift rate alpha, decay becomes linear and bounded at D* = alpha/gamma"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Recovery mechanisms convert exponential compliance decay to linear decay through structured intervention

The ABC framework's Drift Bounds Theorem models behavioral drift via an Ornstein-Uhlenbeck stochastic differential equation and proves a key result: when recovery rate gamma exceeds the natural drift rate alpha, compliance drift converges to a bounded value D* = alpha/gamma. Without recovery mechanisms, compliance decays exponentially over the course of a session. With them, the decay becomes linear and bounded.

This result has direct architectural implications. It means that the question for long-running agent sessions is not "will the agent eventually drift?" (it will) but "can recovery mechanisms keep drift within acceptable bounds?" The answer depends on the ratio of recovery rate to drift rate. Fast, lightweight recovery mechanisms (like per-action contract checking with sub-10ms overhead as implemented in ABC's AgentAssert) maintain compliance even in extended sessions.

The finding connects to the multi-turn degradation literature: since [[multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss]], the drift is in reliability rather than capability. Recovery mechanisms address exactly this — they don't make the agent more capable, they make it more reliably compliant. This also explains why since [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]], fresh context works: it resets drift to zero rather than accumulating it.

The practical design principle: any long-running agent system needs recovery mechanisms, not just monitoring. Detecting drift without correcting it simply produces better-documented failure. Recovery mechanisms also partially dissolve the contract size ceiling -- since [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]], comprehensive upfront contracts become less necessary when fast recovery can correct drift in-flight rather than requiring exhaustive pre-specification.

The drift rate alpha is unlikely to be constant across model architectures. Since [[different model architectures exhibit distinct unconstrained behavioral patterns suggesting contracts interact differently across model families]], models with production-oriented defaults (e.g., GPT-5) may have lower drift rates for task-completion contracts but higher drift rates for reasoning contracts, and vice versa for philosophically-inclined models (e.g., Claude Opus). This implies that the required recovery rate gamma — and therefore the choice of recovery mechanism — should be calibrated to the specific model-contract interaction, not assumed universal.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[ABC framework defines probabilistic compliance where hard constraints hold with high probability and soft violations recover within bounded steps]] -- the framework containing this theorem
- [[multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss]] -- the type of drift recovery addresses
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the zero-drift reset alternative
- [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]] -- recovery partially dissolves the contract size ceiling
- [[memory wipe per review turn prevents attention degradation treating each attempt as fresh start guided by coach feedback]] -- the most radical recovery form: resetting drift to zero every turn
- [[different model architectures exhibit distinct unconstrained behavioral patterns suggesting contracts interact differently across model families]] -- drift rate alpha likely varies by model architecture, implying model-specific recovery requirements

Topics:
- [[agent-governance]]
