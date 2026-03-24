---
description: "metaswarm enforces FAIL to retry-or-escalate only — orchestrator validates independently of subagent self-reports, adversarial reviewers require file and line evidence for DoD compliance"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# No instruction path from failure to commit is the critical safety invariant in automated code pipelines

In metaswarm's 4-phase orchestrated execution loop — IMPLEMENT, VALIDATE, ADVERSARIAL REVIEW, COMMIT — the most critical design invariant is negative: no instruction path exists from FAIL to COMMIT. When any phase fails, the only options are retry (re-enter the loop) or escalate (surface the failure to a higher authority). There is never an option to skip validation and proceed to commit.

This is a structural safety invariant, analogous to the principle that since [[test integrity truth table prevents the most dangerous agent failure mode of modifying tests to match bugs]]. Both patterns prevent the same category of failure — shortcuts that create the appearance of quality without the substance. In the test integrity case, the forbidden action is modifying expectations; in the pipeline case, the forbidden action is committing unvalidated code.

This zero-path invariant is the pipeline-level equivalent of the approval gate: since [[externalized reasoning at approval gates forces agents to improve their plans before executing them]], the ANALYSIS->EXECUTION forbidden transition prevents premature action, while FAIL->COMMIT prevents premature completion. Both enforce structural pauses at the most consequential transition points.

Three additional design choices reinforce the invariant: (1) the orchestrator validates independently, never trusting subagent self-reports of success; (2) adversarial reviewers check Definition of Done compliance with file:line evidence, making vague claims of completion impossible; (3) the broader pipeline includes both a Design Review Gate (5 parallel reviewers) and a Final Comprehensive Review, creating redundant verification layers.

metaswarm is production-tested with 100% test coverage across hundreds of PRs. The zero-path invariant has survived this testing without exceptions — suggesting that the "never FAIL to COMMIT" rule is not just theoretically sound but practically enforceable at scale. This validates the architectural approach to safety, since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] — the invariant is enforced by the absence of a code path, not by an instruction to "never commit failing code."

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[test integrity truth table prevents the most dangerous agent failure mode of modifying tests to match bugs]] -- the analogous quality gate for test expectations
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- structural over instructional enforcement
- [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] -- false completion is the specific failure mode this prevents
- [[externalized reasoning at approval gates forces agents to improve their plans before executing them]] -- the analogous forbidden transition: ANALYSIS->EXECUTION as FAIL->COMMIT

Topics:
- [[agent-governance]]
