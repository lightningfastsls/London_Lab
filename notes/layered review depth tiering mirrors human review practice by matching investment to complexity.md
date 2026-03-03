---
description: "Three tiers: automated checks plus cheap model for style, mid-tier for logic and patterns, expensive model with full codebase context for security and architecture"
type: pattern
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Layered review depth tiering mirrors human review practice by matching investment to complexity

A layered review strategy applies different investment levels to different types of review work, directly mirroring how effective human review teams already operate. The three tiers:

Tier 1 (automated checks + cheap model): style, linting, obvious issues. These are high-confidence, low-ambiguity checks where a simple model or deterministic tool suffices. This tier catches formatting violations, naming convention breaks, and import ordering — issues that are clearly wrong and have clear fixes.

Tier 2 (mid-tier model + some context): logic correctness, testing adequacy, standard patterns. These require some reasoning but not deep architectural understanding. Does the function handle edge cases? Are the tests covering the new code path? Does the error handling follow the project's established patterns?

Tier 3 (expensive model + full codebase context): security vulnerabilities, architectural impact, performance implications. These require understanding the broader system context — how this change interacts with authentication flows, database access patterns, or concurrent operations.

The pattern mirrors human review practice where basic checks happen before expensive expert time is consumed. A senior engineer does not spend time noting style violations — those are caught by linters and junior reviewers first. The senior engineer's time is reserved for questions about design decisions, security implications, and systemic effects. By encoding this hierarchy into automated review, the system allocates its most expensive resource (premium model reasoning time) to the highest-value tasks.

This connects to since [[model cascading routes 70-90 percent of review to cheap models achieving 60-87 percent cost reduction]] but addresses a different axis — model cascading selects WHICH model, depth tiering determines WHAT to evaluate. Both operate simultaneously: Tier 1 uses a cheap model for shallow checks, Tier 3 uses an expensive model for deep analysis.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[model cascading routes 70-90 percent of review to cheap models achieving 60-87 percent cost reduction]] -- the complementary model selection dimension
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- Tier 1 is often deterministic rather than model-based
- [[adversarial builder-critic separation catches silent performance risks that pass all tests]] -- Tier 3 addresses the same class of issues

Topics:
- [[agent-governance]]
