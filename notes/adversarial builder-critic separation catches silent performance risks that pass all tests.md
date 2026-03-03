---
description: "ASDLC case study caught a table-loading performance violation invisible to tests — adversarial review addresses the gap between correctness and production-readiness"
type: pattern
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# Adversarial builder-critic separation catches silent performance risks that pass all tests

The adversarial builder-critic pattern reveals a fundamental gap in automated quality assurance: the space between "correct" and "production-ready." Deterministic quality gates — tests, linters, type checkers — verify that code does what it claims. Adversarial review asks whether what the code claims is the right thing to claim.

In the ASDLC validation, an adversarial Critic caught a performance violation where code loaded an entire database table into memory for filtering. The code was correct — it produced right results, passed all tests, satisfied type constraints. But it would fail catastrophically at production scale. No deterministic gate could catch this because the gate validates against explicit specifications, and the specification said "filter the data," not "filter the data efficiently."

This finding extends the principle from [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]]. Deterministic tools are necessary — they catch the 90%+ of issues that are specification violations. But they are not sufficient — they cannot catch the class of issues where the specification itself is incomplete. Adversarial review fills this gap by bringing judgment, not just compliance.

The pattern pairs well with asymmetric model selection: the Builder optimized for speed (e.g., Gemini Flash, Claude Haiku) while the Critic is optimized for reasoning (e.g., Gemini Deep Think, DeepSeek V3.2). This asymmetry means the Builder can move fast while the Critic catches what speed misses, creating a complementary rather than redundant quality layer.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- the necessary-but-not-sufficient deterministic layer
- [[fresh context swap between generation and review eliminates conversation drift and confirmation bias]] -- the context mechanism that enables adversarial review
- [[test integrity truth table prevents the most dangerous agent failure mode of modifying tests to match bugs]] -- related quality gate for test integrity
- [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] -- adversarial review specifically catches fabrication that self-review cannot

Topics:
- [[agent-governance]]
