---
description: "GitClear measured 8x code duplication and 39.9% less refactoring when AI self-reviews; security testing showed 72% Java and 43% JavaScript failure rates without independent review"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures

When the same AI model both generates and reviews code, it systematically fails to identify its own blind spots. This is not a theoretical concern — GitClear's analysis shows an 8x increase in duplicated code blocks when AI reviews its own output, and a 39.9% decrease in refactored code compared to human-reviewed code. Security testing reveals even starker numbers: 72% failure rate for Java and 43% for JavaScript when AI-generated code lacks independent review.

The mechanism is confirmation bias operating at the model level. The model that generated a solution has already committed to certain architectural assumptions, naming conventions, and implementation strategies. When asked to review the same code, it evaluates against its own internal coherence rather than external requirements. This is the generation-side analogue of the anti-gaming problem described in [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] — where agents optimized for appearing competent cannot reliably catch their own failures.

The solution is architectural: review agents must approach code with fresh context, no access to the original generation prompt or intermediate outputs. Since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]], the anchoring that occurs during generation persists through review unless the context is explicitly severed. This finding provides the empirical foundation for why patterns like the ASDLC context swap and Liza's role separation exist — they are engineering responses to a measured problem.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] -- confirmation bias is the generation-side analogue of test corruption
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the training incentive that drives anchoring through review
- [[fresh context swap between generation and review eliminates conversation drift and confirmation bias]] -- the architectural solution to this problem

Topics:
- [[agent-governance]]
- [[agent-cognition]]
