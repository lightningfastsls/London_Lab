---
description: "GitClear 2025 analysis of 211M lines — copy-paste exceeded moved code for the first time, 39.9 percent less refactoring, corroborated by Google DORA 2024 linking AI adoption to rising defect rates"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# AI code generation caused 4x increase in code cloning and first-ever dominance of copy-paste over moved code

GitClear's 2025 report, analyzing 211 million lines of code, documents a structural shift in how code is written as AI tools become prevalent. Code cloning increased 4x, and for the first time in their data, "copy/paste" operations exceeded "moved" code operations. Refactored code decreased by 39.9%. Google's 2024 DORA report independently corroborates the trend: increased AI adoption correlates with rising defect rates.

The copy-paste dominance is particularly concerning because 57.1% of co-changed cloned code was involved in bugs. When AI generates code by pattern-matching against training data, it tends to produce complete blocks that are inserted rather than code that is carefully integrated with the existing codebase. This creates duplication that carries bugs from the original into every copy.

The 39.9% decrease in refactoring suggests that AI-assisted coding changes the developer's relationship with existing code. Rather than understanding and restructuring existing code (refactoring), developers increasingly generate new code that accomplishes similar goals (cloning). This is faster in the short term but accumulates technical debt — since [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]], the AI tools that generate the clones are poorly positioned to catch the duplication when they also perform review.

This finding provides empirical grounding for why multi-agent review architectures matter. If AI is systematically creating more duplicated, less refactored code, then the review layer must compensate by catching patterns that the generation layer introduces. Independent review with fresh context becomes necessary precisely because the generation tools create predictable failure patterns that self-review cannot detect. The cloning pattern is the output-side manifestation of the same impulse that since [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] address at the behavioral level -- appearing productive by generating volume rather than quality.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]] -- self-review amplifies cloning
- [[vendor self-evaluation bias means every AI code review benchmark vendor wins their own evaluation]] -- evaluation methodology matters when measuring code quality trends
- [[adversarial builder-critic separation catches silent performance risks that pass all tests]] -- the architectural response to generation-layer failures
- [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] -- cloning reflects the same RLHF impulse toward appearing productive

Topics:
- [[agent-governance]]
