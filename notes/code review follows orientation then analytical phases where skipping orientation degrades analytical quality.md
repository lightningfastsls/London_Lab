---
description: "CRDM ethnographic model from 10 participants and 34 reviews shows cognitive similarity to recognition-primed decision-making — AI tools that skip context establishment miss the foundation of effective review"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Code review follows orientation then analytical phases where skipping orientation degrades analytical quality

An ethnographic think-aloud study (10 participants, 34 reviews) built the Code Review as Decision-Making (CRDM) model, revealing that review follows two distinct cognitive phases: orientation and analytical. In the orientation phase, the reviewer establishes context — understanding the PR's purpose, the broader system it affects, and the rationale behind the changes. In the analytical phase, the reviewer applies that context to understand, assess, and plan responses to specific code changes.

The cognitive process shows similarities to recognition-primed decision-making, a model from naturalistic decision-making research where experts make rapid decisions by recognizing patterns from experience rather than exhaustively analyzing options. Experienced reviewers orient quickly because they recognize the type of change and its typical implications.

The implication for AI review tools is direct: most AI reviewers skip the orientation phase entirely and go straight to finding issues. They scan diffs for patterns matching known bug categories without first understanding WHY the changes were made and WHAT system context they operate within. This is equivalent to a human reviewer who jumps to line-by-line critique without reading the PR description or understanding the feature being implemented.

The CRDM model suggests that improving AI review quality may require investing in better orientation — building repository context, understanding change intent, and establishing a mental model of the system — before attempting issue detection. This connects to the finding that since [[code review provides more value through knowledge transfer and team awareness than through defect detection]], the orientation phase is where the human-AI collaboration has the most potential: AI can synthesize context rapidly, enabling faster human orientation.

The model also explains why since [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]] — without proper orientation, the AI lacks the context to distinguish between genuinely problematic patterns and intentional design choices, generating false positives from misunderstood intent.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[code review provides more value through knowledge transfer and team awareness than through defect detection]] -- orientation is the knowledge transfer phase
- [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]] -- the downstream cost of skipping orientation
- [[LLMs attempt full solution generation on the first turn even when given only a vague initial shard]] -- the analogous premature commitment mechanism

Topics:
- [[agent-governance]]
- [[agent-cognition]]
