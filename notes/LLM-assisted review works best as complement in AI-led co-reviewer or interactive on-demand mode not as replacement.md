---
description: "10-developer study found two effective modes — AI provides upfront summaries or responds on demand — LLMs excel at summarizing complex changes but struggle with domain conventions and false positives"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# LLM-assisted review works best as complement in AI-led co-reviewer or interactive on-demand mode not as replacement

A study with 10 developers reviewing pull requests with LLM assistance (OpenAI o4-mini with RAG) identified two effective integration modes, both preserving human judgment as the final arbiter.

The AI-led co-reviewer mode provides upfront summaries and preliminary findings before the human reviewer begins. This enhances the orientation phase of review (since [[code review follows orientation then analytical phases where skipping orientation degrades analytical quality]]) by giving the human reviewer a faster path to understanding the change's scope and potential issues.

The interactive on-demand mode keeps the AI in the background, responding only when the human reviewer asks specific questions. This avoids the noise problem — since [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]], proactively generating comments creates a filtering burden. On-demand mode ensures every AI output was explicitly requested, eliminating false positives from unwanted reviews.

LLMs excel at: summarizing complex changes (reducing orientation time), identifying subtle issues humans miss (race conditions, resource leaks), and reducing context-switching cost for reviewers handling multiple PRs.

LLMs struggle with: false positives (particularly for style and convention), domain-specific conventions (project-specific patterns that are correct but unusual), and very large codebases (where context window limitations prevent understanding cross-module impacts).

The key conclusion: success requires adaptive integration (both modes available, not forced), trust management through minimal false positives (quality over quantity), seamless embedding in existing tools (not a separate workflow), and rich context access beyond just diffs (repository knowledge, PR history, team conventions). The complementary framing — AI augments human review, does not replace it — reflects the broader finding that since [[code review provides more value through knowledge transfer and team awareness than through defect detection]], the human engagement in review is itself valuable.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[code review follows orientation then analytical phases where skipping orientation degrades analytical quality]] -- AI-led mode enhances orientation
- [[AI review false positive rates of 60-80 percent erode developer trust with concise comments 3x more likely to be acted upon]] -- why on-demand mode avoids noise
- [[code review provides more value through knowledge transfer and team awareness than through defect detection]] -- why human engagement must be preserved

Topics:
- [[agent-governance]]
