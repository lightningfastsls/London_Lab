---
description: "Liu et al. 2026 two-stage pipeline where a Mediator reconstructs ambiguous multi-turn input into explicit instructions before the Assistant executes"
type: method
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[multi-turn-degradation]]"
---

# Mediator-Assistant framework separates intent inference from task execution recovering approximately 20 percentage points

Liu et al. (2026, arXiv 2602.07338) propose a two-stage pipeline that reframes multi-turn degradation as pragmatic intent mismatch rather than capability deficit. The architecture: (1) a Mediator analyzes accumulated context and distilled user-specific guidelines to reconstruct ambiguous multi-turn inputs into explicit, fully-specified instructions, (2) an Assistant executes tasks based on the clarified instructions.

Results: GPT-4o-mini recovered from 53.6% (Sharded) to 73.9% (with Mediator), a 20.3 percentage point recovery. Even reasoning-enhanced DeepSeek-V3.2-Thinking gained 21.1 percentage points, suggesting intent clarity complements reasoning depth — they address different failure modes.

The key architectural insight is that intent understanding and task execution are separable concerns. The Mediator handles the pragmatic challenge (what does the user actually mean?) while the Assistant handles the technical challenge (how to implement it). This parallels the software engineering principle of separating concerns: since [[user utterances are a lossy compression of high-dimensional intent into low-dimensional surface forms]], an explicit decompression step before execution prevents the cascading errors that come from executing on misunderstood intent.

An Experience Refiner module extracts structured textual guidelines from historical success/failure pairs, providing the Mediator with pragmatic priors like "If the user hasn't explicitly approved a solution, they remain unsatisfied."

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[user utterances are a lossy compression of high-dimensional intent into low-dimensional surface forms]] -- the theoretical foundation for why Mediators help
- [[RAG-based memory provides only marginal improvement versus intent resolution demonstrating retrieval is not equivalent to resolving intent]] -- memory alone doesn't substitute for intent inference
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- Concat is more effective but less practical

Topics:
- [[agent-cognition]]
