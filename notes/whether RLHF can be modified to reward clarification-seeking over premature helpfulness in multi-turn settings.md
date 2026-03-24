---
description: "Open question: current training rewards confident early answers — could a modified reward signal make 'I need more information' a positive response?"
type: open-question
confidence: speculative
created: 2026-03-01
meta_state: current
topics:
  - "[[multi-turn-degradation]]"
---

# Whether RLHF can be modified to reward clarification-seeking over premature helpfulness in multi-turn settings

Laban et al. identify RLHF training incentives as the root cause of premature helpfulness: models are trained to appear helpful and confident, which penalizes responses like "I need more information before I can answer." This creates the anchoring behavior that drives multi-turn degradation. The natural question: can the training objective be changed?

A modified RLHF signal might reward clarification-seeking in multi-turn contexts — recognizing that asking "which database schema are you using?" is more helpful than generating a SQL query based on assumptions. But this creates a tension: users also find excessive clarification-seeking annoying. The optimal behavior depends on context (is the missing information critical or can a reasonable default suffice?), which requires meta-reasoning about uncertainty.

The challenge is that single-turn and multi-turn objectives may conflict. In single-turn settings, premature helpfulness IS the right behavior — the user has specified everything, and asking for clarification is unhelpful. The model needs to distinguish between single-turn (act confidently) and multi-turn (wait for information) contexts, which may require different training regimes or explicit context signals.

Since [[approximately 60 percent of relative multi-turn degradation is constant across model sizes suggesting scaling alone cannot solve it]], training-level fixes may be the most promising path forward.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the problem this would fix
- [[the principle of least effort drives conversational underspecification making ambiguity a fundamental feature not a bug]] -- why users won't solve this themselves
- [[Mediator-Assistant framework separates intent inference from task execution recovering approximately 20 percentage points]] -- an architectural workaround

Topics:
- [[agent-cognition]]
- [[rl-alignment]]
