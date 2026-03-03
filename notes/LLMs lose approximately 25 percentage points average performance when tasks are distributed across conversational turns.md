---
description: "Laban et al. ICLR 2026 oral tested 15 models across 6 tasks with 200K+ simulations finding a 39% relative (25pp absolute) drop from single to multi-turn"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns

Laban et al. (2025, ICLR 2026 oral) tested 15 LLMs ranging from 8B open-source (OLMo-2, Llama 3.1-8B) to frontier proprietary (GPT-4.1, Gemini 2.5 Pro, Claude 3.7 Sonnet, o3) across 6 benchmark tasks (code generation, text-to-SQL, API calling, math, data-to-text, summarization). Using 200,000+ simulated conversations at temperature T=1.0, they found that distributing task requirements across multiple conversational turns produces an average 39% relative performance drop — approximately 25 absolute percentage points (from ~90% to ~65% for top-tier models).

The degradation is architecture-independent: all models degrade by 30-46%, with no systematic advantage for larger, more capable, or reasoning-enhanced models. Top-tier full (single-turn) performance clustered at 91-93% (GPT-4.1: 92.4%, GPT-4o: 92.6%, Gemini 2.5 Pro: 92.1%), but multi-turn performance collapsed to 51-55% (GPT-4.1: 55.2%, Claude 3.7 Sonnet: 52.8%). The critical implication: the multi-turn setting is where models are actually used in practice, yet benchmarks predominantly test single-turn performance.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss]] -- decomposes this headline finding into aptitude vs reliability
- [[even two conversational turns trigger multi-turn degradation regardless of task complexity]] -- the degradation threshold
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the most effective mitigation

Topics:
- [[agent-cognition]]
