---
description: "surfacing weak retrieval results teaches agents to ignore activation gates entirely — a lightweight evaluator filtering results before display preserves gate credibility over time"
type: method
confidence: likely
created: 2026-03-07
meta_state: current
---

# CRAG's retrieval evaluator prevents noise-induced gate fatigue through relevance thresholds

Corrective RAG (Yan et al., 2024) adds a lightweight evaluator between retrieval and generation that classifies results as Correct, Incorrect, or Ambiguous before passing them to the model. The key insight is not the classification itself but the architectural position: evaluation happens *after* retrieval but *before* surfacing, creating a filter that prevents low-quality results from polluting the generation context.

Applied to agent knowledge systems, this translates to relevance thresholds on activation mechanisms. When a session-start brief or mid-session knowledge check surfaces results, those results must clear a quality bar before being shown to the agent. The consequence of skipping this filter is gate fatigue: the agent encounters too many irrelevant results, learns that the activation mechanism produces noise, and begins skipping it entirely. The gate becomes invisible.

This is the same dynamic as alarm fatigue in medical systems — when monitors produce too many false alarms, clinicians learn to ignore them, and real emergencies get missed. Since [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]], the storage-side version of this problem is well-documented. The activation-side version follows the same logic: unfiltered activation is worse than no activation because it erodes trust in the mechanism.

The practical implication is: it is better to show zero results with a "no strong matches" message than to show five weak matches. Since [[descriptions are retrieval filters not summaries]], the description quality of vault notes directly affects whether retrieval results pass the relevance threshold. Poorly-described notes that match semantically but don't signal their relevance through descriptions will be filtered out — creating a feedback pressure toward better descriptions.

---

Source: knowledge-activation-architecture-phase1-2026-03-07 (archived inbox)

Relevant Notes:
- [[descriptions are retrieval filters not summaries]] — poor descriptions cause notes to fail the relevance filter, creating quality pressure
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] — this note addresses the quality side of the activation equation
- [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]] — the storage analogue: unfiltered accumulation degrades performance just as unfiltered activation does

Topics:
- [[agent-memory]]
