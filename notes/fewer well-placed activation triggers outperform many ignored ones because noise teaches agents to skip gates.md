---
description: "alarm fatigue applies to knowledge systems — five canary comments that always get read beat fifty that get ignored, so activation precision matters more than coverage"
type: finding
confidence: likely
created: 2026-03-07
meta_state: current
---

# fewer well-placed activation triggers outperform many ignored ones because noise teaches agents to skip gates

The principle is borrowed from alarm fatigue in medical systems: when monitors produce too many false alarms, clinicians learn to ignore all alarms, and genuine emergencies get missed. The same dynamic applies to knowledge activation in agent systems.

If every source file has a `# VAULT:` comment, or if every session-start brief lists twenty weakly-relevant notes, the agent learns that activation mechanisms produce noise. The rational response — under context pressure — is to skip reading the activation output entirely. The gates become invisible. Since [[CRAG's retrieval evaluator prevents noise-induced gate fatigue through relevance thresholds]], filtering weak results before surfacing is the primary defense against this failure mode.

The practical implication is that activation system design should optimize for precision over recall. Five well-placed canary comments on files that have actually caused regressions will be read and respected. Fifty canary comments spread across the codebase will be treated as boilerplate. Since [[Adaptive RAG routes retrieval depth by query complexity which maps to file modification risk in coding agents]], the risk classification provides the decision framework for where to invest activation triggers.

This aligns with the broader trend that since [[context quality over quantity is the defining trend with tools optimizing for less context that matters more not more context that dilutes]], the same precision-over-volume principle applies across context loading, tool description surfacing, and activation triggers.

This parallels the storage-side finding that since [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]], unmanaged accumulation degrades system performance. On the activation side, unmanaged trigger proliferation degrades trigger effectiveness through the same mechanism — information overload leading to disengagement.

---

Source: knowledge-activation-architecture-phase1-2026-03-07 (archived inbox)

Relevant Notes:
- [[CRAG's retrieval evaluator prevents noise-induced gate fatigue through relevance thresholds]] — the filtering mechanism that prevents noise
- [[Adaptive RAG routes retrieval depth by query complexity which maps to file modification risk in coding agents]] — the risk classification that guides trigger placement
- [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]] — the storage analogue
- [[context quality over quantity is the defining trend with tools optimizing for less context that matters more not more context that dilutes]] — the same precision-over-volume principle applied to context loading: MCP Tool Search, Deepcon, and Docfork all independently converge on fewer high-quality tokens outperforming many weak ones
- [[nudge theory explains graduated hook enforcement as choice architecture for agents]] — addresses the same alert fatigue dynamic from the enforcement side: graduated severity preserves signal value, just as selective trigger placement preserves activation credibility
- [[automation should be retired when its false positive rate exceeds its true positive rate or it catches zero issues]] — retirement criteria formalize when accumulated triggers have become noise: a trigger that catches nothing is the opposite extreme of one that fires too often

Topics:
- [[agent-memory]]
