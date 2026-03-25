---
description: "if a note is repeatedly surfaced by retrieval but never fully loaded, its description may be failing as a filter — load-rate becomes a quality signal for note descriptions"
type: open-question
confidence: speculative
created: 2026-03-07
meta_state: current
topics:
  - "[[agent-memory]]"
---

# whether tracking which surfaced notes agents actually load can identify poorly-described vault entries

CRAG (Yan et al., 2024) proposes a retrieval evaluator that classifies results before surfacing them. An extension of this pattern is tracking what happens *after* surfacing: which notes does the agent actually load fully versus ignore?

If a note is repeatedly surfaced by /kcheck or session-relevance briefs but the agent never loads it, two explanations are possible:
1. The note's description is misleading — it looks relevant from the title/description but isn't, indicating a description quality problem
2. The retrieval threshold is too loose — the note genuinely isn't relevant but scores above the threshold

Both are actionable. Since [[descriptions are retrieval filters not summaries]], a low load-rate is direct feedback that the description is failing its filtering function. The description should help agents decide whether to load the full note — if agents consistently decide "no" after reading the description, the description is either misleading or the note itself is poorly scoped.

This feedback loop would directly address the gap identified by the finding that since [[metacognitive confidence can diverge from retrieval capability]], a vault that feels well-organized may silently fail at actual retrieval. Load-rate tracking converts that abstract divergence into a measurable signal.

The open question is whether this tracking is practical in Claude Code's environment. Session transcripts are large, MCP tool calls are logged but not easily queried, and the infrastructure for correlating "surfaced in /kcheck" with "subsequently loaded" would need to be built. <!-- Superseded: qmd get replaced by topic-map-traversal + ripgrep approach, March 2026 --> The diagnostic value is clear — the implementation cost is the uncertainty.

If feasible, this creates a feedback loop: retrieval → surface → load-or-skip → description quality signal → improve description → better retrieval precision. Since [[CRAG's retrieval evaluator prevents noise-induced gate fatigue through relevance thresholds]], load-rate tracking extends the evaluator pattern from binary filtering to continuous quality improvement.

---

Source: knowledge-activation-architecture-phase1-2026-03-07 (archived inbox)

Relevant Notes:
- [[descriptions are retrieval filters not summaries]] — the mechanism this tracking would diagnose
- [[CRAG's retrieval evaluator prevents noise-induced gate fatigue through relevance thresholds]] — the evaluator pattern this extends
- [[fewer well-placed activation triggers outperform many ignored ones because noise teaches agents to skip gates]] — tracking load-rate helps identify which triggers are noise
- [[metacognitive confidence can diverge from retrieval capability]] — load-rate tracking is a concrete diagnostic for the confidence-capability gap: vaults that feel navigable (high structural quality) may silently fail at actual retrieval, and load-rate data exposes the divergence
- [[description quality for humans diverges from description quality for keyword search]] — load-rate tracking would surface exactly this divergence: notes whose descriptions read well to humans but fail to guide agent load decisions
- [[retrieval verification loop tests description quality at scale]] — an alternative mechanism for the same goal: the verification loop tests descriptions through prediction scoring, while load-rate tracking tests them through actual usage patterns

Topics:
- [[agent-memory]]
