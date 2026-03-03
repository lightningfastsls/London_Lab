---
description: "Huntley at Sourcegraph reports tool calls fail, brute-force replaces reasoning — system prompt consumes ~24K leaving ~176K usable before the cliff"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Claude Sonnet exhibits a qualitative performance cliff at 147K-152K tokens which is 73-76 percent of its 200K window

Multiple practitioners (Geoffrey Huntley at Sourcegraph, Claude Code community members) report a qualitative cliff in Claude Sonnet performance around 147K-152K tokens — 73-76% of the 200K context window. At this point, "tool call to tool call invocation starts to fail" and "brute-force solutions replace reasoning." The degradation is not a gradual decline but a noticeable phase transition in behavior quality.

With system prompts consuming approximately 24K tokens, the usable window is effectively ~176K before the cliff hits. Auto-compact triggers between 75-92% capacity depending on implementation, which is designed to prevent reaching this cliff — but since [[context compaction quality degrades cumulatively with multiple compressions regardless of implementation]], relying on auto-compact as a safety net has its own costs.

This practitioner-reported cliff aligns with the academic finding that [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]]. For a 200K window, 60-70% effective capacity is 120K-140K — slightly below the reported cliff at 147K-152K. The gap suggests that gradual degradation begins before the cliff, with the cliff being the point where degradation becomes qualitatively obvious to users.

Notably, this observation comes from real-world agentic use (tool calls, code generation, multi-step reasoning) which is harder than the retrieval tasks in most benchmarks. Since [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]], the cliff appears earlier for complex agentic work than benchmarks would predict.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the rule of thumb this observation supports
- [[context compaction quality degrades cumulatively with multiple compressions regardless of implementation]] -- the mitigation trade-off
- [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]] -- why agentic cliffs appear earlier than benchmarks suggest

Topics:
- [[agent-cognition]]
