---
description: "4x improvement over Sonnet 4.5 (18.5 percent) and 3x over Gemini 3 Pro at 1M (26.3 percent) — sustained multi-needle retrieval at million-token scale"
type: baseline
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Claude Opus 4.6 achieves 76 percent on MRCR v2 8-needle at 1M tokens the strongest verified long-context result

Multi-Round Coreference Resolution v2 (MRCR v2) tests multi-turn synthetic conversations with 2, 4, or 8 hidden "needles" at context lengths from 4K to 1M tokens. Claude Opus 4.6 (February 2026) scored 76% on the hardest variant — 8 needles at 1M tokens — representing a 4x improvement over Claude Sonnet 4.5's 18.5% and nearly 3x Gemini 3 Pro's 26.3% at the same scale.

This is the strongest verified result on sustained multi-needle long-context retrieval. It represents a generation-over-generation leap that suggests the effective context ceiling is rising rapidly, consistent with [[effective context utilization improved 250x in 9 months outpacing the 30x per year growth of raw context window size]].

However, context matters. Since [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]], the 76% on a retrieval-focused benchmark does not imply equivalent performance on reasoning tasks at 1M tokens. And since [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]], the 1M window likely has an effective ceiling well below 1M for complex multi-step reasoning.

For this vault's agent architecture, the Opus 4.6 baseline establishes what the current best-case looks like at maximum scale, while the practical operating point remains governed by the 60-70% rule and the task-dependent degradation curves.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[effective context utilization improved 250x in 9 months outpacing the 30x per year growth of raw context window size]] -- the trend this data point exemplifies
- [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]] -- caveat on interpreting retrieval benchmarks
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the operational implication

Topics:
- [[agent-cognition]]
