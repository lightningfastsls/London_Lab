---
description: "MECW 2025 found some top models failed with as little as 100 tokens in context — effective window depends on task complexity not just architecture"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Maximum effective context window can differ from claimed context by as much as 99 percent and shifts by problem type

The Maximum Effective Context Window (MECW, 2025) study found "significant differences between reported MCW and MECW, with models falling far short by as much as 99 percent." Some top models failed with as little as 100 tokens in context when the task required genuine reasoning rather than pattern matching.

Critically, MECW shifts by problem type. This means there is no single "effective context length" for a model — the effective length depends on what you're asking the model to do. Since [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]], a model might have an effective context of 64K for keyword retrieval but only 2K for latent reasoning (as [[NoLiMa found 11 of 12 models dropped below 50 percent baseline at 32K when lexical shortcuts were removed]] demonstrates).

This finding has architectural implications. An agent system cannot rely on a single context budget — it must account for the task type being performed. Retrieval-heavy operations (searching codebases, finding files) can fill more context than reasoning-heavy operations (debugging, architectural decisions). This is why [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] uses a conservative blanket rule rather than trying to optimize per-task.

The 99% gap figure is an extreme case, but even moderate gaps are enough to undermine systems designed around claimed context sizes. Since [[RULER benchmark showed only half of long-context models maintained performance at 32K despite claiming 32K-plus support]], the "claimed ≠ effective" finding is robust across multiple independent benchmarks.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[RULER benchmark showed only half of long-context models maintained performance at 32K despite claiming 32K-plus support]] -- converging evidence from different benchmarks
- [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]] -- the task-dependency mechanism
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the architectural response

Topics:
- [[agent-cognition]]
