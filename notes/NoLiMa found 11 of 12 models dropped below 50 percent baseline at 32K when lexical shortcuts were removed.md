---
description: "Adobe ICML 2025 — requires latent reasoning not literal matching, exposing true effective contexts of 1-8K for models claiming 128K-plus, with semantic distractors further collapsing performance"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# NoLiMa found 11 of 12 models dropped below 50 percent baseline at 32K when lexical shortcuts were removed

NoLiMa (Adobe, ICML 2025) addresses a fundamental flaw in standard long-context benchmarks: they allow models to succeed through lexical matching between question and needle rather than genuine comprehension. NoLiMa removes this shortcut by requiring latent reasoning — for example, a question mentions "Dresden" while the needle mentions "Semper Opera House," requiring the model to connect them semantically rather than via keyword overlap.

The results are devastating. Of 13 models tested with claimed 128K+ context, 11 of 12 primary models dropped below 50% of their baseline performance at just 32K tokens:
- GPT-4o: 99.3% baseline → 69.7% at 32K (effective context at 85% threshold: ~8K)
- Llama 3.3 70B: 97.3% → 42.7% at 32K (effective: ~2K)
- Llama 3.1 405B: 94.7% → 38.0% at 32K (effective: ~2K)
- Claude 3.5 Sonnet: 87.6% → 29.8% at 32K (effective: ~2K)

Two additional findings compound the picture. Adding semantic distractors (related but irrelevant content) reduced GPT-4o's effective length from 8K to just 1K — because since [[structured coherent text creates more context interference than shuffled unstructured text across all tested models]], semantically relevant distractors compete for attention far more effectively than random noise. Chain-of-thought prompting helped (Llama 3.3 70B improved by 16.5% at 4K and 32.7% at 32K) but still underperformed one-hop tasks without CoT, suggesting reasoning under context load has fundamental limits.

This benchmark is arguably the most important for practical agent design because real-world retrieval rarely involves literal keyword matching — it requires the latent reasoning that NoLiMa tests.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[RULER benchmark showed only half of long-context models maintained performance at 32K despite claiming 32K-plus support]] -- the complementary benchmark with lexical shortcuts intact
- [[structured coherent text creates more context interference than shuffled unstructured text across all tested models]] -- explains why semantic distractors are so much worse
- [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]] -- NoLiMa is the strongest evidence for this

Topics:
- [[agent-cognition]]
