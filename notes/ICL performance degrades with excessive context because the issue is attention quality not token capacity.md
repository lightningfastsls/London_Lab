---
description: "Even large-context LLMs degrade when given irrelevant demonstrations — more tokens does not help when attention distributes over noise, echoing the 'lost in the middle' phenomenon"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# ICL performance degrades with excessive context because the issue is attention quality not token capacity

A counterintuitive finding about in-context learning: providing more demonstrations can actually hurt performance when the additional examples are irrelevant or poorly selected. This degradation persists even in models with large context windows — the issue is not running out of tokens but degradation of attention quality.

This connects to broader findings about attention limitations. Liu et al. (2023, "Lost in the Middle") showed that information placed in the middle of long contexts is poorly attended to, and since [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]], the attention distribution is inherently biased toward primacy and recency. Adding more demonstrations pushes earlier, potentially more relevant examples into the underattended middle region.

The mechanism may also relate to the implicit optimization view: since [[sequential ICL context processing follows dynamics resembling online stochastic gradient descent with learning rate determined by attention magnitude]], irrelevant examples act as noisy gradient steps that can push the implicit weight modification away from the task-relevant direction. More noise steps don't average out — they accumulate.

The practical implication is that example selection and ordering matter enormously for ICL quality. The best ICL results come from carefully curated, relevant, diverse examples rather than exhaustive demonstration sets. This is why since [[structured coherent text creates more context interference than shuffled unstructured text across all tested models]] — coherent but irrelevant demonstrations are worse than random text because they actively mislead the model's implicit optimization.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]] -- the attention distribution bias that contributes to this
- [[structured coherent text creates more context interference than shuffled unstructured text across all tested models]] -- why coherent irrelevant context is particularly harmful
- [[sequential ICL context processing follows dynamics resembling online stochastic gradient descent with learning rate determined by attention magnitude]] -- the SGD interpretation of why noise accumulates
- [[context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present]] -- when attention quality degrades, context distillation offers an escape by moving knowledge from context to weights
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- this degradation motivates moving along the spectrum from attention-based ICL toward weight-based internalization

Topics:
- [[transformer-architecture]]
