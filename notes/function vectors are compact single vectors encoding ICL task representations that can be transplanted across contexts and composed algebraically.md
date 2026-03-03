---
description: "Todd et al 2024 (Baulab, ICLR) — extracted via causal mediation analysis of attention heads, FVs trigger learned behavior in new contexts and combine for compound tasks, suggesting modular ICL"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# function vectors are compact single vectors encoding ICL task representations that can be transplanted across contexts and composed algebraically

Todd et al. (2024, ICLR, Baulab) discovered that LLM hidden states contain compact "function vectors" (FVs) — single vectors that encode the task demonstrated by in-context examples. FVs are extracted by identifying causal attention heads via causal mediation analysis, then summing their task-conditioned average outputs.

Two remarkable properties distinguish function vectors from other interpretability findings:

**Transplantability**: FVs can be extracted from one context and injected into an entirely different context, triggering the same learned behavior. If you extract the FV for "translate English to French" from a set of translation examples, you can add it to a completely different prompt and the model will perform translation — without any in-context examples present. This suggests ICL creates identifiable, modular computational pathways rather than opaque distributed processes.

**Algebraic compositionality**: Some FVs exhibit algebraic structure — combining function vectors for simple tasks creates vectors that trigger compound task behavior. The FV for "capitalize" plus the FV for "translate to French" approximates the FV for "capitalize and translate to French." This compositionality is partial (not all combinations work) but its existence at all suggests that the model's internal task representations have structure that mirrors the logical structure of the tasks themselves.

These findings bridge the gap between the weight-update view (since [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]]) and the activation-space view. Function vectors are the activation-space manifestation of the implicit weight update — a "task vector" compressed from demonstrations that modulates the model's processing.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]] -- the weight-space perspective on what FVs represent
- [[induction heads implement pattern completion via a two-layer circuit where previous-token heads write context and induction heads read it to predict continuations]] -- the lower-level mechanism FVs may build on
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- both FVs and LoRA work by amplifying existing directions, ICL implicitly and LoRA explicitly
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- FVs are the activation-space manifestation of this shared mechanism

Topics:
- [[transformer-architecture]]
