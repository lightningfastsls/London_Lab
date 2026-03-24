---
description: "Elhage et al 2021 'Mathematical Framework for Transformer Circuits' — each attention head and MLP reads from and adds to a shared residual stream, enabling independent analysis of components"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# the residual stream architecture lets transformer components read from and write to a shared information stream enabling additive accumulation

Elhage et al. (2021, "A Mathematical Framework for Transformer Circuits") formalized the transformer as a series of components that read from and write to a shared "residual stream." Each attention head and each MLP layer reads its input from the residual stream and adds its output back via residual connections. This framing has three important consequences.

First, components can be analyzed somewhat independently. Because each component reads from the same stream and adds back to it, the contribution of any individual attention head or MLP layer can be isolated and studied. This is the foundation of mechanistic interpretability — you can ask "what does head 7 in layer 3 do?" and get a meaningful answer because its contribution is additive.

Second, information accumulates additively across layers. The residual stream carries a running sum of all contributions from earlier components. Nothing is overwritten — each component adds new information on top of what's already there. This explains why transformers can be deep without catastrophic information loss: the skip connections guarantee that the input signal persists through the entire network, with each layer optionally enriching it.

Third, any component can read information written by any earlier component, enabling "virtual circuits" that span multiple layers. An attention head in layer 5 can read information that an MLP in layer 2 wrote, even though they don't directly communicate. This is how since [[induction heads implement pattern completion via a two-layer circuit where previous-token heads write context and induction heads read it to predict continuations]] — the two heads communicate through the residual stream.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently]] -- the components that read/write to this stream
- [[induction heads implement pattern completion via a two-layer circuit where previous-token heads write context and induction heads read it to predict continuations]] -- example of cross-layer communication through the residual stream

Topics:
- [[transformer-architecture]]
