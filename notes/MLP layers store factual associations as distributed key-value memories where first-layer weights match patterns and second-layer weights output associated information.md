---
description: "Geva et al 2021 showed FFN first layer acts as keys matching input patterns and second layer as values outputting associated info — though storage is distributed across many neurons, not one-to-one"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# MLP layers store factual associations as distributed key-value memories where first-layer weights match patterns and second-layer weights output associated information

Geva et al. (2021, "Transformer Feed-Forward Layers Are Key-Value Memories") demonstrated that the two-layer MLP in transformer blocks functions analogously to a key-value memory system. The first layer's weight matrix acts as "keys" — its rows define patterns that match against input representations. When an input activates a particular row (via high dot product), the corresponding column in the second layer's weight matrix acts as the "value" — contributing specific output information.

The 4× expansion ratio in the standard FFN (d_model → 4×d_model → d_model) is significant here: it projects into a much higher-dimensional space where individual neurons can specialize in detecting specific input patterns. This expansion creates enough capacity for the MLP to store a rich set of pattern-value associations.

However, the framing of "specific neurons for specific facts" should not be taken as a clean one-neuron-one-fact mapping. Factual storage in modern LLMs is distributed across many neurons rather than cleanly localized. More recent work, particularly Meng et al. (2022, "Locating and Editing Factual Associations in GPT"), provides a more nuanced view: factual recall involves distributed representations that can be localized to specific layers and regions but not to individual neurons. The key-value memory analogy captures the functional pattern but oversimplifies the representation.

This connects to our USV pipeline because since [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]], understanding what MLPs store and how they retrieve it informs what our probing experiments are actually measuring.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently]] -- MLPs are the "transform" component
- [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] -- our USV probing of these representations
- [[the residual stream architecture lets transformer components read from and write to a shared information stream enabling additive accumulation]] -- the stream MLPs read from and write to
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- ICL's implicit low-rank updates modify these MLP key-value memories virtually; LoRA makes the modification permanent

Topics:
- [[transformer-architecture]]
