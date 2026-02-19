---
description: "General principle -- let the model learn freely first, then discover discrete structure in learned representations"
type: pattern
confidence: experimental
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# separating representation learning from discretization enables richer feature discovery

The transformer-first-then-VQ-VAE architecture (ADR-007) instantiates a general principle: separating the learning of continuous representations from the discovery of discrete structure produces richer features than end-to-end joint training. When discretization is imposed simultaneously with representation learning, the model is constrained in what it can represent -- the information bottleneck of the codebook limits exploration. By first training the transformer without any bottleneck, it freely develops whatever representations best predict the next spectrogram column. The VQ-VAE then discovers discrete structure within these already-rich representations. Since [[transformer-first then VQ-VAE avoids forcing premature discretization]], the discrete vocabulary emerges from the model's own learned features rather than being imposed during learning. This principle applies beyond USV analysis to any task where discovering a discrete vocabulary from continuous data is the goal.

---

Source:
- DECISIONS.md (ADR-007) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- the specific instantiation
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- the resulting discrete vocabulary
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- where to extract representations

Topics:
- [[classification]]
