---
description: "General principle -- let the model learn freely first, then discover discrete structure in learned representations"
type: pattern
confidence: experimental
conditions: []
meta_state: current
topics:
  - "[[representation-learning]]"
---

# separating representation learning from discretization enables richer feature discovery

The transformer-first-then-VQ-VAE architecture (ADR-007) instantiates a general principle: separating the learning of continuous representations from the discovery of discrete structure produces richer features than end-to-end joint training. When discretization is imposed simultaneously with representation learning, the model is constrained in what it can represent -- the information bottleneck of the codebook limits exploration. By first training the transformer without any bottleneck, it freely develops whatever representations best predict the next spectrogram column. The VQ-VAE then discovers discrete structure within these already-rich representations. Since [[transformer-first then VQ-VAE avoids forcing premature discretization]], the discrete vocabulary emerges from the model's own learned features rather than being imposed during learning. This principle applies beyond USV analysis to any task where discovering a discrete vocabulary from continuous data is the goal. Empirical evidence from bioacoustics now supports this: [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] -- post-hoc VQ on frozen HuBERT features achieved only 35% UAR versus 49% for continuous representations, a 14 percentage point gap that demonstrates the cost of applying discretization after the fact rather than jointly training it.

---

Source:
- DECISIONS.md (ADR-007) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- the specific instantiation
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- the resulting discrete vocabulary
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- where to extract representations
- [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] -- empirical evidence: post-hoc discretization loses 14pp UAR vs continuous baselines
- [[STSG spectrogram token skip-gram achieved only 0.559 AUC versus 0.810 for transfer learning on bioacoustic classification]] -- even harsher evidence: K-means discretization without learned representations performs dramatically worse
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- the same principle at the weight level: LoRA succeeds because pre-trained weights already contain task-relevant directions, just as our transformer already contains rich representations before VQ-VAE discovers discrete structure in them
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- separation principle directly addresses this tension: learn the continuum first, then discretize it carefully
- [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]] -- directly tests the separation principle: raw features found one big cluster, but learned representations might reveal sub-structure that justifies discretization

Topics:
- [[representation-learning]]
