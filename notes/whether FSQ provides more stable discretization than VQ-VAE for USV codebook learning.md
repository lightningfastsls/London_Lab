---
description: "Finite Scalar Quantization as alternative to VQ-VAE if codebook collapse persists despite prevention mechanisms"
type: open-question
confidence: speculative
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning

Finite Scalar Quantization (FSQ) is listed as a fallback approach if VQ-VAE codebook collapse persists despite the multiple prevention mechanisms described in [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]]. FSQ avoids the codebook collapse problem entirely by quantizing each dimension of the representation independently into a fixed number of levels, rather than learning codebook entries. The discrete vocabulary emerges from the combinations of quantized dimensions rather than from learned prototypes. The tradeoff is that FSQ may produce less interpretable codes (combinations of scalar values vs. learned cluster centers) and the effective vocabulary size is the product of per-dimension levels rather than a directly controlled parameter. This has not yet been tested in the USV context.

### ROADMAP Context

ROADMAP provides the specific citation: Mentzer et al., ICLR 2024. The FSQ mechanism rounds each scalar channel of the representation to a fixed set of discrete levels instead of performing nearest-neighbor lookup against learned codebook entries. This design achieves 100% codebook utilization by construction — every combination of rounded scalars is reachable, so collapse is impossible. FSQ is explicitly positioned as a fallback to activate only if VQ-VAE collapse persists despite EMA + dead code reset + k-means initialization + L2 normalization (and optionally entropy regularization). The tradeoff noted elsewhere — less interpretable codes, vocabulary size determined by the product of per-dimension levels rather than a directly set K — remains.

---

Source:
- DECISIONS.md (ADR-007) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- the current approach this would replace
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- the parent architecture
- [[separating representation learning from discretization enables richer feature discovery]] -- the principle both approaches follow
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- FSQ would also need to choose an extraction point from the frozen transformer

Topics:
- [[classification]]
