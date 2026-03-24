---
description: "GPT-2 pioneered pre-norm over Vaswani's post-norm — in 2024-2025, Peri-LN emerged as a third approach normalizing at both input and output of sublayers, distinct from simple post-norm return"
type: pattern
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# pre-norm leaves the residual path untouched enabling stable gradient flow while OLMo 2 and Gemma 3 adopted hybrid peri-normalization combining pre-norm and output-norm with QK-Norm

The original transformer (Vaswani, 2017) used post-norm: apply sub-layer, add residual, then normalize. This means the normalization sits on the residual path, through which gradients must flow during backpropagation. In deep models, this can cause gradient instability.

GPT-2 (Radford, 2019) switched to pre-norm: normalize first, then apply sub-layer, then add residual. This leaves the residual connection as a clean identity shortcut — gradients flow directly through skip connections without passing through normalization. Pre-norm dramatically stabilized training for deep models and became the default for almost all modern LLMs (GPT-2, GPT-3, LLaMA, Falcon, Mistral).

In 2024-2025, a more nuanced picture emerged. OLMo 2 and Gemma 3 adopted what Kim et al. (2025) call "Peri-LN" — normalization applied both before AND after sublayers, combined with QK-Norm (normalizing Q and K before attention scoring to stabilize attention logits). This is NOT a simple "return to post-norm" but a genuinely new hybrid approach that aims to combine pre-norm's gradient stability with post-norm's quality benefits.

The distinction matters: post-norm places normalization only after the sublayer on the residual path, while Peri-LN places normalization at both input and output positions, creating a different gradient flow pattern. This is an active area of architectural research with the field still converging on best practices.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[pre-norm transformer architecture improves training stability for spectrogram prediction]] -- our USV project's application of pre-norm
- [[transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently]] -- the sublayers being normalized

Topics:
- [[transformer-architecture]]
