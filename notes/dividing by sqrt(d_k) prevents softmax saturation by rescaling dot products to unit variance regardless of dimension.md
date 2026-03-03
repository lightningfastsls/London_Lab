---
description: "Dot product variance grows linearly with d_k — without scaling, large magnitudes push softmax into near-one-hot saturation with vanishing gradients, stalling training"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# dividing by sqrt(d_k) prevents softmax saturation by rescaling dot products to unit variance regardless of dimension

The attention formula divides the Q·K dot product by sqrt(d_k) before applying softmax: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V. This scaling factor is not cosmetic — it prevents a critical training pathology.

When Q and K have components drawn from standard normal distributions (mean 0, variance 1), their dot product has mean 0 but variance d_k. This is because the dot product is a sum of d_k independent products, each with variance 1. As d_k grows (typically 64 per head in standard architectures), dot product values become large in magnitude.

Large-magnitude inputs to softmax push it into saturation regions where the output is nearly one-hot: one value close to 1, all others close to 0. In this regime, the softmax gradient becomes vanishingly small — the Jacobian entries approach zero for all but the dominant element. This causes both vanishing gradients (weight updates become negligible) and attention collapse (model locks onto single tokens rather than distributing attention across relevant context).

Dividing by sqrt(d_k) rescales the dot products to have unit variance regardless of dimension. With d_k = 64, unscaled dot products might range from -20 to +20, producing near-one-hot softmax outputs. After dividing by sqrt(64) = 8, the range compresses to approximately -2.5 to +2.5, where softmax produces smooth distributions with healthy gradients. This is a self-adjusting mechanism — if you change model dimension, the scaling adjusts automatically.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[unscaled attention dot products grow with dimension causing softmax collapse to one-hot distributions with vanishing gradients]] -- the problem this solves
- [[Q-K-V separation enables asymmetric context-dependent relevance matching through three independently specialized projections]] -- the projections whose outputs are being scaled

Topics:
- [[transformer-architecture]]
