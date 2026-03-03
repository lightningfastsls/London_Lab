---
description: "The variance of q·k equals d_k for unit-variance components — at d_k=64, dot products reach ±20, pushing softmax into saturation where gradient flow effectively stops"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# unscaled attention dot products grow with dimension causing softmax collapse to one-hot distributions with vanishing gradients

The dot product between query and key vectors has a variance problem that grows with dimensionality. If the components of q and k are independent with mean 0 and variance 1, their dot product q·k = sum of d_k products, each with variance 1. By the properties of variance for sums of independent variables, Var(q·k) = d_k.

This means dot product magnitudes scale with sqrt(d_k). At typical per-head dimension d_k = 64, dot products can easily reach ±16 to ±20. When these large values enter softmax, the exponential function amplifies the differences enormously — e^20 is about 5×10^8 while e^(-20) is about 2×10^-9. The result is a nearly one-hot distribution where essentially all probability mass concentrates on the single highest-scoring key.

This creates a compound problem. First, attention becomes "hard" rather than "soft" — instead of blending information from multiple relevant positions, the model fixates on a single token, losing the weighted-mixture property that makes attention powerful. Second, the softmax Jacobian entries approach zero in the saturated regime, creating vanishing gradients that stall learning.

Vaswani et al. (2017) explicitly note this concern: "we suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients." The fix is since [[dividing by sqrt(d_k) prevents softmax saturation by rescaling dot products to unit variance regardless of dimension]].

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[dividing by sqrt(d_k) prevents softmax saturation by rescaling dot products to unit variance regardless of dimension]] -- the solution to this problem

Topics:
- [[transformer-architecture]]
