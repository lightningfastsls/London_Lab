---
description: "Su et al 2021 — each dimension pair rotated by angle proportional to position, applied per-layer to Q/K only (not V), used by LLaMA and Mistral and now dominant in practice"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# RoPE encodes relative position through rotation of Q and K vectors where the dot product naturally incorporates the position difference between tokens

Rotary Position Embedding (RoPE, Su et al., 2021) takes a fundamentally different approach to position encoding: instead of adding position information to the input, it encodes position through rotation of the Q and K vectors. Each pair of dimensions (q_{2i}, q_{2i+1}) is rotated by an angle proportional to position using a 2D rotation matrix with angle m·theta_i, where theta_i follows the same geometric frequency progression as sinusoidal encoding.

The key insight is that when two rotated vectors are dot-producted, the angular difference between them naturally appears in the result. The dot product <R(m)q, R(n)k> incorporates (m-n) as a relative position signal — the model receives relative position information without any explicit relative position computation. Note that the dot product is a function of both the token embeddings and the relative position difference, not purely angular difference.

RoPE is applied to Q and K only, not to V. This is a principled choice: position information is needed only where comparison happens — in the attention score computation via Q·K. Values carry content information that should not be position-warped. This is applied at every attention layer, not just at the input, giving each layer fresh position information.

Key differences from sinusoidal encoding: sinusoidal is additive (adds to embeddings at input), RoPE is multiplicative (rotates vectors at every layer). Sinusoidal mixes position with semantics at the input; RoPE keeps them more separate. RoPE provides better extrapolation than sinusoidal but still degrades beyond training length without modifications like position interpolation, NTK-aware scaling, or YaRN. Despite this, RoPE has become the dominant positional encoding in practice, used by LLaMA, Mistral, Qwen, and most major open-source LLMs.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[sinusoidal positional encoding is added not concatenated to token embeddings preserving dimension while forcing position-content interaction]] -- the additive predecessor
- [[ALiBi adds linear distance penalties to attention scores enabling train-short-test-long extrapolation with equivalent perplexity at 2x and reasonable degradation at longer ranges]] -- the score-bias alternative
- [[positional encoding diversified from additive sinusoidal and learned embeddings to multiplicative RoPE and score-bias ALiBi with RoPE becoming dominant in practice]] -- the broader evolutionary pattern

Topics:
- [[transformer-architecture]]
