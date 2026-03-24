---
description: "Vaswani 2017 used element-wise addition of same-dimensioned PE vectors — avoids parameter bloat of concatenation but forces model to disentangle position from semantics in shared space"
type: decision
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# sinusoidal positional encoding is added not concatenated to token embeddings preserving dimension while forcing position-content interaction

The original transformer (Vaswani et al., 2017) assigns each absolute position a unique vector using sine and cosine functions at different frequencies: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)). These position vectors have the same dimensionality as the token embeddings and are added element-wise.

Addition rather than concatenation was a deliberate design choice with clear tradeoffs. Addition preserves the embedding dimension, avoiding the parameter bloat that concatenation would introduce (doubling the input dimension to all subsequent layers). It also requires no additional hyperparameter for the position embedding size. The model can learn to use different embedding dimensions for positional versus semantic information through the training process.

The tradeoff is that addition forces position and content to interact in the same vector space. The network must learn to disentangle positional information from semantic content — they share the same dimensions and can interfere. The authors also hypothesized that the sinusoidal functions would enable the model to learn relative position attention, since PE(pos+k) can be expressed as a linear transformation of PE(pos) for any fixed offset k.

However, empirical evaluation revealed significant limitations. Sinusoidal encoding has very limited extrapolation capability beyond training sequence lengths. This drove the development of more sophisticated approaches: since [[RoPE encodes relative position through rotation of Q and K vectors where the dot product naturally incorporates the position difference between tokens]] and [[ALiBi adds linear distance penalties to attention scores enabling train-short-test-long extrapolation with equivalent perplexity at 2x and reasonable degradation at longer ranges]], both addressing extrapolation more effectively.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[RoPE encodes relative position through rotation of Q and K vectors where the dot product naturally incorporates the position difference between tokens]] -- multiplicative successor addressing extrapolation
- [[ALiBi adds linear distance penalties to attention scores enabling train-short-test-long extrapolation with equivalent perplexity at 2x and reasonable degradation at longer ranges]] -- score-bias alternative

Topics:
- [[transformer-architecture]]
