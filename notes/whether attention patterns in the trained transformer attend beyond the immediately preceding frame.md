---
description: Open question on whether transformer attention heads learn long-range USV context or collapse to local autoregressive behavior resembling a linear model.
type: open-question
confidence: speculative
topics:
  - "[[representation-learning]]"
---

# whether attention patterns in the trained transformer attend beyond the immediately preceding frame

If the trained transformer's attention patterns concentrate entirely on the 1–2 immediately preceding spectrogram frames, the model is functionally equivalent to a high-capacity autoregressive linear predictor — it learns local spectral momentum but not the temporal structure of USV bouts. This would be a negative result worth documenting but would not validate the transformer choice over simpler baselines.

The diagnostic is straightforward: during staged training, periodically visualize attention weight matrices averaged across heads and layers. The expected positive result is that at least some heads attend to temporally distant events — the onset of the current bout, the previous USV call, or the inter-call silence structure. These long-range patterns are precisely what [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] predicts should be learned if the transformer is capturing genuine temporal context.

Interpreting attention patterns in transformers is itself an open research area. High attention weight to a frame is not equivalent to that frame being causally important to the prediction — attention is a routing mechanism, not a causal graph. However, the absence of any long-range attention (all heads purely local) would be strong evidence that the architecture is not leveraging the long context window. Probing tasks — predicting bout-level statistics from internal representations — provide a complementary diagnostic.

This open question is resolvable only after staged training on real bout data, making it a monitoring priority during [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]]. The answer directly constrains interpretation of [[excess entropy measures long-range structure complexity in discrete code sequences]] -- if attention is purely local, high excess entropy in code sequences would indicate structure imposed by the VQ-VAE quantization rather than learned by the transformer. The training context for this monitoring is described in [[bout-level spectrograms preserve inter-USV timing context for transformer training]].

---

Source: [[ROADMAP.md]]

Relevant Notes:
- [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] -- the architecture whose attention this question evaluates
- [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]] -- monitoring priority during staged training
- [[excess entropy measures long-range structure complexity in discrete code sequences]] -- interpretation depends on whether attention is local or global
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- the data format providing the long context to attend over
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- if attention is purely local, it cannot capture the sequence-level statistics Hertz identified

Topics:
- [[classification]]
