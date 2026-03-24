---
description: Method for interpreting VQ-VAE codebook entries by decoding each through the full transformer pipeline to produce predicted spectrogram columns.
type: method
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# VQ-VAE codebook visualization decodes entries through the full pipeline back to spectrogram space

Each VQ-VAE codebook entry e_k can be decoded to a human-interpretable representation by routing it through the remaining transformer layers and the output head: e_k → VQ-VAE decoder → reconstructed hidden state h_k → remaining transformer layers → output head → predicted spectrogram column. This reveals the acoustic continuation that each discrete concept implies — what spectral content the model expects to follow when it assigns a frame to codebook entry k.

This decoding visualization is a form of feature inversion: instead of asking "what does the network activate on?", it asks "what does the network predict when this concept is active?" The distinction matters for interpretation. High activation on a feature could reflect many stimulus types; the predicted continuation is a more constrained, interpretable quantity anchored to actual spectrogram space.

Combined with exemplar galleries (the N=10 nearest encoder outputs per entry, with ±50 frames of surrounding spectrogram context), the decoding visualization provides two complementary views: what real acoustic events were assigned to this concept (exemplars), and what acoustic future the concept predicts (decoding). A codebook entry is well-characterized when these two views are mutually consistent — the predicted continuations should resemble the acoustic context observed in the exemplar frames.

t-SNE or UMAP projection of all K codebook vectors colored by the mean frequency of their exemplars provides a global map of the learned concept space, revealing whether the codebook organizes by frequency, call type, temporal context, or some combination. This connects to [[concept injection decodes what each codebook entry predicts as acoustic continuation]] (generative complement), [[exemplar galleries ground abstract codebook entries in concrete acoustic examples]] (observational complement), and [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] (vocabulary size that determines visualization complexity). The visualization depends on which layer's hidden states feed the VQ-VAE, as determined by [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]].

---

Source: [ROADMAP](../ROADMAP.md)
