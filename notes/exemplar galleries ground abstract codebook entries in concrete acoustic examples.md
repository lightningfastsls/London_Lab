---
description: Method of finding N=10 nearest encoder outputs per codebook entry and extracting ±50 frames of surrounding spectrogram context to make discrete concepts interpretable.
type: method
confidence: experimental
topics:
  - "[[classification]]"
---

# exemplar galleries ground abstract codebook entries in concrete acoustic examples

For each codebook entry k, exemplar galleries are constructed by finding the N=10 encoder outputs (from real recording data) with smallest L2 distance to e_k in the hidden state space, then extracting the surrounding spectrogram context (±50 frames, approximately ±85ms at the standard STFT hop size) for each exemplar. Displaying these ten clips side by side shows what real acoustic events the model assigns to concept k.

Without exemplar galleries, codebook entries are abstract vectors in a high-dimensional space — meaningful to the model but opaque to the researcher. The galleries provide the interpretive bridge between discrete symbol and acoustic reality. A codebook entry that clusters whistle calls with rising frequency profiles can be recognized as such from the exemplar spectrograms, even if the entry vector itself carries no human-readable label.

The ±50 frame context window is chosen to include the call that triggered the assignment plus neighboring calls. This is important because a codebook entry may represent not a call type per se but a temporal context — for example, "the first call in a new bout" or "a call following a long silence." Context windows too narrow would miss this temporal structure; too wide would include so much surrounding content that the distinctive trigger becomes obscured.

Exemplar galleries complement [[VQ-VAE codebook visualization decodes entries through the full pipeline back to spectrogram space]], which provides the predictive view. Together they constitute a full characterization: what the model has seen (exemplars) and what it expects to come next (decoding). The choice of which hidden layer feeds the VQ-VAE is discussed in [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]].

---

Source: [[ROADMAP.md]]
