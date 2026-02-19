---
description: Replacing a hidden state with a decoded codebook entry then generating autoregressively reveals what acoustic future each discrete concept implies.
type: method
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# concept injection decodes what each codebook entry predicts as acoustic continuation

Concept injection operationalizes the question "what does code k mean acoustically?" by forcing the model to act on each code and observing the consequences. The procedure: take a real context sequence (frames 0..t-1), replace the hidden state at time t with the VQ-VAE decoder output for codebook entry k, then run the transformer autoregressively forward for t+1 through t+N steps, recording the predicted spectrogram frames. The result is a spectrogram segment representing "what the model predicts will happen next, given this context followed by concept k."

Concept scanning extends this to systematic exploration: hold the context fixed, inject each of the K codebook entries at position t, record the predicted next frame for each entry. This produces a K×170 matrix — K rows (one per codebook entry) × 170 frequency bins — showing the distinctive acoustic prediction associated with each discrete concept. Clustering the rows of this matrix identifies functionally similar concepts: entries that predict similar acoustic continuations despite being distinct codes. Such clusters might correspond to recognizable USV patterns (frequency rise, flat segment, descending sweep).

This method connects directly to [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] — the concept injection analysis will reveal whether 64 codes produce interpretably distinct acoustic predictions or whether many codes are functionally redundant. It also depends on [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]], since injection at a middle layer allows the remaining transformer depth to translate the injected concept into acoustic predictions. The broader goal of interpretability connects to [[separating representation learning from discretization enables richer feature discovery]], where the two-stage approach is justified in part by the interpretability gains from explicit discretization. Concept injection provides the generative complement to [[exemplar galleries ground abstract codebook entries in concrete acoustic examples]] (what real data was assigned to each code) and [[VQ-VAE codebook visualization decodes entries through the full pipeline back to spectrogram space]] (what the model predicts at inference). These three methods form a complete interpretability toolkit: observed inputs, predicted outputs, and counterfactual generations.

---

Source: [[ROADMAP.md]], Phase 8

Relevant Notes:
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- whether 64 codes produce distinct acoustic predictions or redundant ones
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- injection at a middle layer allows remaining depth to translate concept to acoustics
- [[exemplar galleries ground abstract codebook entries in concrete acoustic examples]] -- observed input complement to concept injection's generative approach
- [[VQ-VAE codebook visualization decodes entries through the full pipeline back to spectrogram space]] -- predicted output complement forming the interpretability toolkit
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- concept injection results differ by extraction layer

Topics:
- [[classification]]
