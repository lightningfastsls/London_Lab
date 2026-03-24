---
description: "First audio-language foundation model for bioacoustics; 19.6% zero-shot species ID (vs 0.4% baseline); trained on 15K+ hours of bioacoustics, speech, and music"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# NatureLM-audio combines BEATs encoder with Llama 3.1-8B for zero-shot bioacoustic species identification

NatureLM-audio (November 2024, arXiv 2411.07186) represents a new paradigm in bioacoustics: an audio-language model that combines a BEATs audio encoder, Q-Former connector, and Llama 3.1-8B language model. Trained on over 15,000 hours spanning bioacoustic recordings (Xeno-canto, iNaturalist), speech, and music, it achieves zero-shot species identification at 19.6% accuracy on held-out species by scientific name — vastly outperforming 0.4% from general audio baselines. It also performs novel tasks never seen during training, like counting individual callers (38.3% accuracy vs 24.3% random).

The cross-modal training (audio + language) may capture semantic structure that purely acoustic models miss — species descriptions, habitat associations, and behavioral context encoded in the language model could inform acoustic representations. However, the 8B-parameter language model makes NatureLM-audio impractical for embedding extraction compared to smaller specialized models like Perch 2.0 (12M parameters). The model's primary value is in zero-shot classification and open-ended audio understanding, not as an embedding backbone.

For our USV pipeline, NatureLM-audio is more relevant as a conceptual direction (multimodal bioacoustic understanding) than a practical tool. Since [[Perch 2.0 trained on 14795 species achieves state of the art bioacoustic embeddings that transfer across taxa]], the smaller supervised model is far more practical for embedding extraction.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[Perch 2.0 trained on 14795 species achieves state of the art bioacoustic embeddings that transfer across taxa]] -- practical alternative for embedding extraction (12M vs 8B params)
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] -- NatureLM-audio's cross-domain training is consistent with this
- [[self-supervised models detect seasonal vocal plasticity without temporal labels demonstrating unsupervised biological discovery]] — NatureLM-audio's language component could narrate the acoustic patterns that SSL models discover
- [[phylogenetic proximity to humans does not influence transfer learning effectiveness from speech models to animal vocalizations]] — universal acoustic features explain why cross-species audio-language training works

Topics:
- [[bioacoustic-ssl]]
