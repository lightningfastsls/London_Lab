---
description: "Best et al 2023 trained one autoencoder on all 6 species simultaneously — performance nearly matched species-specific models, implying universal acoustic features across vocal production systems"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# A generic cross-species autoencoder performs nearly as well as species-specific models suggesting shared vocalization structure

Best et al. (2023) made a striking discovery while evaluating their UMAP+HDBSCAN clustering pipeline: a generic autoencoder trained on all six species datasets simultaneously performed nearly as well as species-specific models trained on individual species. This suggests that vocalization structure shares fundamental properties across taxa — spectral envelopes, temporal modulations, harmonic patterns, and amplitude dynamics are not species-specific but reflect universal constraints of vocal production systems.

This finding has practical implications: rather than training a custom model for each species, a generic bioacoustic autoencoder may suffice as the embedding backbone for clustering pipelines. It also aligns with the broader finding that since [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]], acoustic representations transfer across even the human-animal divide.

For mouse USV analysis, this means embeddings from models trained primarily on bird and frog vocalizations may still capture relevant structure — though the ultrasonic frequency range (20-120 kHz) of mouse USVs could limit transfer from models trained on audible-range vocalizations.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] -- extends with the cross-species transfer finding
- [[Perch 2.0 trained on 14795 species achieves state of the art bioacoustic embeddings that transfer across taxa]] -- Perch's cross-taxa transfer validates this at much larger scale
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] -- consistent cross-domain transfer pattern
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- USV-specific features may already exist as underemphasized directions in generic models, explaining near-parity
- [[GmSLM is a London-Omer collaboration applying self-supervised speech models to marmoset vocalizations]] -- GmSLM applies human speech SSL to marmosets; concrete evidence from our collaborators that cross-species transfer works in practice
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- species-specific counterpoint: AMVOC trained only on mouse USVs achieves strong clustering (37% over handcrafted), so a generic cross-species AE should at least match this baseline to be worth adopting over a species-specific tool

Topics:
- [[bioacoustic-ssl]]
