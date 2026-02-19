---
description: "Despite adjacent approaches narrowing the gap in 2024-2025, no one has published end-to-end VQ-VAE trained on animal vocalizations"
type: finding
confidence: likely
conditions:
  - "literature review as of February 2026"
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[classification]]"
---

# End-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026

A systematic review of published and preprint literature as of February 2026 confirms that the unique combination of **end-to-end trained discrete codebook** applied to **animal vocalizations** has not been published. This updates and strengthens [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] with 2024-2025 evidence.

The gap is narrowing from multiple adjacent directions, but none close it:

| Approach | Discrete? | End-to-end? | Learned codebook? | Applied to animals? |
|----------|-----------|-------------|-------------------|---------------------|
| **Our VQ-VAE** | Yes | Yes | Yes | Planned (mice) |
| Post-hoc VQ on HuBERT (Sarkar 2025) | Yes | No | Partially | Yes (marmosets, dogs) |
| K-means tokens / STSG (2025) | Yes | No | No | Yes (birds, insects) |
| Convolutional AE (Best 2023) | No | Yes | N/A | Yes (6 species) |
| VAE/AVA (Goffinet 2021) | No | Yes | N/A | Yes (mice, finches) |
| AVES SSL (Hagiwara 2023) | No | Yes | N/A | Yes (multiple) |
| FSQ (Mentzer 2024) | Yes | Yes | No (fixed grid) | No |

The field is converging on discrete representations from multiple directions -- self-supervised speech models with post-hoc quantization, convolutional autoencoders for clustering, and VAEs for latent space analysis -- but none combine VQ-VAE's end-to-end learned discrete codebook with animal vocalization data. Both the post-hoc approaches ([[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]]) and the fixed clustering approaches ([[STSG spectrogram token skip-gram achieved only 0.559 AUC versus 0.810 for transfer learning on bioacoustic classification]]) have demonstrated significant performance gaps compared to continuous baselines, motivating the end-to-end approach.

---

Source:
- [[learn-vqvae-bioacoustics-state-of-art-2026-02]] (inbox)

Relevant Notes:
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- the original gap claim, now updated
- [[Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations]] -- closest approach
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] -- continuous embedding approach
- [[FSQ eliminates codebook collapse by construction achieving 100 percent utilization through fixed scalar quantization]] -- end-to-end but not applied to animals

Topics:
- [[representation-learning]]
- [[classification]]
