---
description: "Sternberg et al. 2025 (EMNLP Findings) — joint Mickey London + David Omer paper using SSL on marmoset calls, establishing the direct collaboration path and SSL approach"
type: finding
confidence: proven
conditions: []
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[bioacoustic-ssl]]"
  - "[[classification]]"
---

# GmSLM is a London-Omer collaboration applying self-supervised speech models to marmoset vocalizations

Sternberg, T., London, M., Omer, D., & Adi, Y. (2025). "Generative Marmoset Spoken Language Modeling." Findings of EMNLP. This paper applies self-supervised speech models to marmoset vocalizations — a direct collaboration between Mickey London and David Omer (Didi), the senior author of the Oren 2024 vocal labeling paper.

This is strategically significant because:
1. It establishes a **direct collaboration path** between London and Omer labs — our mouse USV work could naturally interface with their marmoset methods
2. It shows the Omer lab is actively exploring **SSL approaches** for primate vocalizations, making the ridge vectorization in Oren 2024 part of a broader methodological toolkit rather than a standalone technique
3. The SSL representation approach may contain additional vectorization/tokenization methods worth extracting in a future deep read

Additionally, Omer wrote a commentary on mouse vocalizations: Omer, D. (2025). "Mouse vocalization: Singing the line." Current Biology, 35(12), R611-R612 — indicating active thinking about mouse vocalizations that is directly relevant to any collaboration path.

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Sternberg et al. (2025). Findings of EMNLP.

Relevant Notes:
- [[Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations]] -- different group's SSL approach to marmoset vocalizations
- [[AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks]] -- SSL in bioacoustics more broadly
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- the non-SSL Omer lab technique
- [[a generic cross-species autoencoder performs nearly as well as species-specific models suggesting shared vocalization structure]] -- GmSLM is concrete evidence from our collaborators that cross-species vocal representation transfer works

Topics:
- [[bioacoustic-ssl]]
- [[classification]]
