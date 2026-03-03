---
description: "Running both supervised CNN (for Scattoni category comparability) and unsupervised UMAP+HDBSCAN (for data-driven discovery) hedges the taxonomy uncertainty"
type: method
confidence: likely
conditions:
  - "sufficient labeled data for supervised component"
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# Dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions

The recommended strategy for USV syllable classification combines **supervised classification** (for comparability with published literature using Scattoni categories) with **unsupervised UMAP + HDBSCAN clustering** on spectrogram embeddings (for data-driven discovery). This dual approach hedges against the unresolved taxonomy problem: if discrete categories are real, the supervised branch captures them; if USVs are a continuum as [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]], the unsupervised branch reveals the true structure.

The supervised component uses a CNN fine-tuned on existing labeled data (e.g., [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]]) classifying into standard categories. This enables direct comparison with the published USV literature. The unsupervised component extracts CNN penultimate-layer embeddings and clusters them without predefined categories, potentially revealing population-specific structure that predefined taxonomies miss.

This is especially important because [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- the supervised branch may underperform on wild mice where training data is limited, but the unsupervised branch operates without such bias. The dual approach also aligns with our broader architecture where [[separating representation learning from discretization enables richer feature discovery]] -- keep the representation continuous, then apply both supervised and unsupervised lenses.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the continuum finding that motivates the dual approach
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- training data for the supervised component
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- why the unsupervised complement matters
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- potential tool for the unsupervised component
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- potential tool for the supervised component
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- the risk the dual approach mitigates
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the bridge format enabling supervised classification via DeepSqueak while unsupervised methods are developed in parallel

Topics:
- [[classification-methodology]]
- [[experimental-methods]]
