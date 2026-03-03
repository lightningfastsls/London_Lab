---
description: "AMVOC (Giannakopoulos et al. 2022) is MIT-licensed Python/PyTorch using convolutional autoencoder for unsupervised USV clustering with both batch and real-time modes"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering

AMVOC (Giannakopoulos et al., 2022, *Bioacoustics*) at `github.com/tyiannak/amvoc` is the **best available open-source Python tool** for unsupervised USV analysis. It uses a convolutional autoencoder for feature extraction and clustering of USV spectrograms. Key properties:

- Pure Python 3.8 with PyTorch and scikit-learn
- MIT license
- Supports both offline batch processing and real-time analysis via a Dash web GUI
- Detection module outputs CSVs with onset/offset
- Clustering module processes detected USVs

AMVOC is potentially adaptable to accept externally detected segments if formatted correctly, which makes it relevant to our pipeline where USVs are already detected at F1 91.7%. Its unsupervised autoencoder approach is philosophically aligned with our VQ-VAE strategy -- both learn representations without predefined categories, since [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]]. However, AMVOC uses a standard autoencoder without the discretization step that our VQ-VAE provides.

In the landscape of Python USV tools, AMVOC fills the unsupervised niche while [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] fills the supervised niche. The [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] strategy recommends using both approaches.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Giannakopoulos et al. (2022), *Bioacoustics*

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the continuum finding AMVOC's approach respects
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- supervised complement to AMVOC's unsupervised approach
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] -- similar unsupervised embedding approach across species
- [[No single Python tool cleanly accepts pre-detected USV segments and classifies them into syllable types as of 2026]] -- AMVOC partially addresses but doesn't fully solve this gap
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the strategy that combines tools like AMVOC and BootSnap

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-tools]]
