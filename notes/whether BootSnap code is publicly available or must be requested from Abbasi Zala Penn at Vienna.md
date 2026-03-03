---
description: "BootSnap (Abbasi et al. 2022) has no confirmed public GitHub repo -- the code may need to be requested from the authors at University of Veterinary Medicine, Vienna"
type: open-question
confidence: speculative
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# Whether BootSnap code is publicly available or must be requested from Abbasi Zala Penn at Vienna

BootSnap (Abbasi et al., 2022, *PLOS Computational Biology*) is the closest conceptual match to our classification needs -- a Python CNN that classifies pre-detected USVs into 12 syllable types using gammatone spectrograms. However, there is **no confirmed public GitHub repository**. The code may need to be requested directly from the authors (Abbasi, Zala, Penn) at the University of Veterinary Medicine, Vienna.

This is a practical blocker for the strategy outlined in [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]], since BootSnap would be the ideal supervised component. If the code is unavailable, we would need to reimplement the approach based on the paper's description:
- Gammatone spectrogram extraction (see [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]])
- CNN with snapshot ensemble learning
- 12 syllable classes + noise class (see [[including a noise-false-positive class in the USV classifier catches residual detection errors]])

BootSnap's labeled data from wild-derived and lab mice is available through the PLOS Computational Biology supplementary materials, which is valuable regardless of whether the code itself is accessible.

**Action items:**
1. Check PLOS Comp Bio supplementary materials for code/data links
2. Search GitHub for "bootsnap USV" or similar
3. If not found, email corresponding author requesting code access

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Abbasi et al. (2022), *PLOS Computational Biology*

Relevant Notes:
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- the tool's performance
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- the spectral representation choice
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the strategy that could use BootSnap
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- BootSnap uniquely addressed this
- [[no Python USV tool cleanly accepts pre-detected segments for classification creating an integration gap]] -- BootSnap is the best candidate to fill this gap

Topics:
- [[classification-tools]]
