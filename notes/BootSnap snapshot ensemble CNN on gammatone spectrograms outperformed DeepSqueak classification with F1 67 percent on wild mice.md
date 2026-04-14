---
description: "BootSnap (Abbasi et al. 2022, PLOS Comp Bio) uses gammatone spectrograms + CNN snapshot ensemble for 12-class USV classification, achieving F1 67% on wild mice vs DeepSqueak's 41%"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice

BootSnap (Abbasi et al., 2022, *PLOS Computational Biology*) was explicitly designed to classify pre-detected USVs into **12 syllable types** using gammatone spectrograms fed into a CNN with snapshot ensemble learning. It is Python-based and the closest conceptual match to our pipeline's needs -- a classification stage that operates on already-detected USV segments.

Key performance results: macro F1 of **67% on wild mice** versus DeepSqueak's 41%, and **F1 67-74.5%** across wild and lab mice. BootSnap also showed the best cross-generalization between wild-derived and laboratory mice. Critically, it found that t-SNE distributions differ substantially between wild and lab mice for certain classes (inverted-U, complex), while simpler categories (up, down, flat, short) show large overlap -- evidence that [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]].

BootSnap's use of gammatone spectrograms is notable because [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]], and it includes a noise/false-positive class to catch residual detection errors -- a practice we should adopt since [[including a noise-false-positive class in the USV classifier catches residual detection errors]].

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Abbasi et al. (2022), *PLOS Computational Biology*

Relevant Notes:
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- the spectral representation advantage
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- key cross-population finding
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- practical quality control method from BootSnap
- [[whether BootSnap code is publicly available or must be requested from Abbasi Zala Penn at Vienna]] -- unresolved access question
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- our research question that BootSnap's cross-population results directly inform
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- another gammatone-based tool
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- BootSnap could serve as the supervised branch of the dual approach, especially for wild mice where it achieves the best F1

Topics:
- [[classification-tools]]
- [[experimental-methods]]
