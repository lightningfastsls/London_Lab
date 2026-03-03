---
description: "BootSnap's key finding: supervised USV classifiers trained on one population perform poorly on the other, necessitating cross-population or fine-tuned training"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# Classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data

BootSnap (Abbasi et al., 2022) demonstrated a key finding for our research: **classifiers trained on one mouse population generalize poorly to the other**. A model trained on lab mice underperforms when applied to wild mice, and vice versa. This was shown across multiple classifier architectures including both BootSnap's own CNN and DeepSqueak's classification.

The t-SNE distributions from BootSnap show that certain syllable classes differ substantially between wild and lab mice (inverted-U, complex), while simpler categories (up, down, flat, short) show large overlap. This asymmetric generalization gap means that some call types are population-distinctive while others are shared -- directly relevant to [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]].

The practical implication is clear: existing labeled datasets like [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] (from lab mice) will not suffice for accurate classification of wild mouse USVs. We must either:
1. **Label a subset of our own wild mouse USVs** for fine-tuning
2. **Train on pooled wild + lab data** for a generalizable classifier
3. **Use unsupervised methods** like [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] that avoid the population-specificity problem

This finding also supports the [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] strategy -- the unsupervised component may discover population differences that the supervised component misses due to training bias.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Abbasi et al. (2022), *PLOS Computational Biology* -- BootSnap

Relevant Notes:
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question this directly affects
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- lab mouse training data that won't generalize alone
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- the study demonstrating this gap
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- unsupervised alternative avoiding the bias
- [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] -- wild mice have distinct behavioral modulation patterns
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the combined strategy addressing this limitation
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- population-specific repertoire differences this finding predicts would need PERMANOVA to quantify
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- enables population-specific adaptation of a shared base classifier with minimal wild mouse labels
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- small amount of available wild mouse labeled data may suffice for LoRA adaptation

Topics:
- [[classification-methodology]]
- [[experimental-methods]]
