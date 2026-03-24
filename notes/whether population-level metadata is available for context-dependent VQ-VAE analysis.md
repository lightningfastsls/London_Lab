---
description: Open question on whether mouse ID, sex, strain, and social context metadata are available to enable population-level comparison of VQ-VAE codebook usage.
type: open-question
confidence: speculative
topics:
  - "[[experimental-methods]]"
---

# whether population-level metadata is available for context-dependent VQ-VAE analysis

The scientific value of the VQ-VAE codebook analysis depends critically on the availability of population-level metadata. Comparing codebook usage frequency distributions across mouse groups — wild versus lab, male versus female, isolated versus pair-housed — requires knowing which recording belongs to which group. This metadata (mouse ID, sex, strain, social context, population origin) may be encoded in the WAV directory structure, embedded in filenames, or stored in a separate metadata CSV that has not yet been confirmed to exist.

If metadata is unavailable, context-dependent analysis must be skipped entirely. The codebook can still be interpreted by exemplar galleries and decoder visualization, but the cross-population comparison that motivates the VQ-VAE approach — testing whether [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] — cannot be executed. This would reduce the VQ-VAE from a scientific instrument to a visualization tool.

Resolving this question requires explicit investigation before committing to the VQ-VAE training phase. The recommended action is to audit the 5970 USV recording directory structure and any accompanying experimental logs to determine what metadata is reliably associated with each WAV file. If metadata exists in any parseable form, a metadata manifest (CSV mapping filename to experimental conditions) should be created before VQ-VAE training begins, so that analysis scripts can join on it at inference time. The downstream impact extends to the codebook analysis battery: population-level comparison of [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] usage frequencies requires group labels, and the [[bigram productivity ratio measures compositionality of USV code sequences]] analysis becomes more meaningful when productivity can be compared across populations.

For the statistical methods that would consume this metadata, see [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] and [[MANOVA on CNN features or chi-squared on VQ-VAE codes tests whether behavioral context predicts vocal repertoire composition]]. The metadata is also necessary for [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]], which showed that BootSnap models trained on one population underperform on the other — making population labels essential for training and evaluation design.

---

Source: [ROADMAP](../ROADMAP.md)
