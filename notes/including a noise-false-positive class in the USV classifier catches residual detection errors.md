---
description: "Adding an explicit noise/FP class to the syllable classifier provides a quality control layer that catches detection errors the upstream pipeline missed"
type: method
confidence: likely
conditions:
  - "sufficient noise training samples available"
meta_state: current
topics:
  - "[[classification]]"
  - "[[detection-landscape]]"
---

# Including a noise-false-positive class in the USV classifier catches residual detection errors

BootSnap explicitly includes a **noise/false-positive class** alongside USV syllable classes in its classification scheme. This is a practical quality control method: the syllable classifier doubles as a second-pass detection filter, catching residual false positives that the upstream detection pipeline missed.

In our two-stage pipeline where [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] at F1 91.7%, approximately 10% of detections are still errors. A classification stage with an explicit noise class would catch some of these without requiring a separate rejection step. VocalMat's dataset includes 2,083 noise samples alongside 10,871 USVs (see [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]]), providing ready-made training data for this class.

This approach complements our existing noise handling where [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- the classification-stage noise class would specifically target these structural noise mimics that fool the detection CNN but may be distinguishable at the syllable-typing level where finer spectral features are analyzed. It also connects to [[good negative training samples must be unambiguously not USV to prevent label noise]] -- the noise class needs high-quality negative examples.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Abbasi et al. (2022) -- BootSnap noise class design

Relevant Notes:
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- the residual error rate this method addresses
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- the specific FP pattern a noise class targets
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- includes 2,083 noise samples for training
- [[good negative training samples must be unambiguously not USV to prevent label noise]] -- quality criterion for the noise class
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- the tool that demonstrated this practice
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the pipeline architecture that benefits from an additional noise filter

Topics:
- [[classification-methodology]]
- [[detection]]
