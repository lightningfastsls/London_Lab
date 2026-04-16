---
description: "MobileNetV2-based fully convolutional autoencoder (4.6M params, 18MB); pixel-level USV masks from spectrograms; 64 images in 0.035s on GPU; trained on 12954 spectrograms from 5 mouse strains"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[detection-landscape]]"
  - "[[classification]]"
---

# SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines

SqueakOut (2024, PMC11071348) provides a critical upstream capability for mouse USV analysis: autoencoder-based semantic segmentation of USVs from spectrograms. Using a fully convolutional architecture with MobileNetV2 backbone, skip connections, and hybrid Focal+Dice loss, it achieves a Dice score of 90.22 (vs 63.82 for VocalMat). The lightweight model (4.6M parameters, 18MB) runs inference on 64 spectrograms in under 0.035 seconds on GPU, trained on 12,954 annotated spectrograms from 5 mouse strains (postnatal day 5-15).

The authors explicitly designed SqueakOut to improve downstream unsupervised analysis: "The resulting segmentation masks can be used for downstream analysis using unsupervised methods such as Variational Autoencoders and dimensionality reduction techniques such as UMAP." Better segmentation directly improves clustering quality by removing noise contamination from the input — a principle that validates our own two-stage approach where since [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]], precision in the first stage compounds downstream.

For our pipeline, SqueakOut's pixel-level masks could provide cleaner input to the VQ-VAE than our current bounding-box-style detection, though integrating a separate segmentation model adds complexity.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- our detection feeds downstream analysis similarly
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- same VocalMat dataset used as comparison baseline
- [[AMVOC dual-criterion dynamic spectral thresholding achieved Event F1 90.5 percent outperforming DeepSqueak and VocalMat on the same benchmark]] -- comparable detection quality (Dice 90.2 vs Event F1 90.5) but different representation: SqueakOut produces pixel masks feeding downstream VAE/UMAP, AMVOC produces event boundaries feeding its own AE; the two tools occupy complementary slots in a detection-then-cluster pipeline
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- the downstream "unsupervised methods" SqueakOut masks are designed to feed — AMVOC is the most obvious candidate, taking spectrogram patches that SqueakOut could provide cleaner versions of

Topics:
- [[unsupervised-usv-discovery]]
- [[detection]]
- [[classification-methodology]]
