---
description: "Ivanenko et al 2023 tested AE, U-Net, RNN — all exceeded 90% precision/recall; U-Net and AE achieved over 95% with best generalization on external datasets"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[classification]]"
---

# U-Net semantic segmentation exceeded 95 percent precision recall for USV detection in systematic DL comparison

An extended deep learning comparison study (Ivanenko et al. 2023, Nature Scientific Reports) tested seven methods head-to-head on the same annotated dataset with IoU >= 0.6 threshold. The specific results for the top performers: U-Net achieved 91.1% precision / 92.1% recall with only 187K parameters — one quarter of DeepSqueak's 693K parameters — while DeepSqueak managed only 66.4% precision / 63.7% recall (75.8%/63.7% with denoiser). USVSEG (classical signal processing) achieved 85.7%/88.0%, outperforming DeepSqueak without any deep learning. On an external validation dataset (diverse mouse strains), U-Net maintained its lead: 74.3% precision / 69.1% recall vs DeepSqueak's 61.2% / 56.7%.

The significance of semantic segmentation (U-Net) versus object detection (DeepSqueak's YOLO) is that pixel-level masks provide precise temporal and spectral boundaries, while bounding boxes give only approximate regions. Since [[SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines]], the U-Net paradigm is converging as the preferred deep learning approach for USV detection, replacing the older object-detection paradigm.

For our pipeline, this suggests that if we ever replace the energy-detector + CNN approach, U-Net semantic segmentation would be the strongest alternative.

---

Source: usv-detection-methods-landscape-2024-2026-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines]] -- another semantic segmentation approach for USVs
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- our baseline for comparison

Topics:
- [[detection]]
- [[classification]]
