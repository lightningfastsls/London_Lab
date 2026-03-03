---
description: "Taxonomy: object detection (DeepSqueak), semantic segmentation (SqueakOut/U-Net), temporal models (DAS), classical signal processing, speech transfer (WhisperSeg), hybrid pipelines (BootSnap)"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[detection]]"
---

# Six USV detection architectural approaches span object detection to speech model transfer with distinct tradeoff profiles

The USV detection field as of 2024-2026 has converged on six distinct architectural approaches, each with characteristic strengths and weaknesses:

1. **Object detection on spectrograms** (DeepSqueak: Faster R-CNN, YOLO) — treats USVs as visual objects; interpretable bounding boxes but coarse temporal boundaries
2. **Semantic segmentation** (SqueakOut, U-Net) — pixel-level masks; precise boundaries but needs dense annotations
3. **Temporal sequence models** (DAS: TCN, HybridMouse: CNN+BiLSTM) — frame-level classification with temporal context; but DAS requires raw audio
4. **Classical signal processing** (USVSEG, A-MUD, energy detection) — fast, no training data needed; but parameter-sensitive
5. **Speech model transfer** (WhisperSeg, SSL models) — cross-species transfer, few-shot capable; but compute-heavy
6. **Hybrid pipelines** (BootSnap, VocalMat) — end-to-end with multiple stages; but multiple failure points

Our two-stage approach (energy detector + CNN) falls in the hybrid category, combining classical signal processing (stage 1) with deep learning (stage 2). This taxonomic framing helps evaluate future improvements: we could upgrade stage 1 to entropy-based detection, or stage 2 to U-Net segmentation, without changing the hybrid architecture.

---

Source: usv-detection-methods-landscape-2024-2026-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- our pipeline as hybrid
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- the general pattern

Topics:
- [[detection]]
