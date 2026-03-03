---
description: "DeepSqueak versions 1-2 used Faster-RCNN; v3.1 adopted YOLO v2 for faster and more accurate USV bounding box detection in spectrograms"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection

DeepSqueak (Coffey et al., 2019, *Neuropsychopharmacology*) underwent a significant architecture change between major versions. Versions 1-2 used Faster R-CNN for detecting USV bounding boxes in spectrograms; **version 3.1 switched to YOLO v2**, and version 3.2 added high-precision neural networks and automatic horizontal noise band removal. The YOLO v2 transition improved both speed and detection accuracy.

The tool remains MATLAB-only (MATLAB 2020a+ with seven toolboxes: Deep Learning, Computer Vision, Image Processing, Curve Fitting, Parallel Computing, Statistics and Machine Learning, Signal Processing). A GPU with >=2 GB VRAM is recommended. No Python port exists. This contrasts with our pipeline where [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- our two-stage approach was designed precisely because monolithic detection (whether Faster-RCNN or YOLO) does not separate recall from precision control.

The version history is relevant because published papers cite different DeepSqueak versions with different detection backends, which affects reproducibility of reported detection metrics.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Coffey et al. (2019), *Neuropsychopharmacology* -- DeepSqueak original paper

Relevant Notes:
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- our architectural response to DeepSqueak's monolithic approach
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- pragmatic use of DeepSqueak despite architectural limitations
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- the classification side of DeepSqueak
- [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]] -- alternative detection tool with superior metrics but different input requirements

Topics:
- [[detection]]
- [[classification-tools]]
