---
description: "Python reimplementation of MATLAB USVSEG tool — PyQt5 GUI, MIT license, v1.0.2 April 2025; achieved 85.7% precision / 88.0% recall in Ivanenko benchmark outperforming DeepSqueak"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# USVSEG Python port provides signal-processing-based USV segmentation without deep learning

USVSEG Python (MatsumotoJ/usvseg_python, MIT license, v1.0.2 April 2025) is a Python reimplementation of the MATLAB USVSEG tool, using signal-processing-based segmentation with no deep learning. It uses stable spectrogram computation plus dynamic thresholding, reducing background noise variation via flattened spectrograms.

In the Ivanenko et al. 2023 head-to-head comparison, USVSEG achieved 85.7% precision and 88.0% recall — outperforming DeepSqueak (66.4%/63.7%) without any neural network. This makes it a strong baseline demonstrating that classical signal processing methods remain competitive. Its dependencies are minimal: NumPy, SciPy, Matplotlib, OpenCV, soundfile — no GPU required.

For our pipeline, USVSEG Python represents a potential replacement for our energy detector stage if we ever need a more sophisticated classical approach. Since [[entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative]], multiple signal-processing approaches exceed our energy detector's permissive threshold strategy.

---

Source: python-alternatives-deepsqueak-usv-classification-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[U-Net semantic segmentation exceeded 95 percent precision recall for USV detection in systematic DL comparison]] -- U-Net beats USVSEG, but USVSEG needs no training data
- [[entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative]] -- another competitive classical approach
- [[six USV detection architectural approaches span object detection to speech model transfer with distinct tradeoff profiles]] — USVSEG falls in the classical signal processing category of the six-approach taxonomy
- [[the Python vs MATLAB divide in USV tools is shrinking but remains a practical barrier]] — USVSEG's Python port is a concrete example of this divide closing
- [[spectrogram segmentation tools like SqueakOut and VocalMat are binary detectors that cannot separate overlapping USVs]] — USVSEG shares this binary limitation for overlapping calls

Topics:
- [[detection]]
- [[classification-tools]]
