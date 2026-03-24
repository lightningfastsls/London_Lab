---
description: "DeepSqueak's MATLAB dependency with 7 required toolboxes and no Python reimplementation makes it unusable in Python-first pipelines without a bridge strategy"
type: finding
confidence: proven
conditions:
  - as of DeepSqueak v3.2, February 2026
meta_state: current
source: "inbox/deepsqueak-usv-syllable-classification-practical-guide.md"
topics:
  - "[[classification]]"
---

# DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port

DeepSqueak (Coffey et al., 2019, *Neuropsychopharmacology*) is entirely MATLAB-dependent. Running it requires MATLAB 2020a or later plus seven toolboxes: Deep Learning, Computer Vision, Image Processing, Curve Fitting, Parallel Computing, Statistics and Machine Learning, and Signal Processing. A GPU with at least 2 GB VRAM is recommended but not required — without one, processing slows dramatically.

Extensive searching confirms no "pysqueak" or Python reimplementation exists. This MATLAB lock-in is the primary reason our pipeline requires a bridge strategy (Raven selection table export/import) rather than direct Python API integration with DeepSqueak.

The tool runs on Windows, macOS, and Linux, and is hosted at `github.com/DrCoffey/DeepSqueak` (~417 stars, BSD-3-Clause license).

---

Source:
- Coffey et al. (2019), Neuropsychopharmacology — original DeepSqueak paper
- Compass synthesis: inbox/deepsqueak-usv-syllable-classification-practical-guide.md

Relevant Notes:
- [[DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection]] -- detection architecture evolution
- [[DeepSqueak is fundamentally GUI-centric with no officially supported headless or scriptable operation]] -- compounding constraint beyond MATLAB dependency
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the bridge format that works around this limitation
- [[Reading DeepSqueak mat outputs in Python uses scipy loadmat for v5 format or h5py for v7.3 HDF5 format]] -- the Python interop needed because of this MATLAB lock-in
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- strategic decision to use DeepSqueak despite this constraint

Topics:
- [[classification-tools]]
