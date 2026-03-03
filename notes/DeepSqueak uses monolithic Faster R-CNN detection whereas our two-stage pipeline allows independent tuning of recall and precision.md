---
description: "DeepSqueak (Coffey et al.) is the most widely used USV tool but its monolithic architecture doesn't separate recall from precision tuning"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision

DeepSqueak (Coffey et al., 2019, *Neuropsychopharmacology*; `github.com/DrCoffey/DeepSqueak`, ~417 stars, BSD-3-Clause) is the most widely used tool for USV detection and analysis. Its architecture has evolved: versions 1-2 used Faster R-CNN; [[DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection]] in v3.1; v3.2 added high-precision networks and automatic horizontal noise band removal.

**System requirements are non-trivial:** MATLAB 2020a+ with seven toolboxes (Deep Learning, Computer Vision, Image Processing, Curve Fitting, Parallel Computing, Statistics and Machine Learning, Signal Processing). GPU with >=2 GB VRAM recommended. Runs on Windows, macOS, and Linux. No Python port exists.

**File format support:** Accepts WAV, FLAC, and Ultravox (.UVD) files. Only the first channel of multichannel files is processed. Accepts any sample rate because FFT windows are specified in duration not samples (see [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]]). **Output is primarily MATLAB .mat files** (readable via `scipy.io.loadmat()` or `h5py` for v7.3/HDF5 format). Excel export produces .xlsx with 16 metrics per call (ID, label, begin/end time, call length, principal frequency, low/high frequency, bandwidth, frequency SD, slope, sinuosity, mean power, tonality, peak frequency). Also exports Raven selection tables and spectrogram images. No native CSV or SQLite output.

However, its monolithic detection approach does not allow independent control of recall and precision — the detector either finds a region or it doesn't. Our pipeline was built specifically for more control: since [[two-stage detection uses permissive energy detector followed by CNN precision filter]], we can tune the energy detector for maximum recall (via [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]]) independently of the CNN's precision filtering. This separation is a deliberate architectural choice motivated by the specific needs of the wild vs lab mouse comparison study.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- Coffey et al. (2019), *Neuropsychopharmacology* -- DeepSqueak
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23) -- version history, system requirements, file formats

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- our architectural response to DeepSqueak's limitation
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- the general pattern behind our approach
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- another detection tool comparison
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- pragmatic strategy: use DeepSqueak's classification immediately while building our VQ-VAE pipeline
- [[LMT USV Toolbox provides Python-based offline USV processing as a reference implementation]] -- another USV tool in the competitive landscape
- [[DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection]] -- version evolution detail
- [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]] -- STFT parameter specifics
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- the classification side
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the interchange format for feeding our detections into DeepSqueak's classification
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- DeepSqueak recomputes spectrograms from WAV files, so exported bounds need only be approximate

Topics:
- [[detection]]
- [[classification-tools]]
