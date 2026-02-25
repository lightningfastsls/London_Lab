---
description: "Strategic decision: use DeepSqueak's existing classification for immediate repertoire comparison before the custom VQ-VAE pipeline is ready"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations

The core courtship degradation finding can likely be demonstrated before the VQ-VAE pipeline is complete, by using DeepSqueak's built-in classification to categorize each USV call and comparing repertoire distributions between wild and lab populations. This is a strategic decision: the most important scientific question (do wild and lab mice differ in USV repertoire?) does not require a custom pipeline. DeepSqueak, despite its limitations for detection since [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]], has a useful classification system that can provide immediate scientific value. The VQ-VAE + Transformer work then becomes a deeper investigation into whether sequential structure has language-like properties — a separate, more ambitious question.

**Integration pathway for pre-detected USVs:** The **Raven selection table import** (`File -> Import Calls -> Import from Raven`) is the most flexible way to feed external detections into DeepSqueak. Format detected USV timestamps and frequency ranges as a Raven .txt selection table (Begin Time, End Time, Low Freq, High Freq columns). The original audio files must be accessible -- DeepSqueak needs them for spectrogram generation during classification. Pure "classification only" mode without audio access is not possible.

**DeepSqueak classification output: 16 acoustic features per call.** The Excel export provides rich acoustic characterization organized into four groups:
- *Temporal:* Begin Time, End Time, Call Length
- *Spectral:* Principal Frequency, Low Freq, High Freq, Bandwidth, Freq Std Dev, Peak Frequency
- *Shape:* Slope, Sinuosity
- *Energy:* Mean Power, Tonality
- *Metadata:* ID, Label/Type

**Why hybrid detection + classification:** Our detection pipeline has high recall (93.8%) and precision (89.7%) — better than running DeepSqueak detection from scratch. DeepSqueak has pre-trained syllable classifiers we don't have yet (our VQ-VAE pipeline is still in development). The Raven export adapter lets us use the best of both: our detection (validated against ~840 human labels) + DeepSqueak's classification (pre-trained syllable types). This gives immediate scientific value while the VQ-VAE pipeline matures.

**Reading DeepSqueak outputs in Python:** Use `scipy.io.loadmat()` for MATLAB v5 format or `h5py` for v7.3 (HDF5) format. DeepSqueak v3.x may use either. All bounding box, score, type, and audio data are accessible as NumPy arrays after loading. This enables programmatic extraction of DeepSqueak's classification results for downstream statistical analysis without requiring MATLAB.

---

Source:
- Researcher brain-dump on scientific hypotheses (2026-02-19)
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23) -- Raven import pathway, scipy.io.loadmat reading
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23) -- 16 feature list, hybrid detection rationale

Relevant Notes:
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- DeepSqueak's detection limitation vs classification utility
- [[VQ-VAE investigation of language-like sequential structure in USVs is a separate deeper question from courtship degradation]] -- the two-tier research strategy
- [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]] -- the finding this approach would formalize
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question this approach directly serves
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- DeepSqueak classification uses traditional types; VQ-VAE later tests whether continuum-based discretization is more informative
- [[DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection]] -- version context for the tool
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- the unsupervised classification that feeds repertoire comparison
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the interchange format used for the bridge
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- how classified results are matched back to original detections

Topics:
- [[classification]]
- [[experimental-methods]]
