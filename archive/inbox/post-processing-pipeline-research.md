---
title: USV Post-Processing Pipeline — Research Context
source_type: web_claude_session
date: 2026-03-27
topics: detection, classification, bioacoustics methodology
---

# USV Post-Processing Pipeline — Research Synthesis

## Hysteresis (Dual-Threshold) Detection

No existing mouse USV tool (DeepSqueak, DAS, VocalMat, USVSEG, MUPET) uses explicit hysteresis for event detection. They all use single threshold + gap-filling + minimum duration. Hysteresis subsumes and improves on this approach by naturally handling gap-filling (the sustain threshold keeps events alive) and minimum duration (events must reach the onset threshold, which isolated noise rarely does for multiple consecutive windows).

Evidence from DCASE sound event detection (Cances et al., 2019, WASPAA) shows class-dependent post-processing parameters improved F1 from 37.1% to 43.9%. Their dichotomic search method (coarse grid then refine) is efficient for the 4-parameter space. WhaleVAD-BPN (2024, arXiv:2510.21280) demonstrates a comprehensive hysteresis pipeline for bioacoustic detection in whale call detection.

scikit-maad (Ulloa et al., 2021) implements double-threshold hysteresis binarization in Python for ecological acoustics.

## Temperature Scaling for CNN Calibration

Modern CNNs are systematically miscalibrated (Guo et al., 2017, ICML). Temperature scaling divides logits by a single learned scalar T before the sigmoid, making probabilities more interpretable. ROC AUC is invariant to temperature scaling (monotonic transformation), but threshold interpretability improves significantly. This is the simplest effective calibration method — 1 parameter, fits in seconds.

For small validation sets, avoid isotonic regression (O(n) effective parameters, overfits). If T comes out abnormally low (<0.5), consider Attended Temperature Scaling (Mozafari et al., 2019).

## PCEN (Per-Channel Energy Normalization)

PCEN (Lostanlen et al., 2019, PLOS ONE) is the gold standard from bioacoustic literature. It operates at the spectrogram level before the CNN, replacing log-magnitude spectrograms with adaptive normalization. In the BirdVoxDetect system, PCEN reduced false alarm rates by 50x in near-field and 5x in far-field recordings. However, it requires retraining the model. Recommended for next model iteration, not current pipeline.

## Two-Stage Detection for False Positive Reduction

Clarfeld et al. (2025, Ecological Informatics) showed that a secondary logistic regression on features from primary detections achieved 84.5-89.8% accuracy for bioacoustic FP filtering. VocalMat (Fonseca et al., 2021, eLife) uses a two-stage approach: morphological filtering followed by CNN noise classification, achieving >98% detection rate. BootSnap (Abbasi et al., 2022, PLOS Comp Bio) includes an explicit "false positive" class alongside 11 USV categories.

## F-beta Scoring for Bioacoustic Detection

When false negatives are more costly than false positives (as in USV counting for behavioral analysis), F2 score (beta=2) weights recall ~4x more than precision. This is standard practice in bioacoustic evaluation where missed vocalizations bias population-level statistics.

## Event-Level Evaluation Methods

Two standard approaches for matching detected events to ground truth:
1. **Collar-based** (±onset/offset tolerance): Standard in DCASE sound event detection. Allow ±200ms tolerance on boundaries. Better for bioacoustics where event boundaries are inherently uncertain.
2. **IoU-based** (Intersection over Union ≥ threshold): Standard in object detection. Requires ≥50% overlap. More strict, penalizes fragmented detections.

Kershenbaum et al. (2025, Biological Reviews) provides a practical guide for bioacoustic detection validation, recommending transparent reporting of matching criteria.

## Key References

- Cances et al., 2019 (WASPAA): Class-dependent hysteresis + dichotomic search for SED post-processing
- WhaleVAD-BPN, 2024 (arXiv:2510.21280): Comprehensive hysteresis pipeline for bioacoustic detection
- DAS / Steinfath et al., 2021 (eLife): Threshold + gap-fill + min-duration for insect/mouse vocalizations
- DeepSqueak (Coffey et al., 2019, Neuropsychopharmacology): Object detection + tonality filtering for USVs
- VocalMat (Fonseca et al., 2021, eLife): Two-stage detection with curvature-based spectral filtering
- BootSnap (Abbasi et al., 2022, PLOS Comp Bio): Explicit false-positive class in USV classification
- Guo et al., 2017 (ICML): Temperature scaling for CNN calibration
- Clarfeld et al., 2025 (Ecological Informatics): Two-stage models for bioacoustic FP reduction
- Lostanlen et al., 2019 (PLOS ONE): PCEN for adaptive normalization
- Kershenbaum et al., 2025 (Biological Reviews): Practical guide for bioacoustic detection validation
- scikit-maad (Ulloa et al., 2021): Double-threshold hysteresis binarization in Python
