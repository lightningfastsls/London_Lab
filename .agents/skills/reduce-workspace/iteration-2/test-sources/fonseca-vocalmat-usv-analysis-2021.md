---
description: "VocalMat: automated detection and classification of mouse USVs using computer vision and machine learning without user-defined parameters"
source_type: paper
url: "https://elifesciences.org/articles/59161"
author: "Antonio HO Fonseca, Gustavo M Santana, Gabriela M Bosque Ortiz, Sergio Bampi, Marcelo O Dietrich"
date_accessed: "2026-02-27"
status: unprocessed
---

# Analysis of Ultrasonic Vocalizations from Mice Using Computer Vision and Machine Learning (Fonseca et al., 2021)

Published in eLife, March 2021. Yale School of Medicine + Universidade Federal do Rio Grande do Sul.

## Core Contribution

VocalMat is a software tool for automatic detection and classification of mouse USVs that requires no user-defined parameters. It combines image-processing and differential geometry for detection with computer vision and machine learning for classification into 11 distinct categories.

## The USV Detection Problem

Mouse USVs typically range from 30-110 kHz and last 5-200 ms. Manual labeling is time-intensive and subjective — different human annotators disagree on USV boundaries and classifications. Existing tools (MUPET, DeepSqueak) require user-defined threshold parameters, making results sensitive to parameter choices and reducing reproducibility.

## VocalMat Detection Pipeline

1. **Spectrogram Generation**: Convert raw audio to high-resolution spectrograms via short-time Fourier transformation
2. **Contrast Enhancement**: Improve signal-to-noise ratio in spectrogram images
3. **Adaptive Thresholding**: Binarize spectrogram without fixed threshold — adapts to local noise floor
4. **Morphological Operations**: Refine segmented objects (close gaps, remove artifacts)
5. **Local Median Filter**: Novel contribution — eliminates noise artifacts by comparing each detected element's contrast against the local median. Elements whose contrast ratio falls below the median are classified as noise.
6. **Differential Geometry Analysis**: For segmented USVs, extract geometric features (curvature, length, frequency modulation) that serve as classification features

The key innovation is the parameter-free detection: adaptive thresholding and the local median filter together handle varying noise conditions without user intervention.

## Classification System

### 11 USV Categories
Based on spectral shape and frequency modulation patterns:
1. Complex
2. Two-syllable
3. Upward frequency modulation
4. Downward frequency modulation
5. Flat
6. Short
7. Chevron (inverted-U)
8. Reverse chevron (U-shaped)
9. Multi-step frequency changes
10. Frequency step up
11. Frequency step down

### Machine Learning Approach
- Supervised classification using extracted features (duration, frequency range, frequency modulation rate, harmonics)
- Training on manually labeled USV library
- ~86% classification accuracy across 11 categories

## Quantitative Results

### Detection Performance
- **Sensitivity**: Detected 4,428 of 4,441 manually labeled USVs (>99.7% detection rate)
- **Specificity**: From 7,741 candidate detections, correctly identified 4,428 as real USVs and rejected 3,300 noise artifacts
- **Only 13 USVs missed** in validation set
- Performance maintained across different mouse strains, ages, and recording conditions

### Comparison with Other Tools
- MUPET: Good detection but limited classification (4 categories only), requires manual threshold setting
- DeepSqueak: Uses neural networks for detection but relies on user-defined confidence thresholds
- VocalMat: No user parameters needed, 11-category classification, comparable or better detection rates

### Classification Accuracy
- ~86% correct classification across 11 categories
- Best performance on flat calls (most distinctive spectral shape)
- Worst performance distinguishing complex vs. multi-step categories (overlapping features)
- Confusion matrix shows most errors occur between acoustically similar categories

## Dataset and Experimental Conditions

Analyzed >4,000 USVs from:
- Multiple mouse strains (C57BL/6J, BALB/cJ, and others)
- Different ages (juvenile and adult)
- Various social contexts (male-female interaction, pup isolation calls)
- Different recording setups and microphone configurations

## Key Insights for the Field

### 1. Parameter-free detection is achievable
The adaptive thresholding + local median filter approach eliminates the single biggest source of variability in USV research: threshold selection. This is a significant reproducibility improvement.

### 2. 11 categories may not capture full repertoire
The classification scheme is based on traditional ethological categories. The paper acknowledges these discrete categories may not capture the full continuous variation in USV acoustics — some USVs fall between categories or exhibit features of multiple types.

### 3. Dimensionality reduction reveals structure
When USVs are projected into 2D space (t-SNE/UMAP), clear clusters emerge that generally correspond to the 11 categories — but with significant overlap zones. This suggests the true acoustic space is more continuous than discrete.

### 4. Batch effects in USV research
Different recording conditions (microphone distance, room acoustics, recording hardware) introduce systematic biases. VocalMat's adaptive approach mitigates but does not eliminate these batch effects.

## Tensions and Open Questions

1. **Discrete categories vs. continuous acoustic space**: The 11-category system is convenient for statistical analysis but may lose information. Is a continuous representation (like embeddings or latent codes) more appropriate?

2. **Detection threshold implicit in morphological operations**: While VocalMat claims to be "parameter-free," the morphological operation parameters (kernel size, iterations) are fixed design choices that implicitly set detection sensitivity. These were tuned on specific recording conditions.

3. **Generalization across species**: VocalMat is designed for mouse USVs specifically. The frequency range, duration assumptions, and classification categories are mouse-specific. How well would the adaptive detection approach generalize to rat USVs (which are lower frequency and longer duration)?

4. **Comparison fairness**: DeepSqueak uses deep learning for detection while VocalMat uses image processing — the comparison may not be apples-to-apples since DeepSqueak can be retrained for new conditions while VocalMat's pipeline is fixed.

## Technical Details

- Spectrogram parameters: Not specified in abstract but uses STFT with parameters optimized for 30-110 kHz range
- The local median filter is the key innovation: for each detected element, compute contrast ratio = (element_intensity - local_background) / local_background. If ratio < local_median_ratio, classify as noise.
- Differential geometry features: curvature computed along the frequency contour of each USV, providing shape information independent of absolute frequency or duration
