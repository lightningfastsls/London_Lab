---
source: researcher brain-dump
topic: literature context
date: 2026-02-19
method: structured interview (AskUserQuestion)
---

# Literature Context — Brain Dump

## Detection Tools Landscape

### DeepSqueak (Coffey et al.)
Faster R-CNN / YOLO-based detection + k-means clustering. Widely used in the field. We built our own pipeline for more control over the detection stage — DeepSqueak is a monolithic tool whereas our two-stage approach (energy detector → CNN) allows independent tuning of recall and precision.

### VocalMat
AlexNet CNN, ~86% accuracy on 11 predefined categories. Represents the traditional supervised classification approach.

### MUPET
Gammatone filterbank + unsupervised k-means, discovers 100-140 data-driven types. Notable for being unsupervised — closer to our VQ-VAE philosophy of data-driven discovery than VocalMat's fixed categories.

## Classification and Analysis Literature

### Goffinet et al. (2021, eLife) — AVA tool
VAE-based approach. Key finding: USVs form a continuum rather than discrete clusters. This directly motivated our VQ-VAE approach — imposing discrete codes on a continuum. This is the most important single paper for the classification architecture.

### Chabout et al. (2015)
Males change syllable syntax with social context. Establishes that USV sequences are context-dependent, not random — motivating sequence analysis.

### Hertz et al. (2020)
Sequence statistics carry predictive information. Supports the hypothesis that USV sequences have structure worth modeling.

### Ivanenko et al. (2020)
DNNs achieve 77-84% classifying emitter sex from spectrograms. Demonstrates that spectrograms contain identity information beyond call type.

### Tjandra et al. (Interspeech 2020)
Transformer VQ-VAE for unsupervised unit discovery in human speech, K=128 codes. Closest architectural analog to our approach — but applied to human speech, not animal vocalizations.

## Classification Taxonomy

Traditional taxonomy from Holy & Guo (2005) defines discrete call types. Goffinet et al. (2021) challenged this by showing USVs form a continuum rather than discrete clusters. Our VQ-VAE approach sidesteps predefined categories entirely — letting the codebook discover its own discrete vocabulary from data. No published work has applied VQ-VAE to animal vocalizations, making this a genuine research gap.

## Wild Mouse Literature

Wild mouse vocalization literature is primarily from Michael London's own research group. This is part of why the comparison is novel — most USV research is conducted on inbred lab strains (e.g., C57BL/6), not wild-caught mice.
