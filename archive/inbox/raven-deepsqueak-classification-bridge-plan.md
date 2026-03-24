---
description: "Implementation plan for converting our detection JSONs to Raven selection table format for DeepSqueak classification, plus results ingestion and repertoire statistics"
source_type: conversation
author: "Web Claude + researcher"
date_accessed: "2026-02-23"
status: processed
---

# Raven Selection Table Export Adapter for DeepSqueak Classification

Plan for bridging our CNN detection pipeline with DeepSqueak's syllable classification via the Raven selection table interchange format.

## Key Points
- Raven selection table format (.txt) is the standard interchange format between bioacoustic analysis tools (Raven Pro, DeepSqueak, Audacity, etc.)
- DeepSqueak regenerates its own spectrograms from raw audio, so exported bounding boxes serve as regions of interest, not precise frequency boundaries
- 25,000-125,000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest
- DeepSqueak classification outputs 16 acoustic features per call (ID, Label/Type, Begin Time, End Time, Call Length, Principal Frequency, Low Freq, High Freq, Bandwidth, Freq Std Dev, Slope, Sinuosity, Mean Power, Tonality, Peak Frequency)
- Timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations
- PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether community compositions (here: syllable repertoires) differ between groups
- Shannon entropy H = -sum(p_i * log2(p_i)) quantifies repertoire diversity: higher H means more evenly distributed syllable usage
- Jensen-Shannon Divergence provides a symmetric, bounded [0,1] measure for comparing probability distributions of syllable type usage between populations
- Row-stochastic transition matrices P(type_{t+1}|type_t) capture sequential structure in syllable sequences, testable between populations via Frobenius norm with permutation test
- The chi-squared test on pooled syllable counts provides a simpler alternative when sample sizes are sufficient

## Raw Notes

### Raven Selection Table Format
Tab-separated .txt file with mandatory columns:
```
Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)
1	Spectrogram 1	1	1.7006	1.7420	25000	125000
```
- Selection numbers are 1-indexed
- View is always "Spectrogram 1" for single-view analysis
- Channel is 1 for mono recordings
- Time in seconds, frequency in Hz (NOT kHz)
- Naming convention: `{wav_stem}.Table.1.selections.txt`

### DeepSqueak Classification Pipeline
DeepSqueak (Coffey et al., 2019) provides:
1. Detection (Faster R-CNN) — we skip this, using our own two-stage pipeline
2. Classification — built-in syllable type classification trained on mouse USV datasets
3. Excel export — 16 acoustic features per classified call

The 16 features provide rich acoustic characterization:
- Temporal: Begin Time, End Time, Call Length
- Spectral: Principal Frequency, Low Freq, High Freq, Bandwidth, Freq Std Dev, Peak Frequency
- Shape: Slope, Sinuosity
- Energy: Mean Power, Tonality
- Metadata: ID, Label/Type

### Statistical Methods for Repertoire Comparison

**PERMANOVA (Permutational Multivariate Analysis of Variance)**
- Non-parametric test for multivariate community composition
- Uses Bray-Curtis dissimilarity matrix on syllable proportions
- Tests whether group centroids differ in multivariate space
- Borrowed from ecology (Anderson 2001) where it compares species communities
- Advantage: makes no distributional assumptions
- Null hypothesis: no difference in syllable composition between populations

**Shannon Entropy for Diversity**
- H = -sum(p_i * log2(p_i)) where p_i = proportion of syllable type i
- H = 0 when repertoire has only one type (minimum diversity)
- H = log2(K) when all K types equally represented (maximum diversity)
- Comparing H between wild and lab populations tests whether domestication reduced vocal diversity
- Prediction: wild mice should show higher H (more diverse repertoires)

**Jensen-Shannon Divergence**
- JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
- Symmetric (unlike KL divergence) and bounded [0, 1] (when using log2)
- Measures how different two syllable type distributions are
- JSD = 0 means identical distributions; JSD = 1 means completely non-overlapping

**Transition Matrix Comparison**
- Compute per-animal transition matrix P(type_{t+1} | type_t) from time-ordered detections
- Average within each population → population-level transition matrix
- Compare using Frobenius norm: ||M_wild - M_lab||_F
- Statistical significance via permutation test: shuffle population labels, recompute difference
- Identifies specific transitions that differ (e.g., "wild mice more likely to follow type A with type C")

### Why This Bridges Our Pipeline with DeepSqueak
Our detection pipeline has high recall (93.8%) and precision (89.7%) — better than running DeepSqueak detection from scratch. But DeepSqueak has pre-trained syllable classifiers we don't have yet (our VQ-VAE pipeline is still in development). The Raven export adapter lets us use the best of both:
- Our detection (high-quality, validated against ~840 human labels)
- DeepSqueak's classification (pre-trained syllable types)
This gives immediate scientific value while the VQ-VAE pipeline matures.

## Processing Notes
Processed 2026-02-24 via /reduce (seed-004). 8 new notes, 3 enrichments, 1 tension, 1 open question. 2 implementation ideas skipped (already in ROADMAP Phase 14.1/14.2). See queue.json seed-004 for full manifest.
