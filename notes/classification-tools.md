---
description: DeepSqueak interop, Python USV classification tools landscape, and Raven interchange for bridging detection pipelines
type: moc
parent_map: classification
topics: "[[classification]]"
---

# classification-tools

Tools and interoperability for USV classification. DeepSqueak remains the dominant MATLAB tool despite its GUI-only design. The Python landscape has no single drop-in replacement, but compositional strategies combining detection + feature extraction + clustering outperform monolithic approaches. The Raven selection table format bridges between tools.

## DeepSqueak

- [[DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port]] -- MATLAB lock-in with 7 required toolboxes, no Python reimplementation
- [[DeepSqueak is fundamentally GUI-centric with no officially supported headless or scriptable operation]] -- no CLI/API, batch ops require GUI initiation
- [[DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection]] -- detection architecture evolution
- [[DeepSqueak v3.1 added VAE-based contour-invariant clustering as upgrade over k-means for continuous USV variation]] -- VAE clustering alongside k-means and ARTwarp
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- unsupervised clustering yielding k=20
- [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]] -- STFT parameters favoring frequency resolution
- [[DeepSqueak uses constant-duration FFT windows making it inherently sample-rate agnostic]] -- duration-based windows accept any sample rate including 300 kHz
- [[DeepSqueak Excel export provides 16 per-call metrics including principal frequency bandwidth slope and tonality]] -- richest structured output (16 acoustic features per call)
- [[Reading DeepSqueak mat outputs in Python uses scipy loadmat for v5 format or h5py for v7.3 HDF5 format]] -- Python interop for reading .mat results
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- competitive positioning vs our architecture

## Python USV Classification Tools (Landscape)

- [[No single Python tool cleanly accepts pre-detected USV segments and classifies them into syllable types as of 2026]] -- landscape gap motivating custom pipeline development
- [[three viable Python strategies for replacing DeepSqueak target segmentation-first unsupervised discovery and supervised classification]] -- three compositional strategies outperform monolithic MATLAB
- [[USVSEG Python port provides signal-processing-based USV segmentation without deep learning]] -- MIT license, v1.0.2; 85.7%/88.0% outperforms DeepSqueak without NN
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- best supervised classifier for pre-detected USVs, macro F1 67% on wild mice
- [[BootSnap includes an explicit false-positive class alongside 11 USV syllable categories]] -- unified noise class competing in softmax as alternative to two-stage FP filtering
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- MIT-licensed Python autoencoder, adaptable to external detections
- [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]] -- highest detection metrics but raw-audio-only
- [[WhisperSeg adapts OpenAI Whisper transformer for animal vocalization segmentation with positive cross-species transfer]] -- Whisper-based, outperforms DAS but raw-audio-only
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- largest freely available labeled USV dataset (10,871 USVs + 2,083 noise)
- [[whether BootSnap code is publicly available or must be requested from Abbasi Zala Penn at Vienna]] -- unresolved access question for the best-fit tool

## DeepSqueak Bridge & Raven Interchange

- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- tab-separated .txt format used by Raven Pro, DeepSqueak, Audacity
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- frequency bounds need only be approximate
- [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] -- cross-tool frequency convention (vs our 20-120 kHz)
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- re-associating DeepSqueak results with our detections
- [[DeepSqueak import previously required exact subdirectory name matches while Raven export already supported prefix matches creating a silent asymmetric round-trip]] -- the 2026-03-07 bug: export supported prefix match, import required exact name, breaking round-trips for suffixed dirs

## Omer Lab (Marmoset Methods Adaptable to Mouse USVs)

- [[Oren 2024 Zenodo repository provides complete MATLAB implementation of spectrogram ridge vectorization for adaptation]] -- CC-BY 4.0 MATLAB code, small enough to port to Python
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- the 80D vectorization technique
- [[Oren 2024 16 acoustic features for marmoset calls parallel DeepSqueak 16 Excel export metrics for rodent USVs]] -- 16-feature convergence across species/tools
- [[GmSLM is a London-Omer collaboration applying self-supervised speech models to marmoset vocalizations]] -- SSL approach from London-Omer collaboration

## Broader Bioacoustics Tools

- [[scikit-maad implements double-threshold hysteresis binarization for ecological acoustics]] -- Ulloa et al 2021 open-source Python library for soundscape analysis; provides reference hysteresis implementation on spectrogram masks (2D) vs our CNN probability stream (1D)

## Pipeline Architecture Principles

- [[separating deterministic vectorization from stochastic clustering into distinct modules lowers iteration cost when two stages have different costs or randomness properties]] -- module boundaries should follow iteration cost gradients; vectorization caches across clustering parameter sweeps and lets one clustering implementation run over multiple vectorizers (Oren, AMVOC)

## Open Questions

- Whether BootSnap source code is obtainable for integration testing

## Related Areas

- [[classification]] -- parent map; CNN pipeline that produces the detections these tools process
- [[classification-methodology]] -- clustering and comparison methods applied after tool-based feature extraction
- [[detection]] -- upstream detection pipeline whose candidates feed classification tools
- [[signal-processing]] -- STFT parameters shared across tools

---

Topics:
- [[classification]]
- [[index]]
