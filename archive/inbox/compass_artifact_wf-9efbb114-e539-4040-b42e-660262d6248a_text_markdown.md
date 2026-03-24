# DeepSqueak and USV syllable classification: a practical guide

**DeepSqueak is a MATLAB-only GUI application (v3.2, BSD-3 license) that detects and classifies rodent ultrasonic vocalizations using YOLO v2 detection and k-means/VAE unsupervised clustering—but no standalone Python port exists.** For a Python-based pipeline that already detects USVs and needs syllable typing, the most practical path is building a custom CNN classifier on spectrogram patches, supplemented by AMVOC's unsupervised autoencoder or BootSnap's snapshot ensemble approach. A critical insight from recent VAE studies: mouse USVs form a **continuous manifold rather than discrete syllable categories**, which means the traditional Holy & Guo taxonomy may impose artificial boundaries.

---

## DeepSqueak is powerful but locked inside MATLAB

DeepSqueak was developed by Kevin Coffey and colleagues at the University of Washington (Coffey et al., 2019, *Neuropsychopharmacology*). It lives at `github.com/DrCoffey/DeepSqueak` (~417 stars, BSD-3-Clause license). The current release is **v3.1** (February 2025) with v3.2 features described in the README.

**System requirements are non-trivial.** DeepSqueak needs MATLAB 2020a or later plus seven toolboxes: Deep Learning, Computer Vision, Image Processing, Curve Fitting, Parallel Computing, Statistics and Machine Learning, and Signal Processing. A GPU with ≥2 GB VRAM is recommended but not required—without one, processing slows dramatically. The tool runs on Windows, macOS, and Linux.

The architecture evolved significantly across versions. Versions 1–2 used Faster-RCNN for detection; **v3.1 switched to YOLO v2**, improving speed and accuracy. Version 3.2 added high-precision neural networks and automatic horizontal noise band removal. Detection and classification are deliberately separated—DeepSqueak detects USV bounding boxes in spectrograms first, then classifies them using one of three approaches: unsupervised clustering (k-means, ARTwarp, or VAE), supervised CNN classification, or manual labeling with custom categories.

**There is no Python port.** Extensive searching confirms no "pysqueak" or Python reimplementation exists. DeepSqueak remains entirely MATLAB-dependent.

---

## Classification uses data-driven clustering, not a fixed taxonomy

DeepSqueak deliberately avoids hardcoding the Holy & Guo (2005) or Scattoni et al. (2008) syllable categories. The original paper states there is "no consensus yet on exactly how to categorize USVs" and takes an agnostic, data-driven approach.

**Unsupervised clustering** is DeepSqueak's primary classification method. The k-means implementation operates on three feature types—contour shape (1st derivative at 10 segments), frequency (contour reduced to 10 segments), and duration—all z-score normalized with user-adjustable weighting. The elbow method on within-cluster error automatically determines the optimal cluster count; in the original paper, this yielded **20 optimal syllable types** for mouse USVs. Version 3.1 added **Variational Autoencoder (VAE)-based contour-invariant clustering**, a significant upgrade for capturing continuous variation. t-SNE visualization is built in for inspecting cluster distributions.

**Supervised classification** uses a CNN operating on spectrogram images. Pre-trained networks ship with DeepSqueak, including one based on Wright et al.'s rat USV categories. Users can train custom classifiers by first producing clean clusters via unsupervised methods, then using those labeled examples to train a supervised network for faster future classification. To use the Holy & Guo taxonomy, you would manually label exemplars into the 10 Scattoni categories (flat, chevron, short, up-FM, down-FM, complex, step-up, step-down, two-syllable, composite), then train a supervised classifier on those labels.

---

## Input and output formats support 300 kHz recordings

DeepSqueak accepts **WAV, FLAC, and Ultravox (.UVD)** files. It was primarily tested at 250 kHz sample rate, but the documentation explicitly states that "spectrograms are created using FFT windows of constant duration, rather than constant sample numbers, so other sample rates are accepted." **A 300 kHz WAV file will work.** Only the first channel of multichannel files is processed.

Internal spectrogram parameters for mouse USVs use **3.2 ms FFT windows with 2.8 ms overlap** (specified in seconds, not samples). Frequency cutoffs default to roughly 30–120 kHz for mouse USVs but are user-configurable. For a 300 kHz sample rate, these parameters translate to 960-sample FFTs and 840-sample overlaps.

**Output is primarily MATLAB .mat files** containing per-call structures with fields for bounding box position `[Begin Time, Min Frequency, Duration, Frequency Range]`, detection confidence score, raw audio snippet, accept/reject status, classification label, and power. Excel export (`File → Export to Excel Log`) produces **.xlsx files** with 16 metrics per call: ID, label, begin/end time, call length, principal frequency, low/high frequency, bandwidth, frequency standard deviation, slope, sinuosity, mean power, tonality, and peak frequency. Additional exports include Raven selection tables and spectrogram images. **No native CSV or SQLite output**, though the .xlsx export is functionally equivalent.

---

## Python alternatives exist but none perfectly fits post-detection classification

The landscape of Python-based USV tools is fragmented. No single tool cleanly accepts pre-detected USV segments and classifies them into syllable types. Here are the most relevant options:

**BootSnap** (Abbasi et al., 2022, *PLOS Computational Biology*) is the **closest conceptual match**. It was explicitly designed to classify pre-detected USVs into 12 syllable types using Gammatone spectrograms fed into a CNN with snapshot ensemble learning. It outperformed both pre-trained and retrained DeepSqueak classification (macro F1 of **67% on wild mice vs. DeepSqueak's 41%**), and showed the best cross-generalization between wild-derived and laboratory mice. It is Python-based. The limitation: no confirmed public GitHub repository—the code may need to be requested from the authors (Abbasi, Zala, Penn at the University of Veterinary Medicine, Vienna).

**AMVOC** (Giannakopoulos et al., 2022, *Bioacoustics*) is the **best available open-source Python tool** (`github.com/tyiannak/amvoc`). It uses a convolutional autoencoder for unsupervised feature extraction and clustering of USV spectrograms. It is pure Python 3.8 with PyTorch and scikit-learn, MIT-licensed, and supports both offline batch processing and real-time analysis via a Dash web GUI. Its detection module outputs CSVs with onset/offset, and the clustering module processes detected USVs—making it potentially adaptable to accept externally detected segments if formatted correctly.

**DAS (Deep Audio Segmenter)** (`pip install das`) uses temporal convolutional networks and achieves **98% precision / 99% recall** on mouse USVs. It is well-maintained Python (TensorFlow/Keras) with CLI and Python API. However, it operates on raw audio streams with frame-level annotation and is not designed to classify pre-extracted segments. It would require restructuring your pipeline.

**WhisperSeg** (Gu et al., 2024, ICASSP) adapts OpenAI's Whisper transformer for animal vocalization segmentation. It outperforms DAS across multiple species and shows positive transfer learning across species. Available on HuggingFace (`nccratliri/whisperseg-large-ms`). Like DAS, it processes raw audio end-to-end.

| Tool | Language | Accepts pre-detected USVs? | Classification type | Maintenance |
|------|----------|---------------------------|-------------------|-------------|
| **BootSnap** | Python | **Yes (designed for it)** | Supervised CNN (12 classes) | Published 2022 |
| **AMVOC** | Python | Partially (adaptable) | Unsupervised autoencoder | Active (GitHub) |
| **DAS** | Python | No (raw audio) | Supervised TCN | Active |
| **WhisperSeg** | Python | No (raw audio) | Supervised transformer | Active |
| **VocalMat** | MATLAB | Partially | Supervised CNN (11 classes) | Inactive since ~2021 |
| **USVSEG** | MATLAB + Python port | N/A | Detection only | Published 2020 |

---

## The most practical path: build a custom spectrogram CNN classifier

Given that you already have a detection pipeline at F1 91.7%, the most efficient approach is to add a classification stage that operates on spectrogram patches extracted from your detected USV segments. Here is a concrete architecture:

**Spectrogram extraction.** For each detected USV, extract the audio segment with ~15 ms padding on each side. Compute an STFT with **512–1024 point FFT** (at 300 kHz), Hamming window, and 75% overlap, restricted to **25–125 kHz**. A critical finding from recent work: **fine frequency resolution matters far more than time resolution** for CNN classification—the network learns from the frequency contour "skeleton." Use at least 512 FFT points; 1024 is preferable at 300 kHz for ~293 Hz frequency resolution. Resize all spectrogram patches to a fixed size (128×128 or 224×224 for transfer learning). Consider **Gammatone spectrograms** as an alternative—BootSnap found these outperform standard STFTs for USV classification.

**Model architecture.** Fine-tune a pretrained MobileNetV2 or ResNet-18 backbone from ImageNet on your spectrogram patches. VocalMat achieved **~86% accuracy on 11 syllable categories** using fine-tuned AlexNet. BootSnap's snapshot ensemble learning on a CNN achieved **F1 67–74.5%** across wild and lab mice. Include a "noise/false positive" class (as BootSnap does) to catch residual detection errors.

**Training data.** VocalMat provides **12,954 labeled spectrograms** (10,871 USVs across 11 categories + 2,083 noise) freely available on GitHub. BootSnap's labeled data from wild-derived and lab mice is available through their PLOS Computational Biology supplementary materials. You will likely need to supplement these with your own labeled data from wild mouse recordings, since classifiers trained on lab mice **generalize poorly to wild mice** (BootSnap's key finding).

**Dual classification approach.** Run both supervised classification (for comparability with published literature using Scattoni categories) and unsupervised UMAP + HDBSCAN clustering on spectrogram embeddings (for data-driven discovery). This addresses the taxonomy problem from both directions.

---

## Mouse USVs may not form discrete syllable types at all

The most important recent finding for your research comes from Goffinet et al. (2021, *eLife*), who applied Variational Autoencoders to mouse USV spectrograms. Their "Autoencoded Vocal Analysis" (AVA) tool (`github.com/pearsonlab/autoencoded-vocal-analysis`) learned a 32-dimensional latent space and found that **mouse USV syllables form a continuous spectrum, not discrete clusters**. Smooth interpolations exist between any two syllable types. Gaussian mixture model clustering on the VAE latent space was only supported for k≤2 clusters in mice—a stark contrast to zebra finch syllables, which cluster cleanly.

This has profound implications for comparing wild and lab mouse repertoires. Traditional approaches compute proportions of each discrete syllable type per animal, then test for differences using chi-square or PERMANOVA. But if the underlying space is continuous, **distributional comparisons in latent space** (using Earth Mover's Distance or Jensen-Shannon divergence on VAE embeddings) may be more biologically meaningful than forcing USVs into categorical bins.

Hertz et al. (2020, *Communications Biology*) reinforced this by showing that different classification schemes (Holy & Guo, MUPET, DeepSqueak) produce **no one-to-one mapping** between labels. They developed the "Syntax Information Score" to rank classification schemes by how well syllable labels predict the next syllable in a sequence—essentially using temporal structure to validate whether categories capture meaningful biological variation.

---

## Batch processing and integration require workarounds

DeepSqueak's **Multi-Detect button** supports batch detection across multiple files, and batch operations exist for post-hoc denoising, threshold-based rejection, unsupervised clustering, supervised classification, and Excel export. However, the tool is **fundamentally GUI-centric**—headless/scriptable operation is not officially supported. Advanced MATLAB users can call underlying functions (like `SqueakDetect`) programmatically, but this is undocumented and fragile.

For importing pre-detected USVs, the **Raven selection table import** is the most flexible pathway. Format your detected USV timestamps and frequency ranges as a Raven .txt selection table (with Begin Time, End Time, Low Freq, High Freq columns), then use `File → Import Calls → Import from Raven`. The original audio files must be accessible—DeepSqueak needs them for spectrogram generation during classification. Pure "classification only" mode without audio access is not possible.

**Reading DeepSqueak outputs in Python** is straightforward: use `scipy.io.loadmat()` for MATLAB v5 format or `h5py` for v7.3 (HDF5) format. DeepSqueak v3.x may use either. All bounding box, score, type, and audio data are accessible as NumPy arrays after loading.

---

## Statistical framework for comparing wild and lab syllable repertoires

The standard statistical methods for comparing USV repertoire distributions between populations, in order of common use:

- **PERMANOVA** on Bray-Curtis distance matrices of per-animal syllable proportions (the most widely used method; available in Python via `scikit-bio`)
- **Chi-square or Fisher's exact tests** for comparing overall syllable type proportions between groups
- **KL divergence / Jensen-Shannon divergence** for quantifying distributional distance between populations (Bhattacherjee et al. used KL divergence at 95% CIs)
- **Markov chain transition analysis** comparing syllable-to-syllable transition probability matrices between groups, testing with chi-square or permutation tests
- **Shannon entropy** for quantifying repertoire diversity per individual or group

For the wild vs. lab comparison specifically, Zala et al. (2020, *Frontiers in Zoology*) showed wild-derived house mice modulate USVs based on social context with **9 syllable types during direct interaction vs. 6 during introduction**. The BootSnap study found that t-SNE distributions differ substantially between wild and lab mice for certain classes (inverted-U, complex), while simpler categories (up, down, flat, short) show large overlap. **Classifiers trained on one population generalize poorly to the other**, which means you should train or fine-tune on both wild and lab data.

---

## Conclusion

The USV syllable classification landscape in 2026 has no single perfect solution for a Python-first pipeline with pre-detected USVs. The most practical strategy combines three elements: **(1)** a custom PyTorch CNN classifier fine-tuned on VocalMat's 12,954 labeled spectrograms for supervised Scattoni-style categorization, **(2)** a VAE or convolutional autoencoder (following AMVOC or Goffinet's AVA approach) for unsupervised exploration of the continuous USV space, and **(3)** distributional statistics (PERMANOVA, JSD, transition matrices) for comparing wild and lab repertoires. The dual supervised/unsupervised approach is essential because forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations. Include a noise class in your classifier for quality control, and plan to label at least a subset of your own wild mouse USVs—no existing labeled dataset adequately represents wild mouse vocal variation.