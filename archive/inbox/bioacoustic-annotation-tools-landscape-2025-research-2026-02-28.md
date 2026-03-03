---
status: processed
date_accessed: "2026-02-28"
source_type: "web-research"
description: "Survey of open-source bioacoustic annotation tools covering detection through labeling, review, and export"
---

# Bioacoustic Annotation Tools Landscape (2024-2025)

> Source: Multi-query web research, Feb 2026
> Query: "open-source bioacoustic annotation tools: segmentation, labeling, review, export pipelines"
> Capture date: 2026-02-28

## Research Context

Survey of the open-source bioacoustic annotation tool landscape as of early 2026, covering the full pipeline from detection/segmentation through labeling, review, and export. Focuses on tools relevant to rodent USV research but includes the broader ecosystem for context.

---

## 1. Full-Pipeline Annotation Platforms

### Whombat
- **Type**: Web-based annotation platform (Python/FastAPI backend, React frontend)
- **Paper**: Martinez Balvanera et al. (2025), Methods in Ecology and Evolution
- **GitHub**: github.com/mbsantiago/whombat
- **Key features**:
  - Browser-based UI — no coding needed for annotators
  - Project management: organize recordings, create annotation projects, assign annotators
  - ML-assisted labeling: import model predictions, human reviewers correct/confirm
  - Collaborative review workflows: multiple annotators, hosted on shared server or local
  - Spectrogram visualization with adjustable parameters
  - Evaluation tools: compare model predictions against ground truth
  - Export: custom JSON (COCO-inspired) and CSV formats
- **Limitations**: Relatively new (published Dec 2024), smaller community than Raven. Custom JSON export format means interop requires conversion. No built-in detection — annotations only.
- **USV relevance**: Frequency range not limited — should handle ultrasonic if spectrograms are configured correctly. The ML-assisted iterative workflow (annotate → train → predict → review) maps well to active learning pipelines.

### Raven Pro / Raven Lite (Cornell Lab of Ornithology)
- **Type**: Desktop application (Java-based)
- **Website**: ravensoundsoftware.com
- **Key features**:
  - Gold-standard spectrogram viewer with extensive parameter control
  - Selection tables: manual bounding-box annotation on spectrograms
  - Measurement tools: extract acoustic features from selections
  - Batch processing capabilities
  - Export: Raven selection table (.txt, tab-separated) — the de facto interchange format
  - Raven Lite is free; Raven Pro is paid but widely used in academia
- **Limitations**: Not open-source (Lite is free but closed). No built-in ML/deep learning. No collaborative workflows. Java dependency. Manual annotation only — no ML assist.
- **USV relevance**: Handles high sample rates. Raven selection table format is the standard we already use for DeepSqueak bridge. But manual-only annotation is slow for large USV datasets.

### AviaNZ
- **Type**: Desktop application (Python/Qt)
- **Paper**: Marsland et al. (2019), Methods in Ecology and Evolution
- **GitHub**: github.com/smarsland/AviaNZ
- **Key features**:
  - Spectrogram annotation with species labels
  - Built-in detection filters: energy-based, FIR, median clipping
  - Filter training: train species-specific recognizers, clustering-based subfilters
  - Two-pass review: automated detection → human review workflow
  - Batch processing over folder hierarchies
  - Export: Python list format, spreadsheet with presence/absence tables
  - Filter sharing via JSON — community-contributed recognizers
  - Recently added bat detection support
- **Limitations**: Primarily designed for birds (but extensible). Export format is custom (not Raven-compatible natively). Small dev team. Species-centric rather than general acoustic event annotation.
- **USV relevance**: The two-pass detect→review workflow is architecturally similar to our two-stage pipeline. Filter training approach could inform how we build species-specific USV models. But it's bird-centric in practice.

### Sonic Visualiser
- **Type**: Desktop application (C++/Qt)
- **Website**: sonicvisualiser.org
- **Key features**:
  - Advanced spectrogram visualization with plugin architecture (Vamp plugins)
  - Annotation layers: point, time-value, region, note annotations
  - Plugin ecosystem for feature extraction (e.g., onset detection, pitch tracking)
  - Import/export: Sonic Visualiser layer files, CSV, Audacity labels
  - MIDI output
- **Limitations**: No ML integration. No collaborative features. Complex UI. More suited to music/speech research than bioacoustics. Plugin development requires C++.
- **USV relevance**: Excellent visualization but limited utility for large-scale USV annotation. Plugin architecture could theoretically support custom USV features.

---

## 2. Detection + Segmentation Tools (Feed into Annotation)

### DAS (Deep Audio Segmenter)
- **Type**: Python library + GUI (TensorFlow/Keras)
- **Paper**: Steinfath et al. (2021), eLife
- **GitHub**: github.com/janclemenslab/das
- **Key features**:
  - Temporal convolutional network (TCN) for frame-level annotation
  - Built-in GUI for annotation, training, and prediction review
  - Species-agnostic: tested on insects, birds, mammals
  - Raw audio input (not spectrogram patches)
  - 98% precision / 99% recall on mouse USVs
  - Trainable with small amounts of manual annotation
  - Can combine with unsupervised methods for novel call type discovery
- **Limitations**: Requires raw audio input — cannot classify pre-extracted segments. TensorFlow dependency. Frame-level output, not bounding-box format. Not designed for collaborative review.
- **USV relevance**: Best reported detection metrics for mouse USVs. The frame-level annotation approach is fundamentally different from our bounding-box detection. Already captured in vault notes.

### SqueakOut (2024)
- **Type**: Python library (PyTorch)
- **Paper**: 2024, published in PMC
- **Key features**:
  - Convolutional autoencoder (MobileNetV2 backbone) for USV segmentation
  - Pixel-level spectrogram segmentation (not bounding boxes)
  - Dice score 90.22 — best segmentation performance
  - Lightweight: 4.6M parameters
  - Includes 12,954 annotated spectrogram dataset
  - Hybrid loss function + data augmentation for robustness
- **Limitations**: Segmentation only — no detection from raw audio, no classification. Relatively new, small community.
- **USV relevance**: Directly relevant. Pixel-level segmentation could complement our bounding-box approach for precise frequency contour extraction. Could potentially replace the CNN precision filter in our two-stage pipeline for more precise masks.

### VocalMat (2021)
- **Type**: MATLAB toolbox
- **Paper**: Fonseca et al. (2021), eLife
- **Key features**:
  - Image-processing + differential geometry for USV detection
  - No user-defined parameters needed (fully automatic)
  - ML-based classification into syllable types
  - 98%+ detection rate on mouse USVs
  - Handles both pup and adult recordings
- **Limitations**: MATLAB dependency. Suboptimal segmentation in noisy conditions (Dice 63.82). Proprietary ecosystem.
- **USV relevance**: Strongest at parameter-free detection but weak at precise segmentation. The no-parameter-tuning philosophy is interesting contrast to our configurable DetectionConfig approach.

### USVSEG (2020)
- **Type**: MATLAB toolbox
- **Paper**: Tachibana et al. (2020), PLOS ONE
- **Key features**:
  - Multitaper spectrogram + signal processing for robust USV segmentation
  - Best precision among non-DL tools (7.58% false discovery rate)
  - Spectral peak tracking within syllables
  - Handles continuous recordings with background noise
- **Limitations**: MATLAB dependency. Requires manual parameter tuning. Not DL-based — may not generalize as well.
- **USV relevance**: Already compared with in vault notes (MUPET context). The multitaper spectrogram approach is a signal processing alternative to our single-window STFT.

### DeepSqueak
- **Type**: MATLAB toolbox with GUI
- **Already well-documented in vault** — Faster R-CNN detection, MATLAB 2020a + 7 toolboxes, VAE-based clustering, Raven import/export, GUI-centric.

---

## 3. Format Interoperability Layer

### Crowsetta
- **Type**: Python library
- **Paper**: Nicholson (2023)
- **GitHub**: github.com/vocalpy/crowsetta
- **Key features**:
  - Standardized Python API for reading/writing annotation formats
  - Built-in support: Audacity labels, Praat TextGrid, Raven selection tables, generic CSV/JSON
  - Extensible: custom format readers/writers
  - Part of the VocalPy ecosystem (with vak)
  - Converts between formats programmatically
- **Limitations**: Library only — no GUI, no detection, no annotation interface. Requires Python scripting.
- **USV relevance**: Critical interoperability tool. Could replace or complement our custom Raven export adapter. The format abstraction means pipeline outputs could target multiple tools simultaneously.

### Raven Selection Table Format
- **Already in vault**: Standard .txt tab-separated format (Selection, View, Channel, Begin Time, End Time, Low Freq, High Freq). De facto interchange standard.

---

## 4. ML Training + Inference Frameworks

### OpenSoundscape (OPSO)
- **Type**: Python library
- **Paper**: Lapp et al. (2023), Methods in Ecology and Evolution
- **GitHub**: github.com/kitzeslab/opensoundscape
- **Key features**:
  - Full ML pipeline: data prep, training, inference, evaluation
  - BoxedAnnotations class for viewing/manipulating annotations
  - CNN-based classification with transfer learning
  - Active learning workflows
  - Handles Raven selection table format natively
  - Batch prediction on large audio collections
  - Well-documented with tutorials
- **Limitations**: Focused on birds/general bioacoustics — no USV-specific features. Classification-oriented rather than precise segmentation. No annotation GUI.
- **USV relevance**: The active learning workflow and Raven format support are directly applicable. Could serve as a training framework if we move beyond our custom CNN.

### Koogu
- **Type**: Python library (TensorFlow)
- **GitHub**: github.com/shyamblast/Koogu
- **Key features**:
  - Full ML pipeline: data prep → training → evaluation → inference
  - Reads annotations from Raven, Audacity, SonicVisualiser formats
  - Custom annotation format readers via extensible base classes
  - Spectrogram-based features (removed librosa dependency — internal audio loading)
  - Batch inference with detection output writing (Raven format)
  - Waveform normalization built into models
- **Limitations**: Smaller community than OpenSoundscape. TensorFlow dependency. Documentation less mature.
- **USV relevance**: The multi-format annotation reading + Raven-format output writing is useful for interoperability. Internal audio loading (no librosa) aligns with our approach of avoiding librosa defaults.

### vak
- **Type**: Python library (PyTorch)
- **Paper**: Cohen et al. (2024), SciPy Proceedings
- **GitHub**: github.com/vocalpy/vak
- **Key features**:
  - Neural network framework for benchmarking vocalization models
  - TweetyNet: spectrogram segmentation model (original use case)
  - Parametric UMAP for vocalization embedding/clustering
  - Part of VocalPy ecosystem (with crowsetta)
  - CLI + Python API
  - Designed for reproducible model comparison
- **Limitations**: Focused on songbird research. Not a detection tool per se — more of a model benchmarking framework. Steeper learning curve.
- **USV relevance**: TweetyNet's spectrogram segmentation approach is architecturally interesting — frame-level classification on spectrograms rather than bounding boxes. Could inform how we think about segmentation vs. detection.

---

## 5. Landscape Summary: Pipeline Stages × Tool Coverage

| Stage | GUI Tools | Python Libraries | MATLAB Tools |
|-------|-----------|-----------------|--------------|
| **Raw Audio → Detection** | AviaNZ, Raven (manual) | DAS, OpenSoundscape | DeepSqueak, VocalMat, USVSEG |
| **Segmentation (pixel-level)** | — | SqueakOut | VocalMat (weak) |
| **Labeling/Annotation** | Whombat, Raven, AviaNZ, Sonic Visualiser | — | DeepSqueak |
| **ML-Assisted Review** | Whombat, AviaNZ, DAS | — | DeepSqueak |
| **Collaborative Review** | Whombat | — | — |
| **Classification/Clustering** | DeepSqueak (VAE) | vak (TweetyNet), OpenSoundscape | DeepSqueak, VocalMat |
| **Format Interop** | Raven (standard) | crowsetta | — |
| **Export to Downstream** | Raven tables, Audacity labels | crowsetta, Koogu | DeepSqueak (Excel, Raven) |

## 6. Key Observations

1. **No single tool covers the full pipeline well.** The ecosystem is fragmented: detection, annotation, review, and export are handled by different tools, often with format conversion friction between them.

2. **Whombat is the most promising new entrant** (Dec 2024) for collaborative annotation + ML-assisted review — the two capabilities most lacking in the USV-specific tools.

3. **Raven selection table remains the lingua franca.** Every serious tool either reads or writes it. Our existing Raven export adapter positions us well.

4. **Mouse USV tools are detection/segmentation focused** — VocalMat, USVSEG, SqueakOut, DAS all focus on finding calls, not on the human review/labeling workflow that follows.

5. **Crowsetta is the missing interop layer** — it could replace custom format adapters with a standardized Python API covering Raven, Audacity, Praat, and custom formats.

6. **The Python vs. MATLAB divide is shrinking** but still real. DeepSqueak and VocalMat remain MATLAB-only. The Python ecosystem (DAS, SqueakOut, OpenSoundscape, Koogu, vak, Whombat, crowsetta) is now more comprehensive.

7. **Active learning / human-in-the-loop workflows** are the frontier: Whombat, OpenSoundscape, and DAS all support iterative annotate→train→predict→review cycles. This is where the field is heading.
