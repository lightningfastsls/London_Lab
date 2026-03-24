---
description: USV detection approaches beyond our pipeline -- architectural taxonomy, alternative tools, source separation methods, and annotation ecosystem
type: moc
parent_map: "[[detection]]"
topics: "[[detection]]"
---

# detection-landscape

The broader landscape of USV detection tools, architectures, source separation methods, and annotation ecosystems. Split from [[detection]] which retains our specific pipeline implementation (energy detector, CNN filter, segment continuity, bout extraction). This sub-map covers what else exists and what the field is doing.

## Synthesis

Six architectural approaches span the detection space, from object detection (DeepSqueak) through semantic segmentation (U-Net, SqueakOut) to speech model transfer (WhisperSeg). The highest reported metrics come from [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]] and [[U-Net semantic segmentation exceeded 95 percent precision recall for USV detection in systematic DL comparison]], though our two-stage pipeline achieves competitive results with simpler architecture. Source separation remains the hardest unsolved problem: since [[no published single-channel USV source separation method exists as of 2026]], multi-mouse recordings still require discarding overlapping calls or using hardware solutions. The annotation ecosystem is fragmented -- since [[no single bioacoustic tool covers the full detection-annotation-review-export pipeline]], researchers must stitch together detection, annotation, review, and export tools.

## Architectural Taxonomy

- [[six USV detection architectural approaches span object detection to speech model transfer with distinct tradeoff profiles]] -- taxonomy: object detection, segmentation, temporal, classical, speech transfer, hybrid
- [[entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative]] -- entropy measures spectral complexity; outperforms energy detector precision
- [[U-Net semantic segmentation exceeded 95 percent precision recall for USV detection in systematic DL comparison]] -- Ivanenko 2023: AE, U-Net, RNN all >90%; U-Net best generalization
- [[HybridMouse CNN plus BiLSTM first combined spatial and temporal features for USV detection outperforming DeepSqueak in low SNR]] -- spatial+temporal hybrid; low-SNR robustness

## Alternative Detection Tools

- [[DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection]] -- DeepSqueak's MATLAB-only detection architecture evolution
- [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]] -- highest reported detection metrics (Python, TensorFlow)
- [[WhisperSeg adapts OpenAI Whisper transformer for animal vocalization segmentation with positive cross-species transfer]] -- outperforms DAS with cross-species transfer
- [[SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines]] -- pixel-level USV masks, MobileNetV2 backbone (4.6M params, 18MB)
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- classification-stage noise class as second-pass detection filter
- [[unsupervised clustering as post-detection filtering eliminates 88 percent false positives while retaining 95 percent true positives]] -- unsupervised clustering as third-stage precision filter

## Source Separation & Overlap Handling

- [[no published single-channel USV source separation method exists as of 2026]] -- BioCPPNet handles macaques/dolphins/bats but 25-120 kHz USVs untested
- [[BioCPPNet U-Net architecture with permutation-invariant training enables single-channel bioacoustic source separation]] -- first neural bioacoustic source separation; STFT encoder + U-Net masks
- [[spectrogram segmentation tools like SqueakOut and VocalMat are binary detectors that cannot separate overlapping USVs]] -- pixel is USV/not-USV with no multi-source distinction
- [[synthetic mixture training is the standard approach for training bioacoustic source separation networks]] -- mix isolated single-source recordings; viable for USVs from single-animal data
- [[frequency separation provides a partial solution when overlapping USVs occupy different spectral bands]] -- spectral peak splitting; no NN needed
- [[hardware approaches solve USV attribution but not signal separation for overlapping calls]] -- mic arrays/wearables identify who vocalized but don't decompose the mixture
- [[HyVL hybrid beamforming achieves 3 to 5 mm USV localization precision with 91 percent source assignment]] -- 64-element acoustic camera + 4 ultrasonic mics
- [[wearable miniature microphones achieve 90 percent USV attribution from amplitude alone]] -- 1.17g headgear-mounted mic; 97% with video
- [[Conv-TasNet time-domain separation architecture could handle 300 kHz USV recordings directly but requires ultrasonic training data]] -- time-domain separation natively handles any sample rate

## Annotation Tools & Ecosystem

- [[no single bioacoustic tool covers the full detection-annotation-review-export pipeline]] -- ecosystem fragmented across detection, annotation, review, export tools
- [[Whombat is the first web-based platform for collaborative bioacoustic annotation with ML-assisted review]] -- Python/FastAPI + React; multi-annotator, ML-assisted labeling
- [[Crowsetta standardizes annotation format interoperability across bioacoustic tools via a unified Python API]] -- VocalPy ecosystem; reads/writes Raven, Audacity, Praat formats
- [[mouse USV annotation tools focus on detection and segmentation rather than human review and labeling workflows]] -- VocalMat/USVSEG/SqueakOut/DAS find calls but lack review interfaces
- [[active learning annotation workflows are the frontier in bioacoustic tools]] -- Whombat/OpenSoundscape/DAS support annotate-train-predict-review cycles
- [[the Python vs MATLAB divide in USV tools is shrinking but remains a practical barrier]] -- DeepSqueak/VocalMat MATLAB-only; comprehensive Python ecosystem emerging
- [[OpenSoundscape provides a full ML training pipeline with native Raven format support for bioacoustic classification]] -- BoxedAnnotations, CNN transfer learning, active learning
- [[A-MUD classical signal processing detector outperforms USVSEG and MUPET in true positive rate for USV detection]] -- 90.6%/80.0% precision/recall; requires proprietary STx software
- [[a comprehensive practical bioacoustics detection guide was published in Biological Reviews 2025]] -- best practices for training data, evaluation, cross-dataset generalization

## Related Areas

- [[detection]] -- parent: our specific two-stage detection pipeline
- [[classification]] -- downstream from detection; tools like DeepSqueak bundle detection + classification
- [[signal-processing]] -- STFT parameters that feed all detection approaches
- [[bioacoustic-ssl]] -- foundation models (WhisperSeg, Perch) overlap with detection via transfer learning

---

Topics:
- [[index]]
- [[detection]]
