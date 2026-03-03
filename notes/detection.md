---
description: USV detection pipeline -- energy detection, candidate generation, segment continuity, bout extraction
type: moc
---

# detection

How we find USVs in raw audio. The pipeline uses a two-stage architecture: a permissive energy detector generates candidates, then a CNN classifier filters for precision. Segment continuity and bout extraction add temporal structure.

## Core Ideas
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the architectural pattern: energy for recall, CNN for precision
- [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]] -- padded 30-110 kHz band defining the frequency domain for detection
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- permissive first stage by design
- [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] -- max energy per frame avoids signal dilution
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- rejects candidates spanning more than 20 kHz
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- 5 ms gap bridging with frequency/energy tolerance
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- grouping USVs into behavioral episodes
- [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] -- the specific grouping parameter
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- the general pattern behind our pipeline
- [[recall versus precision tradeoff in two-stage USV detection]] -- the designed tradeoff between detection stages
- [[batch detection with skip-existing enables incremental processing of large WAV collections]] -- headless processing of ~6,500 WAV files with error recovery
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- F1 91.7% baseline on ~840 labels
- [[noise-interrupted long USVs get split into two detections by the CNN sliding window]] -- CNN-level splitting of long calls with noisy gaps, distinct from energy detector fragmentation
- [[harmonics of a USV are treated as one call not multiple detections]] -- labeling convention: fundamental + harmonic = one call
- [[overlapping calls from multiple mice are labeled positive because USV presence is the classification target not individual identity]] -- detection target is presence, not source attribution
- [[implicit two-tier labeling emerges from CNN probability scores versus human binary overrides]] -- CNN scores + human overrides create two confidence tiers
- [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]] -- faint/short calls are the hardest to detect
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- noise structural mimicry triggers false positives

## Label Persistence
- [[JSON label files provide human-readable version-controllable persistence for detection labels and metadata]] -- one JSON per WAV stores detections, user labels, probability curves; git-friendly and inspectable

## Detection Landscape & Architecture Taxonomy
- [[six USV detection architectural approaches span object detection to speech model transfer with distinct tradeoff profiles]] -- taxonomy: object detection, segmentation, temporal, classical, speech transfer, hybrid
- [[entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative]] -- entropy measures spectral complexity; outperforms our energy detector precision
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
- [[no published single-channel USV source separation method exists as of 2026]] -- BioCPPNet handles macaques/dolphins/bats but 25-120 kHz USVs untested; labs discard overlapping recordings
- [[BioCPPNet U-Net architecture with permutation-invariant training enables single-channel bioacoustic source separation]] -- first neural bioacoustic source separation; STFT encoder + U-Net masks for 2-3 vocalizers
- [[spectrogram segmentation tools like SqueakOut and VocalMat are binary detectors that cannot separate overlapping USVs]] -- pixel is USV/not-USV with no USV-1 vs USV-2 distinction
- [[synthetic mixture training is the standard approach for training bioacoustic source separation networks]] -- mix isolated single-source recordings; viable for USVs from single-animal data
- [[frequency separation provides a partial solution when overlapping USVs occupy different spectral bands]] -- spectral peak splitting when calls occupy different frequency ranges; no NN needed
- [[hardware approaches solve USV attribution but not signal separation for overlapping calls]] -- mic arrays/wearables identify who vocalized but don't decompose the mixture waveform
- [[HyVL hybrid beamforming achieves 3 to 5 mm USV localization precision with 91 percent source assignment]] -- 64-element acoustic camera + 4 ultrasonic mics; 3x better than prior systems
- [[wearable miniature microphones achieve 90 percent USV attribution from amplitude alone]] -- 1.17g headgear-mounted mic; 10-20 dB proximity advantage; 97% with video
- [[Conv-TasNet time-domain separation architecture could handle 300 kHz USV recordings directly but requires ultrasonic training data]] -- time-domain separation natively handles any sample rate but needs USV training data

## Annotation Tools & Ecosystem
- [[no single bioacoustic tool covers the full detection-annotation-review-export pipeline]] -- ecosystem fragmented across detection, annotation, review, export tools with format friction
- [[Whombat is the first web-based platform for collaborative bioacoustic annotation with ML-assisted review]] -- Python/FastAPI + React; project management, multi-annotator, ML-assisted labeling
- [[Crowsetta standardizes annotation format interoperability across bioacoustic tools via a unified Python API]] -- VocalPy ecosystem; reads/writes Raven, Audacity, Praat, custom formats
- [[mouse USV annotation tools focus on detection and segmentation rather than human review and labeling workflows]] -- VocalMat/USVSEG/SqueakOut/DAS find calls but lack review interfaces
- [[active learning annotation workflows are the frontier in bioacoustic tools]] -- Whombat/OpenSoundscape/DAS support annotate-train-predict-review cycles
- [[the Python vs MATLAB divide in USV tools is shrinking but remains a practical barrier]] -- DeepSqueak/VocalMat MATLAB-only; DAS/SqueakOut/Whombat/Crowsetta form comprehensive Python ecosystem
- [[OpenSoundscape provides a full ML training pipeline with native Raven format support for bioacoustic classification]] -- BoxedAnnotations, CNN transfer learning, active learning, batch prediction
- [[A-MUD classical signal processing detector outperforms USVSEG and MUPET in true positive rate for USV detection]] -- 90.6%/80.0% precision/recall; best classical method but requires proprietary STx software
- [[a comprehensive practical bioacoustics detection guide was published in Biological Reviews 2025]] -- best practices for training data, evaluation metrics, cross-dataset generalization

## Open Questions
- [[optimal bout gap threshold may vary across behavioral contexts and recording conditions]]
- [[whether very short USV signals near the 8-10 ms boundary should be included or excluded from training]]
- Whether segment continuity parameters need per-recording tuning

## Related Areas
- [[classification]] -- the CNN precision filter that follows energy detection
- [[signal-processing]] -- STFT parameters that feed the energy detector
- [[experimental-methods]] -- evaluation methodology for detection performance
- [[representation-learning]] -- detection quality directly shapes what the VQ-VAE codebook can learn, since the transformer trains on detected bouts
- [[generative-modeling]] -- bounded gain stability principle from diffusion analysis applies to two-stage detection: error amplification between stages is the general risk pattern

---

Topics:
- [[index]]
