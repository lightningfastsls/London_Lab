---
description: CNN-based USV classification, the operational pipeline for detecting and labeling USVs, training strategies, and performance baselines
type: moc
---

# classification

The operational classification system. A small CNN (~101K params) classifies energy-detector candidates as USV or noise. The researcher uses this pipeline to process recordings, with every detection going through human validation. The labeled data this process produces feeds the future [[representation-learning]] research.

## CNN Architecture & Training
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- small CNN (~101K params) with variable-size input
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- extreme recall bias (pos_weight ~35.4)
- [[class weight boosting biases toward recall at the cost of precision]] -- the tradeoff from extreme pos_weight
- [[model size should scale with labeled dataset size to balance underfitting and overfitting]] -- small <5K, medium 5-15K, large 15K+ labels
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- F1 91.7% baseline

## Training Data & Labeling
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- random chunks, inter-USV gaps, low-energy regions
- [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] -- the selection bias that motivated multi-source negatives
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- general pattern for pre-filtered pipelines
- [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] -- labeling policy: noise-embedded USVs are valid positives
- [[good negative training samples must be unambiguously not USV to prevent label noise]] -- quality criterion for negative samples
- [[overlapping calls from multiple mice are labeled positive because USV presence is the classification target not individual identity]] -- classification target is presence, not source identity

## Error Patterns
- [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]] -- key failure mode and bias source
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- noise structural mimicry as FP source

## Python USV Classification Tools (Landscape)
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- best supervised classifier for pre-detected USVs, macro F1 67% on wild mice
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- MIT-licensed Python autoencoder, adaptable to external detections
- [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]] -- highest detection metrics but raw-audio-only
- [[WhisperSeg adapts OpenAI Whisper transformer for animal vocalization segmentation with positive cross-species transfer]] -- Whisper-based, outperforms DAS but raw-audio-only
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- largest freely available labeled USV dataset (10,871 USVs + 2,083 noise)
- [[whether BootSnap code is publicly available or must be requested from Abbasi Zala Penn at Vienna]] -- unresolved access question for the best-fit tool

## DeepSqueak Details
- [[DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection]] -- detection architecture evolution
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- unsupervised clustering yielding k=20
- [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]] -- STFT parameters favoring frequency resolution

## Classification Methodology
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- hedges taxonomy uncertainty with supervised + UMAP/HDBSCAN
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- quality control via noise class (BootSnap practice)
- [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]] -- at least 512-point FFT, 1024 preferred at 300 kHz
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- alternative spectral representation
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- SIS validates whether categories capture meaningful structure
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] -- GMM model selection only supports k<=2 for mice

## Cross-Population Generalization
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- key finding from BootSnap
- [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] -- wild mice adjust repertoire with social context

## Literature Context
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- the paradigm shift from discrete to continuous
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- competitive positioning
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- traditional supervised approach baseline
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- motivates sequence analysis
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- supports sequence modeling
- [[Ivanenko et al 2020 showed DNNs achieve 77-84 percent accuracy classifying emitter sex from spectrograms]] -- identity information in spectrograms
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] -- cross-species AE clustering with NMI 0.5-0.75
- [[AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks]] -- SSL backbone outperforming supervised methods
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] -- speech SSL transfers to animal domains
- [[Goffinet 2021 found 64 to 95 percent of traditional USV feature information captured in VAE latent space]] -- quantitative baseline for information retention in learned representations
- [[Garrobe Fonollosa 2024 showed VAE plus temporal convolutional network achieved AUC over 0.9 for sperm whale click classification]] -- VAE-based feature extraction for cetacean classification
- [[STSG spectrogram token skip-gram achieved only 0.559 AUC versus 0.810 for transfer learning on bioacoustic classification]] -- K-means token discretization dramatically underperforms
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] -- systematic gap analysis confirming novelty of VQ-VAE for bioacoustics

## DeepSqueak Bridge & Raven Interchange
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- tab-separated .txt format used by Raven Pro, DeepSqueak, Audacity
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- frequency bounds need only be approximate
- [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] -- cross-tool frequency convention (vs our 20-120 kHz)
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- re-associating DeepSqueak results with our detections

## Repertoire Comparison Methods
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- primary multivariate test from ecology
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- scalar diversity metric; prediction: wild > lab
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- pairwise distributional distance [0,1]
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- syllable syntax comparison

## Open Questions
- Scaling behavior of small CNN as labeled dataset grows from 2K to 30K
- [[whether chi-squared on pooled syllable counts provides sufficient power as a simpler alternative to PERMANOVA for repertoire comparison]]

## Related Areas
- [[detection]] -- upstream pipeline that generates candidates for classification
- [[representation-learning]] -- VQ-VAE and transformer research that builds on the labeled data this pipeline produces
- [[signal-processing]] -- STFT parameters that produce the spectrogram input
- [[experimental-methods]] -- dataset splits, class weighting, negative sampling

---

Topics:
- [[index]]
