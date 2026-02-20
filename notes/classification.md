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

## Open Questions
- Scaling behavior of small CNN as labeled dataset grows from 2K to 30K

## Related Areas
- [[detection]] -- upstream pipeline that generates candidates for classification
- [[representation-learning]] -- VQ-VAE and transformer research that builds on the labeled data this pipeline produces
- [[signal-processing]] -- STFT parameters that produce the spectrogram input
- [[experimental-methods]] -- dataset splits, class weighting, negative sampling

---

Topics:
- [[index]]
