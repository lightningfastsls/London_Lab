---
description: Recording-level splits, negative sampling, augmentation, labeling policy, classification baselines, and tool interoperability for USV model training
type: moc
topics: "[[experimental-methods]]"
---

# training-methodology

How we prepare training data and evaluate model performance. Recording-level splits prevent data leakage. Multi-source negative sampling covers the full non-USV distribution. Active learning scales from 2K to 30K labels across five milestones.

## Data Integrity & Splits
- [[recording-level splits prevent data leakage in USV classification]] -- split by recording file stem, not individual candidates
- [[recording-level splits reduce effective training set size but prevent data leakage]] -- the tradeoff: honest metrics vs smaller training set
- [[normalization statistics must be computed on training set only to prevent data leakage]] -- same leakage principle applied to preprocessing statistics

## Training Sample Quality
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- random chunks, inter-USV gaps, low-energy regions
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- general pattern for pre-filtered pipelines
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- handling class imbalance in training
- [[class weight boosting biases toward recall at the cost of precision]] -- the tradeoff from extreme pos_weight
- [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] -- labeling policy for training data composition
- [[good negative training samples must be unambiguously not USV to prevent label noise]] -- quality criterion for negative training samples
- [[implicit two-tier labeling emerges from CNN probability scores versus human binary overrides]] -- emergent labeling structure from machine + human
- [[whether very short USV signals near the 8-10 ms boundary should be included or excluded from training]] -- unresolved edge case for minimum duration

## Augmentation & Model Scaling
- [[constrained jittering generates diverse positive training examples by shifting detection boundaries within overlap constraints]] -- N=5 jittered copies per positive, 50% min overlap
- [[spectrogram SpecAugment-style augmentation with frequency and time masking improves transformer generalization]] -- 4 augmentations at p=0.5 for transformer training
- [[length-bucketed batching minimizes padding waste when sequences vary in duration]] -- 6-8 buckets for variable-length bout spectrograms
- [[model size growth versus available labeled data at each training milestone]] -- small/medium/large CNN scaling tied to label count
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] -- 5-milestone scaling from 2K to 30K labels
- [[recall versus precision tradeoff in two-stage USV detection]] -- defines evaluation methodology: what metrics matter and why

## Classification Strategy & Baselines
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- supervised CNN + unsupervised UMAP/HDBSCAN hedges taxonomy uncertainty
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- largest publicly available labeled USV dataset for transfer learning
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- best supervised baseline for wild mice (F1 67%)

## Tool Interoperability
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- standard format for passing detections between tools
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- simplifies export requirements
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- re-associating classified results with original detections

## Evaluation Standards
- [[a comprehensive practical bioacoustics detection guide was published in Biological Reviews 2025]] -- best practices for training data, metrics, generalization

## Open Questions
- How split strategy changes as dataset scales from 2K to 30K samples
- Whether augmentation strategies need revision for larger datasets
- [[split ratio inconsistency between DECISIONS.md 80-10-10 and ROADMAP Phase 9 70-15-15 needs resolution]]

## Related Areas
- [[experimental-methods]] -- parent hub for all experimental methodology
- [[classification]] -- model training depends on dataset preparation
- [[detection]] -- detection output feeds training data generation

---

Topics:
- [[experimental-methods]]
