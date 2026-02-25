---
description: Dataset preparation, recording-level splits, augmentation strategies, evaluation metrics, and label storage
type: moc
---

# experimental-methods

How we prepare data, evaluate models, and store results. Recording-level splits prevent data leakage. Multi-source negative sampling ensures the model sees the full distribution of non-USV audio. Labels stored as JSON for human readability and version control.

## Core Ideas
- [[recording-level splits prevent data leakage in USV classification]] -- split by recording file stem, not individual candidates
- [[recording-level splits reduce effective training set size but prevent data leakage]] -- the tradeoff: honest metrics vs smaller training set
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- random chunks, inter-USV gaps, low-energy regions
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- general pattern for pre-filtered pipelines
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- handling class imbalance in training
- [[class weight boosting biases toward recall at the cost of precision]] -- the tradeoff from extreme pos_weight, a key experimental design choice
- [[recall versus precision tradeoff in two-stage USV detection]] -- defines evaluation methodology: what metrics matter and why
- [[normalization statistics must be computed on training set only to prevent data leakage]] -- same leakage principle as recording splits, applied to preprocessing
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] -- 5-milestone scaling from 2K to 30K labels
- [[constrained jittering generates diverse positive training examples by shifting detection boundaries within overlap constraints]] -- N=5 jittered copies per positive, 50% min overlap
- [[spectrogram SpecAugment-style augmentation with frequency and time masking improves transformer generalization]] -- 4 augmentations at p=0.5 for transformer training
- [[length-bucketed batching minimizes padding waste when sequences vary in duration]] -- 6-8 buckets for variable-length bout spectrograms
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the core research question: wild vs lab vocalization differences
- [[model size growth versus available labeled data at each training milestone]] -- small/medium/large CNN scaling tied to label count
- [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] -- labeling policy for training data composition
- [[good negative training samples must be unambiguously not USV to prevent label noise]] -- quality criterion for negative training samples
- [[implicit two-tier labeling emerges from CNN probability scores versus human binary overrides]] -- emergent labeling structure from machine + human
- [[whether very short USV signals near the 8-10 ms boundary should be included or excluded from training]] -- unresolved edge case for minimum duration

## Classification Strategy & Training Data
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- supervised CNN + unsupervised UMAP/HDBSCAN hedges taxonomy uncertainty
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- BootSnap's key cross-population finding
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- largest publicly available labeled USV dataset for transfer learning
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- best supervised baseline for wild mice (F1 67%)
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- continuous alternative to PERMANOVA on category proportions

## Wild Mouse Vocal Behavior
- [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] -- context-dependent repertoire modulation in wild mice

## Research Hypotheses
- [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] -- the directional degradation hypothesis
- [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]] -- preliminary supporting evidence
- [[USVs are one component of a multimodal courtship behavior suite including mounting approach and movement]] -- multimodal courtship framing
- [[temporal alignment between USV detections and LMT behavioral events enables USV-behavior correlation analysis]] -- USV-behavior correlation method
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- immediate science strategy
- [[VQ-VAE investigation of language-like sequential structure in USVs is a separate deeper question from courtship degradation]] -- two-tier research strategy
- [[combined cross-modal evidence from USV repertoire and MiceCraft movement data builds a stronger case for courtship degradation]] -- cross-modal evidence strategy
- [[whether specific USV call types predict specific courtship outcomes like female receptivity to mounting]] -- functional specificity question

## Null Model Methods
- [[shuffled null model preserves code frequencies but destroys all sequential structure]] -- simplest baseline preserving only unigram frequencies
- [[Markov order-k null model generates surrogates preserving k-step transition dependencies]] -- systematic memory depth testing at k=1,2,3
- [[HMM surrogate null model tests whether USV sequences arise from hidden behavioral state switching]] -- tests Chabout et al behavioral state switching hypothesis via Baum-Welch EM
- [[analytically verifiable test cases validate information-theoretic metric implementations]] -- known-answer tests (uniform entropy, Poisson CV, planted idioms) as implementation sanity checks

## Probing & Interpretability Methods
- [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] -- Belinkov 2022 probing adapted for USV transformer with 5-fold CV
- [[probe selectivity measured as accuracy minus majority baseline distinguishes genuine encoding from trivial prediction]] -- corrects misleading accuracy on imbalanced targets
- [[layer-property heatmap is the key output showing where acoustic information lives across transformer depth]] -- (layers x properties) matrix as primary probing deliverable
- [[pooling strategy choice over the time dimension determines what information probing experiments can access from hidden states]] -- mean/max/first/last pooling as control variable in probe experiments

## Converging Hypothesis
- [[the converging research question asks whether transformer encodes behaviorally meaningful vocal categories differing between wild and lab populations]] -- where information theory, probing, and LMT integration converge

## Open Questions
- How split strategy changes as dataset scales from 2K to 30K samples
- Whether augmentation strategies need revision for larger datasets
- [[split ratio inconsistency between DECISIONS.md 80-10-10 and ROADMAP Phase 9 70-15-15 needs resolution]]
- [[whether population-level metadata is available for context-dependent VQ-VAE analysis]]

## DeepSqueak Bridge Methods
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- standard interchange format for passing detections between tools
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- simplifies export requirements
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- re-associating classified results with original detections

## Repertoire Statistical Methods
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- non-parametric test from community ecology (Anderson 2001)
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- diversity metric with wild > lab prediction
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- symmetric [0,1] distributional distance
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- sequential syntax comparison method

## Recording Infrastructure
- [[AviSoft Recorder captures synchronized USV recordings within the LMT behavioral tracking system]] -- recording software
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- behavioral tracking + USV synchronization
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- acoustic environment
- [[Pasteur USV cloud platform enables online testing of detection methods without local infrastructure]] -- external testing platform
- [[LMT USV Toolbox provides Python-based offline USV processing as a reference implementation]] -- reference implementation from the LMT team

## LMT Integration Methods
- [[event-triggered USV rate via PETH in plus-minus 2 second windows per event type serves as LMT integration sanity check]] -- Tier 1 peri-event time histograms as basic USV-behavior correlation sanity check
- [[mutual information between vocal sequence and next behavior quantifies vocal prediction of behavioral transitions]] -- Tier 3 I(vocal_sequence; next_behavior) as strongest test of communicative function

## Related Areas
- [[detection]] -- detection output feeds training data generation
- [[classification]] -- model training depends on dataset preparation
- [[signal-processing]] -- recording parameters constrain experimental design

---

Topics:
- [[index]]
