---
description: Clustering approaches, repertoire comparison methods, few-shot learning, and cross-population generalization for USV syllable classification
type: moc
parent_map: classification
---

# classification-methodology

Methods for determining what USV types exist, comparing repertoires across populations, and generalizing classifiers to new contexts. The central tension: traditional discrete taxonomies (Holy & Guo 2005) versus the empirical continuum finding (Goffinet 2021). Dual supervised + unsupervised classification hedges this uncertainty.

## Clustering & Taxonomy Approaches

- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- hedges taxonomy uncertainty with supervised + UMAP/HDBSCAN
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- quality control via noise class (BootSnap practice)
- [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]] -- at least 512-point FFT, 1024 preferred at 300 kHz
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- alternative spectral representation
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- SIS validates whether categories capture meaningful structure
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] -- GMM model selection only supports k<=2 for mice
- [[MUPET gammatone filterbank with k-means discovers 100 to 140 data-driven syllable types as a handcrafted feature baseline]] -- gammatone features with k-means; tension with GMM finding k<=2
- [[ResNets outperform Vision Transformers for USV classification on neonatal mouse data]] -- 2024: adapted ResNets 86.79% accuracy; ViT did NOT outperform CNNs
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] -- field standard: embed then UMAP then HDBSCAN; auto cluster count
- [[CASE benchmark systematically compared 48 unsupervised clustering methods for animal vocalizations]] -- Schneider 2022 open benchmark; 48 algorithms tested
- [[unsupervised clustering as post-detection filtering eliminates 88 percent false positives while retaining 95 percent true positives]] -- clustering as a label-free precision filter stage
- [[SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines]] -- upstream segmentation for better downstream clustering

## Methodological Tensions

- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- categorical analysis loses within-category variation; mitigated by dual supervised+unsupervised approach

## Cross-Population Generalization

- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- key finding from BootSnap
- [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] -- wild mice adjust repertoire with social context
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- enables population-specific adaptation of a shared base classifier with minimal wild mouse labels; see [[model-adaptation]]

## Repertoire Comparison Methods

- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- primary multivariate test from ecology
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- scalar diversity metric; prediction: wild > lab
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- pairwise distributional distance [0,1]
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- syllable syntax comparison

## Few-Shot Learning for USV Classification

- [[prototypical networks are the dominant paradigm for few-shot bioacoustic event detection]] -- compute class centroids from few examples, classify by nearest distance; dominates DCASE 2021-2024
- [[DCASE few-shot bioacoustic detection improved from F1 40 percent to 70 percent across 2021-2024 challenge editions]] -- 5-shot setup: detect all instances from 5 annotated examples
- [[transductive inference uses unlabeled test data to iteratively refine class prototypes improving few-shot detection by 27 percent]] -- adapts to test distribution; 27% improvement on DCASE 2022
- [[no few-shot learning method has been applied to USV syllable-type classification]] -- VocalMat needs thousands of examples; DeepSqueak is unsupervised; the few-shot regime untested for USVs
- [[PCEN normalization is more robust than log-mel spectrograms for few-shot bioacoustic scenarios]] -- adapts to local noise levels; key for varying recording conditions between support and query
- [[negative prototype construction is critical for few-shot detection without explicit negative annotations]] -- only positives annotated in 5-shot setup; negative class must be constructed from background
- [[foundation model embeddings enable few-shot classification via simple linear probes without end-to-end training]] -- Perch 2.0/BEATs/AVES: embed, then k-NN or linear probe from ~10 examples
- [[prototypical probing with frozen MAE features enables bioacoustic classification with as few as 10 labeled examples]] -- 37% MAP improvement over linear probing on frozen MAE features
- [[the 300 kHz USV sample rate creates a domain shift challenge for applying audio foundation models]] -- 10-19x gap with pretrained models; frequency shifting or spectrogram-as-image workarounds
- [[frequency shifting USVs into the audible range could enable classification with standard audio foundation models]] -- pitch-shift to audible range preserving spectral structure; untested
- [[increasing pretext task difficulty improves embedding quality for downstream few-shot classification]] -- harder tasks produce better-separated embeddings; our binary USV/noise task may be too easy

## Literature Context

- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- traditional supervised approach baseline
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- motivates sequence analysis
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- supports sequence modeling
- [[Ivanenko et al 2020 showed DNNs achieve 77-84 percent accuracy classifying emitter sex from spectrograms]] -- identity information in spectrograms
- [[Goffinet 2021 found 64 to 95 percent of traditional USV feature information captured in VAE latent space]] -- quantitative baseline for information retention in learned representations

## Open Questions

- [[whether chi-squared on pooled syllable counts provides sufficient power as a simpler alternative to PERMANOVA for repertoire comparison]]
- Scaling behavior of small CNN as labeled dataset grows from 2K to 30K

## Related Areas

- [[classification]] -- parent map; CNN pipeline and training data that feed these methods
- [[classification-tools]] -- sibling map; tools that implement these methods
- [[representation-learning]] -- VQ-VAE and latent space approaches that learn representations from USV data
- [[unsupervised-usv-discovery]] -- broader unsupervised discovery literature
- [[experimental-methods]] -- dataset splits, augmentation, evaluation metrics

---

Topics:
- [[classification]]
- [[index]]
