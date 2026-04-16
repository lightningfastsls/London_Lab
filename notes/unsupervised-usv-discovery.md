---
description: Unsupervised clustering methods, USV literature context, and alternative approaches for discovering vocal repertoire structure without predefined categories
type: moc
topics: "[[index]]"
---

# unsupervised-usv-discovery

Methods and evidence for discovering USV type categories without human-defined labels. Covers the dominant UMAP+HDBSCAN pipeline, benchmark comparisons, the continuous-vs-discrete debate in USV categorization, and alternative representation approaches from VAE to transfer learning. These notes provide the scientific context motivating the VQ-VAE approach.

## Unsupervised Clustering Evidence
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] -- embed-reduce-cluster paradigm; auto cluster count, noise handling, O(n log n)
- [[CASE benchmark systematically compared 48 unsupervised clustering methods for animal vocalizations]] -- Schneider 2022: including community detection, affinity propagation, fuzzy clustering
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] -- quantitative evidence USVs resist discrete clustering (GMM k<=2)
- [[MUPET gammatone filterbank with k-means discovers 100 to 140 data-driven syllable types as a handcrafted feature baseline]] -- k-means forces 100+ types that GMM says don't exist
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- k-means finds k=20 but GMM finds k<=2 on learned representations
- [[DeepSqueak v3.1 added VAE-based contour-invariant clustering as upgrade over k-means for continuous USV variation]] -- DeepSqueak's own response to the continuous USV space problem
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- MIT-licensed Python autoencoder alternative
- [[AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression]] -- architecture spec: 3 conv + MaxPool, 8× compression
- [[AMVOC trains for only 2 epochs deliberately because the undercomplete bottleneck acts as implicit regularizer]] -- training philosophy: lossy reconstruction by design
- [[AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions]] -- post-processing: 1280→~320→PCA, cluster in PCA space not t-SNE
- [[AMVOC deep autoencoder features scored 37 percent higher than 4-feature handcrafted baselines in blinded human evaluation]] -- quantitative learned-vs-handcrafted evidence
- [[AMVOC semi-supervised retraining combines reconstruction KL divergence and pairwise constraint losses with uncertainty-based annotation priority]] -- early active learning for USV clustering
- [[AMVOC lacks batch normalization dropout validation monitoring and VAE variant — all high-value improvements for our wild-mouse pipeline]] -- gap analysis and design checklist
- [[AMVOC t-SNE plus user-specified k versus field-standard UMAP plus HDBSCAN for bioacoustic clustering]] -- tension: old method still works, is embedding quality the real discriminant?
- [[SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines]] -- upstream segmentation improves downstream clustering quality
- [[unsupervised clustering as post-detection filtering eliminates 88 percent false positives while retaining 95 percent true positives]] -- clustering as precision filter without labels
- [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] -- own-data replication: density-based clustering confirms k<=2 finding with different method on different dataset
- [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]] -- open question: our HDBSCAN used raw features; encoder embeddings might reveal sub-structure within the main cluster
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- SIS evaluates whether any discretization captures meaningful sequential structure
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- continuous comparison alternative to categorical methods
- [[burstiness by behavioral context bridges information theory and LMT behavioral analysis]] -- burstiness broken down by behavioral context labels

## Literature Context
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the key finding that motivated the VQ-VAE approach
- [[Goffinet 2021 found 64 to 95 percent of traditional USV feature information captured in VAE latent space]] -- quantitative baseline for information retention
- [[Tjandra et al 2020 applied transformer VQ-VAE for unsupervised unit discovery in human speech with K equals 128]] -- closest architectural analog from speech domain
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- original novelty claim (2026-02-19)
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] -- updated gap analysis confirming novelty
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- unsupervised predecessor using handcrafted features

## Omer Lab Vectorization (Ridge-Based)
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- 80D [AM+FM] fixed-length vector via ridge extraction + time resampling (Oren et al. 2024)
- [[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory]] -- core algorithmic step: argmax per spectrogram column
- [[time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information]] -- 2D interpolation to fixed time steps solves variable-length problem
- [[whether Omer-style ridge vectorization applied to mouse USVs produces meaningfully different clustering than AMVOC autoencoder embeddings]] -- open question: does AM component reveal substructure that FM-only misses?

## Adjacent Approaches (Not End-to-End VQ-VAE)
- [[Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations]] -- first discrete tokens in bioacoustics, post-hoc not end-to-end
- [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] -- 35% vs 49% UAR gap validates end-to-end design
- [[Gumbel-softmax VQ suffered severe codebook collapse in bioacoustic token experiments]] -- GVQ negative result validates standard VQ-VAE choice
- [[single codebook with V=50 was insufficient for complex vocalization structure in discrete token experiments]] -- may need RVQ or larger K
- [[VQ token sequences discriminate call types but lose individual identity information during discretization]] -- call types preserved but individual identity lost
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] -- continuous AE for repertoire discovery across species
- [[STSG spectrogram token skip-gram achieved only 0.559 AUC versus 0.810 for transfer learning on bioacoustic classification]] -- K-means tokens dramatically underperform
- [[Garrobe Fonollosa 2024 showed VAE plus temporal convolutional network achieved AUC over 0.9 for sperm whale click classification]] -- VAE for cetacean feature extraction

## Open Questions
- Whether the continuous-vs-discrete debate resolves differently at different USV timescales
- How to evaluate the "quality" of discovered categories when ground truth doesn't exist

## Related Areas
- [[representation-learning]] -- the VQ-VAE pipeline that operationalizes the discrete discovery approach
- [[classification]] -- supervised categorization that contrasts with unsupervised discovery
- [[bioacoustic-ssl]] -- foundation model embeddings as input features for clustering
- [[experimental-methods]] -- evaluation methods for unsupervised category quality
- [[model-adaptation]] -- LoRA adaptation of frozen SSL encoders for USV-specific discrete representation learning

---

Topics:
- [[index]]
