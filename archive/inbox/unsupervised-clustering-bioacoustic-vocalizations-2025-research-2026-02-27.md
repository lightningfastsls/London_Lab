---
description: "Survey of 2024-2025 unsupervised clustering methods for bioacoustic vocalizations -- foundation models, self-supervised embeddings, UMAP+HDBSCAN pipelines, and VQ-based tokenization"
source_type: article
url: "multiple -- see source log"
author: "multiple sources"
date_accessed: "2026-02-27"
status: processed
research_tool: "web-search"
research_query: "unsupervised clustering bioacoustic vocalizations 2025 state of the art"
research_depth: "deep"
---

# Unsupervised Clustering of Bioacoustic Vocalizations: State of the Art in 2025

The field has converged on a dominant two-stage paradigm: extract learned embeddings from a pretrained or self-supervised model, then cluster in that embedding space using UMAP dimensionality reduction followed by HDBSCAN. The most important shift between 2023 and 2025 is that foundation models (Perch 2.0, BirdNET, AVES, NatureLM-audio) now provide off-the-shelf embeddings that outperform custom-trained autoencoders for most species-level clustering tasks, while custom autoencoders and VAEs retain their advantage for fine-grained within-species repertoire analysis (syllable types, individual identity). For mouse USVs specifically, the Goffinet VAE remains the reference unsupervised method, with SqueakOut (2024) adding a critical upstream improvement in segmentation quality that feeds better data to any downstream clustering pipeline.

---

## 1. The Embedding + HDBSCAN Pipeline Is Now Standard

The dominant unsupervised clustering workflow for bioacoustic vocalizations in 2025 follows three steps: (1) extract embeddings from a pretrained model, (2) reduce dimensionality with UMAP, (3) cluster with HDBSCAN. This pipeline was formalized by Best et al. (2023), who showed that learned audio embeddings match species-specific models for vocalization clustering across six species including Bengalese finches, humpback whales, and bottlenose dolphins.

Best et al. trained convolutional autoencoders on spectrogram representations using perceptual loss (VGG-based rather than pixel-level MSE), producing 256-dimensional bottleneck representations. UMAP reduced these to 8 dimensions before HDBSCAN clustering. On eight datasets across six species, this pipeline achieved NMI scores of 0.5-0.88, with 46-97% of clusters meeting a 90% purity threshold. A critical finding: a generic autoencoder trained on all datasets simultaneously performed nearly as well as species-specific models, suggesting cross-species transfer of vocalization structure. The authors released an open-source Python package for this pipeline.

HDBSCAN has become the default clustering algorithm because it automatically determines cluster count, handles variable cluster shapes and densities, explicitly models noise points (important for bioacoustic data), and scales to large datasets with O(n log n) complexity.

---

## 2. Foundation Models Dominate Species-Level Clustering

A comprehensive evaluation by Muenster et al. (2025, arXiv 2504.06710) tested 15 bioacoustic deep learning models on clustering and novel class recognition. The models included purpose-built bioacoustic models (BirdNET, Perch, Animal2Vec, ProtoCLR, AVES, BirdAVES, BioLingual, SurfPerch, Google_Whale, Insect459NET, Insect66NET) and general audio baselines (AudioMAE, BEATs, EAT).

Key quantitative results using K-Means clustering with Adjusted Mutual Information (AMI):
- Supervised models achieved 0.418 AMI on bird data and 0.488 on frog data on average
- Self-supervised models achieved only 0.256 AMI (birds) and 0.414 (frogs)
- Perch and BirdNET "vastly outperformed" all other feature extractors -- the top six models were all supervised, bird-trained models

This is a striking finding: for species-level clustering, supervised models trained on large taxonomic datasets produce far better embedding spaces than self-supervised approaches. The embeddings from supervised models cluster novel species (not seen during training) more effectively than self-supervised models cluster anything.

Perch 2.0 (Google, 2025) represents the current state of the art for general bioacoustic embeddings. It uses an EfficientNet-B3 backbone (12M parameters), trained on 1.5M+ labeled recordings covering 14,795 species across birds, amphibians, insects, and mammals. It achieves AUROC 0.908 on BirdSet and 0.840 classification accuracy on BEANS. Remarkably, it outperforms specialized marine models on marine transfer tasks despite having almost no marine training data, suggesting its embeddings capture fundamental acoustic structure that transfers across taxa.

---

## 3. Self-Supervised Models: Strong but Not Superior

AVES (Animal Vocalization Encoder based on Self-Supervision) adapts HuBERT to bioacoustics by pretraining on 360 hours of AudioSet animal sounds. Sarkar and Magimai-Doss (ICASSP 2025) compared speech-pretrained SSL models (HuBERT, Wav2Vec2, WavLM) against AVES-Bio across three datasets (marmoset calls, marine mammals, dog barks). Their key finding: pretraining on bioacoustic data provides only marginal improvements over speech-pretrained models. HuBERT and AVES showed comparable performance (e.g., 94.18% vs 94.95% UAR on Watkins marine mammal dataset). Fine-tuning on ASR tasks yielded inconsistent results, suggesting general-purpose SSL representations already capture the features relevant to bioacoustic analysis.

A separate 2025 study (arXiv 2509.04166) on cross-species transfer found that phylogenetic proximity to humans does not influence transfer learning effectiveness from speech models. This means HuBERT pretrained on English speech transfers equally well to primate calls and bird songs -- the acoustic features it learns (spectral envelopes, temporal modulations, harmonic structure) are fundamental across vocal production systems.

BirdMAE, a masked autoencoder trained on large-scale bird song data, achieved the best performance on the BirdSet benchmark according to the Foundation Models comparative review (arXiv 2508.01277). The review also noted that transformer-based models require attentive probing (not just linear probing) to extract their full representational power.

---

## 4. TweetyBERT: Self-Supervised Birdsong Analysis

TweetyBERT (Goffin et al., 2025, eLife) applies masked spectrogram prediction to birdsong, operating directly on spectrograms at 2.7ms temporal resolution -- 10x finer than speech models. The compact architecture (2.5M parameters: 4 convolutional layers + 4 transformer encoder blocks) learns to reconstruct 25% masked spectrogram segments without any labels.

The model autonomously discovered canary syllable units as elliptical trajectories in embedding space -- a pattern matching theoretical biophysical models of song production. HDBSCAN clustering of these embeddings achieved a V-measure of 0.88, close to human inter-annotator agreement. Linear probes on frozen embeddings achieved 2.5% total frame error rate (vs. 1.3% for fully supervised fine-tuning).

A notable application: the model captured seasonal vocal plasticity, detecting embedding density shifts between breeding and non-breeding seasons without any temporal labels. This demonstrates unsupervised detection of biologically meaningful variation -- exactly what is needed for mouse USV analysis across behavioral conditions.

---

## 5. Vector Quantization for Discrete Vocalization Tokens

Sarkar and Magimai-Doss (arXiv 2511.10190, November 2025) directly tested whether VQ token sequences from HuBERT embeddings can capture temporal structure in animal vocalizations. Using a codebook of K=50 on four datasets (marmoset, dog), they found:

- VQ token sequences do discriminate call types and callers (pairwise distance analysis shows expected separation patterns)
- However, VQ underperformed linear baselines by 15-39% for call-type classification and 15-71% for caller identification
- Gumbel-softmax VQ performed poorly, confirming codebook collapse issues
- Call-type classification substantially outperformed caller identification, meaning individual identity is lost during discretization

This aligns with the vault's existing finding that post-hoc VQ underperforms continuous representations. The authors concluded a single codebook may lack sufficient expressiveness, particularly for fine-grained distinctions -- potentially motivating multi-codebook approaches (RVQ) or larger codebook sizes.

---

## 6. Audio-Language Foundation Models: NatureLM-audio

NatureLM-audio (November 2024, arXiv 2411.07186) represents a new paradigm: an audio-language model combining BEATs audio encoder, Q-Former connector, and Llama 3.1-8B language model. Trained on 15,000+ hours spanning bioacoustic recordings (Xeno-canto, iNaturalist), speech, and music, it achieves zero-shot species identification (19.6% accuracy on held-out species by scientific name, vastly outperforming 0.4% from general audio baselines). It also performs novel tasks never seen during training, like counting individual callers (38.3% accuracy vs 24.3% random).

While NatureLM-audio is primarily a classification/captioning model, its intermediate representations could serve as clustering features. The cross-modal training (audio + language) may capture semantic structure that purely acoustic models miss. However, its 8B-parameter language model makes it impractical for embedding extraction compared to smaller specialized models.

---

## 7. Mouse USV-Specific Methods

For rodent USV analysis specifically, the landscape in 2025 looks like this:

**Goffinet VAE (eLife, 2021)** remains the reference method for unsupervised mouse USV analysis. The shotgun-VAE trains on randomly sampled audio segments, learning latent features sufficient to reconstruct continuous sequences. Gaussian mixture model clustering in the VAE latent space only supported k of 2 or fewer clusters, providing the key quantitative evidence that mouse USVs form a continuum rather than discrete categories.

**SqueakOut (2024)** adds a critical upstream capability: autoencoder-based USV segmentation achieving 90.22 Dice score (vs 63.82 for VocalMat). The lightweight MobileNetV2-based architecture (4.6M parameters, 18MB) runs inference on 64 spectrograms in under 0.035 seconds. The authors explicitly designed SqueakOut to feed into downstream unsupervised pipelines: "The resulting segmentation masks can be used for downstream analysis using unsupervised methods such as Variational Autoencoders and dimensionality reduction techniques such as UMAP." Better segmentation directly improves clustering quality by removing noise contamination.

**AMVOC** provides an MIT-licensed Python convolutional autoencoder specifically designed for mouse USV feature extraction and clustering, operating on spectrograms with UMAP visualization.

**DeepSqueak v3.1** added VAE-based contour-invariant clustering as an upgrade over k-means, responding to the evidence that USVs resist discrete categorization.

**MUPET** uses gammatone filterbank features with unsupervised k-means to discover 100-140 data-driven syllable types -- a handcrafted-feature baseline that learned embeddings have since surpassed.

---

## 8. The CASE Benchmark

Schneider et al. (2022) published CASE (Cluster and Analyze Sound Events), a systematic comparison of 48 clustering methods combined with various audio transformations for animal vocalizations. The methods tested include community detection, affinity propagation, HDBSCAN, and fuzzy clustering, paired with classifiers including k-nearest neighbor, dynamic time warping, and cross-correlation. CASE uses a windowed, multi-feature extraction approach and provides an open benchmark for evaluating unsupervised vocal classification methods. While published in 2022, it remains a relevant benchmark and has been cited in 2025 clustering evaluations.

---

## 9. Emerging Directions

**Listening Without Labels (UCSD, 2025)**: An active research project using autoencoders to compress spectrograms into vectors, followed by UMAP and HDBSCAN/GMM clustering for unsupervised bioacoustic discovery without any labeled data. Combines this with knowledge graphs for organizing discovered sound types.

**Automated Note Annotation (Ecological Informatics, 2025)**: A pipeline achieving 93% correct annotation of target species acoustic notes through unsupervised clustering of extracted features, while eliminating 88% of false positives and retaining 95% true positives. Demonstrates that unsupervised clustering can serve as a post-processing step after initial detection.

**Hierarchical Contrastive Learning**: Upcoming work (ICASSP 2025) on acoustic identification of individual animals uses hierarchical contrastive learning to learn embeddings that cluster by both species and individual identity simultaneously.

---

## Source Log

| # | URL | Status | Relevance | Key Finding |
|---|-----|--------|-----------|-------------|
| 1 | https://arxiv.org/abs/2504.06710 | fetched | high | 15 bioacoustic models benchmarked for clustering; supervised models vastly outperform self-supervised for species-level clustering |
| 2 | https://arxiv.org/html/2508.01277v1 | fetched | high | Comparative review of 15 foundation models for bioacoustics; BirdMAE tops BirdSet, BEATs tops BEANS |
| 3 | https://arxiv.org/html/2501.05987v1 | fetched | high | Speech SSL models match bioacoustic-pretrained models; domain-specific pretraining gives only marginal gains |
| 4 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10332598/ | fetched | high | Best et al. autoencoder+UMAP+HDBSCAN pipeline for vocalization clustering across 6 species |
| 5 | https://arxiv.org/html/2508.04665v1 | fetched | high | Perch 2.0: 14,795 species, EfficientNet-B3, SOTA on BirdSet and BEANS |
| 6 | https://arxiv.org/html/2511.10190v1 | fetched | high | VQ of HuBERT embeddings discriminates call types but underperforms continuous representations by 15-39% |
| 7 | https://arxiv.org/html/2411.07186v1 | fetched | medium | NatureLM-audio: audio-language foundation model with zero-shot capabilities |
| 8 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12027336/ | fetched | high | TweetyBERT: self-supervised masked prediction achieves V-measure 0.88 for birdsong clustering |
| 9 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11071348/ | fetched | high | SqueakOut: autoencoder USV segmentation (Dice 90.22) feeding into downstream unsupervised analysis |
| 10 | https://www.sciencedirect.com/science/article/pii/S1574954125002316 | blocked (403) | medium | Automated note annotation with unsupervised clustering achieving 93% accuracy |
| 11 | https://www.mdpi.com/2076-2615/12/16/2020 | skipped | medium | CASE benchmark (2022) -- 48 clustering methods, referenced but not re-fetched |
| 12 | https://elifesciences.org/articles/67855 | skipped | medium | Goffinet VAE (2021) -- already well-covered in vault, referenced for context |
| 13 | https://arxiv.org/html/2509.04166v1 | skipped | medium | Cross-species transfer from speech -- search summary captured key finding |
| 14 | https://github.com/earthspecies/aves | skipped | low | AVES GitHub repo -- model already known to vault |
| 15 | https://e4e.ucsd.edu/news-and-updates/aid-knowledge-graphs-reu-2025 | skipped | low | UCSD REU project description, limited technical detail |

## Research Context

- **Query**: state of the art in unsupervised clustering methods for bioacoustic vocalizations 2025
- **Depth**: deep (auto-detected -- broad survey, multi-faceted, theoretical + practical)
- **Existing vault knowledge**: Strong coverage of VQ-VAE pipeline, Goffinet VAE, AMVOC, AVES, DeepSqueak, speech SSL transfer. Weak on: foundation model benchmarks (Perch 2.0, BirdMAE), TweetyBERT, SqueakOut, systematic comparison of embedding models for clustering, NatureLM-audio.
- **Knowledge gap addressed**: (1) Systematic comparison showing supervised foundation models outperform self-supervised for clustering, (2) TweetyBERT masked prediction approach, (3) SqueakOut segmentation as upstream clustering enabler, (4) Perch 2.0 capabilities and benchmark results, (5) NatureLM-audio as new paradigm, (6) Quantitative VQ performance gap on bioacoustic data
