---
description: "Few-shot and zero-shot learning methods for animal sound classification — foundation model embeddings, prototypical networks, transductive inference, and their applicability to rare USV call types with <10 labeled examples"
source_type: paper
url: "multiple — see source list below"
author: "multiple — Stowell et al 2023, Ghani et al (Perch 2.0) 2025, Robinson et al (NatureLM-audio) 2025, DCASE challenge teams 2021-2024"
date_accessed: "2026-02-28"
status: processed
research_tool: "claude-code web search"
research_query: "few-shot learning animal sound classification bioacoustics, prototypical networks animal vocalizations, DCASE few-shot bioacoustic event detection, BEATs audio spectrogram transformer bioacoustics"
---

# Few-Shot Learning for Animal Sound Classification (2023-2026)

Research synthesis on methods that classify animal vocalizations from very few labeled examples (<10 per class), with focus on applicability to rare USV call types.

## Key Points
- Foundation models (Perch 2.0, BEATs, AVES, NatureLM-audio) now provide high-quality audio embeddings that enable few-shot classification via simple linear probes or prototypical networks — no end-to-end training needed
- The DCASE few-shot bioacoustic event detection challenge (2021-2024) established prototypical networks + transductive inference as the dominant paradigm, improving from F1 40% to 70%+ across editions
- The "5-shot" detection paradigm is now standard: given 5 annotated examples of a sound class, detect all instances in long recordings — directly applicable to rare USV types
- No existing work applies few-shot learning specifically to USV syllable-type classification, making this a clear research gap

## Source List
1. Stowell et al. "Learning to detect an animal sound from five examples" (2023) — https://arxiv.org/abs/2305.13210
2. Google DeepMind Perch 2.0 (August 2025) — https://arxiv.org/html/2508.04665v1
3. Robinson et al. "NatureLM-audio: Audio-Language Foundation Model for Bioacoustics" (2025) — https://arxiv.org/html/2411.07186
4. DCASE 2024 Task 5: Few-shot Bioacoustic Event Detection — https://dcase.community/challenge2024/task-few-shot-bioacoustic-event-detection
5. You & Coyotl "Transformer-Based Bioacoustic Sound Event Detection on Few-Shot Learning Tasks" (2023) — https://www.amazon.science/publications/transformer-based-bioacoustic-sound-event-detection-on-few-shot-learning-tasks
6. "Reshaping Bioacoustics Event Detection: FSL with Transductive Inference and Data Augmentation" (2024) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11274013/
7. "Regularized Contrastive Pre-training for Few-shot Bioacoustic Sound Detection" (2024) — https://arxiv.org/html/2309.08971
8. Multi-modal Language Models for Bioacoustics with Zero-Shot Transfer (Feb 2025) — https://www.nature.com/articles/s41598-025-89153-3
9. OpenBEATs: Fully Open-Source General-Purpose Audio Encoder (2025) — https://arxiv.org/html/2507.14129v1
10. "What Matters for Bioacoustic Encoding" (2025) — https://arxiv.org/pdf/2508.11845
11. Sainburg et al. AVGN — latent structure across animal vocal repertoires (2020) — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008228
12. Bioacoustic classification of a small dataset of mammalian vocalisations using deep learning (2024) — https://www.tandfonline.com/doi/full/10.1080/09524622.2024.2354468

## Raw Notes

### 1. Foundation Models as Embedding Extractors

The field has shifted from training classifiers from scratch to using **pre-trained foundation models as feature extractors**, then applying simple classifiers (linear probe, k-NN, prototypical networks) on top. This is the most directly applicable approach for our rare USV types.

**Perch 2.0** (Google DeepMind, Aug 2025):
- Bioacoustics foundation model trained on bird + terrestrial animal audio
- Key finding: despite zero underwater training data, Perch 2.0 embeddings transferred well to marine mammal classification (killer whale ecotype discrimination)
- "Agile modeling" paradigm: build custom classifier from small number of labeled examples in a couple of hours
- Evaluation uses few-shot linear probe — just a linear layer on frozen embeddings
- State-of-the-art on BirdSET and BEANS benchmarks
- Finding: "increasing the difficulty of the classification problem increases overall quality of the embedding model" — harder pretext tasks produce better embeddings

**BEATs** (Microsoft, 2022) and **OpenBEATs** (2025):
- Audio pre-training with acoustic tokenizers — iterative self-supervised framework
- BEATs embeddings used directly for DCASE 2023 few-shot bioacoustic detection (Gelderblom et al.)
- OpenBEATs (2025): fully open-source, multi-domain pre-training, achieves SOTA on 6 bioacoustics datasets
- Outperforms models >1B parameters at 1/4 their size
- NatureLM-audio uses BEATs as its audio encoder, demonstrating that BEATs representations encode bioacoustically-relevant features

**AVES / BirdAVES** (Earth Species Project):
- First self-supervised foundation model specifically for animal vocalizations
- Already in our vault: [[AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks]]
- Already in our vault: [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]]

**NatureLM-audio** (Robinson et al., 2025):
- First audio-language model for bioacoustics
- Architecture: BEATs encoder + Q-former + Llama 3
- Zero-shot capabilities on unseen taxa and species
- Positive transfer from speech and music data to bioacoustics — first evidence of cross-domain transfer
- Extended BEANS benchmark with BEANS-Zero: call-type prediction, lifestage classification, captioning, individual counting
- Most relevant for our case: **call-type prediction** from audio-language model without task-specific training

**"What Matters for Bioacoustic Encoding"** (2025):
- Systematic comparison of bioacoustic encoders: AVES (HuBERT-based), Animal2Vec (data2vec-based), BirdMAE (AudioMAE-based), TweetyBert (BERT-inspired)
- First standardized pipeline comparison — previous works used incomparable setups
- Key for our decision: which embedding model to use as backbone

### 2. Prototypical Networks — The Dominant Paradigm

**Core idea**: Compute a "prototype" (centroid embedding) for each class from its few support examples, then classify query samples by nearest-prototype distance. Softmax over distances gives probabilistic predictions.

**Why prototypical networks dominate bioacoustics:**
- Natural fit for few-shot: explicitly designed for N-way K-shot classification
- Training via episodes that mirror test conditions (e.g., 5-way 1-shot)
- Softmax loss from prototypes prevents representation collapse
- Scale to zero-shot when class descriptions replace support examples

**DCASE challenge evolution** (the main competitive benchmark):
| Year | Best F1 | Key methods |
|------|---------|-------------|
| 2021 | ~40% | Template matching baseline |
| 2022 | ~60% | Prototypical networks, transductive learning, metric learning |
| 2023 | ~63% | Supervised contrastive learning, expanded species |
| 2024 | ~70%+ | Embedding learning, cross-dataset augmentation, domain adaptation |

**Task setup**: 5-shot — given 5 annotated start/end times of a sound class, detect all instances in long recording. Single class of interest per recording.

**Top DCASE methods and techniques:**

a) **Segment-level metric learning** (Surrey, DCASE 2022 1st place):
- Per-channel energy normalization (PCEN) + delta MFCCs as input features
- Better utilization of negative data in loss function
- Transductive inference for test-time adaptation
- Key insight: PCEN is more robust than log-mel for few-shot scenarios

b) **Supervised contrastive learning** (Moummad_IMT, DCASE 2023 2nd place):
- Contrastive learning to maximize distinction between positive and negative events
- Encoder fine-tuned on the 5 provided positive examples per file
- Contrastive loss more effective than cross-entropy for few-shot

c) **Transductive inference + data augmentation** (2024):
- Transductive inference: uses unlabeled test data to iteratively refine class prototypes and feature extractors
- SpecAugment on Mel spectrograms for augmentation
- 27% F-score improvement over non-transductive baseline (DCASE 2022 dataset)
- Key technique for our case: since we have lots of unlabeled USV spectrograms, transductive inference could use those to refine prototypes

d) **Adaptive learning with negative prototype construction** (2024):
- Novel adaptive learning loss for classifier updates
- Negative selection strategy for constructing representative negative prototypes
- Problem: no explicitly annotated negatives in few-shot setup, so negative prototype must be constructed carefully
- F-measure 0.703 on DCASE 2023 Task 5 (12.84% improvement)

e) **Cross-dataset data augmentation + domain adaptation** (DCASE 2024):
- Instance-wise Feature Projection-based Domain Adaptation (IFPDA)
- Modified ResNet for multitask learning: multi-class species classification + binary frame-level detection
- Addresses domain shift between training and evaluation species

### 3. Multi-Modal and Zero-Shot Approaches

**Multi-modal language models** (Scientific Reports, Feb 2025):
- Apply large language model reasoning to audio classification
- Zero-shot transfer: classify sounds from text descriptions alone, no labeled audio needed
- Particularly relevant for well-described but rarely recorded call types
- Limitation: requires good text descriptions of target classes

### 4. The USV-Specific Gap

No existing work applies few-shot learning specifically to USV syllable-type classification. The closest are:
- VocalMat: CNN for 11 USV types but requires thousands of labeled examples per type, accuracy drops to 86% overall and much worse for rare types like "reverse chevron"
- DeepSqueak: VAE-based clustering (unsupervised, not few-shot)
- MUPET: gammatone + k-means (unsupervised)

**Why few-shot is ideal for USVs:**
- Mouse USV types follow highly skewed distributions — some types appear in <1% of calls
- Manual labeling is expensive and operator-dependent
- New experimental conditions may reveal novel call types with few initial examples
- Our lab situation: <10 labeled examples for rare call types, exactly the few-shot regime

### 5. Practical Applicability to Our Pipeline

**Most promising approach for us (in order of effort vs payoff):**

1. **Embedding extraction from pre-trained model** (LOW effort, HIGH payoff):
   - Use BEATs or Perch 2.0 to embed our spectrogram patches
   - Train simple linear probe or prototypical network on embeddings
   - Works even with 5-10 examples per class
   - No need to train or fine-tune the embedding model
   - Challenge: our spectrograms are 300 kHz (150 kHz Nyquist) — most foundation models trained on <48 kHz audio. Would need to either: (a) downsample to audible range for embedding extraction (losing frequency info), or (b) treat spectrograms as images and use vision models

2. **Prototypical network on our existing CNN features** (MEDIUM effort, MEDIUM payoff):
   - Extract features from intermediate layers of our trained USVClassifierCNN
   - Build prototypical network on those features for multi-class syllable classification
   - Advantage: features already tuned for USV spectrograms at 300 kHz
   - Disadvantage: features trained for binary (USV/noise) may not capture within-class differences

3. **Contrastive fine-tuning** (HIGH effort, HIGH payoff):
   - Fine-tune embedding model with supervised contrastive loss on our labeled USVs
   - Uses all available labels, not just per-class
   - Produces embeddings that cluster syllable types
   - Then few-shot classify rare types via prototypical networks

4. **Transductive inference** (MEDIUM effort as ADD-ON):
   - Use unlabeled USV detections to refine prototypes
   - We have thousands of unlabeled detections — perfect for this
   - Add on top of any of the above approaches

**The 300 kHz challenge:**
- Most audio foundation models are trained on 16-48 kHz audio
- Our USVs are 25-120 kHz, recorded at 300 kHz
- Options: (a) frequency-shift trick — pitch-shift USVs into audible range before embedding, (b) treat mel-spectrograms as images and use vision-based few-shot (e.g., CLIP embeddings on spectrogram images), (c) train custom embedding model on ultrasonic audio specifically
- VocalMat and DeepSqueak both handle this by operating on spectrogram images rather than raw audio

## Processing Notes
{After /reduce: what was extracted, what was skipped, what needs follow-up}
