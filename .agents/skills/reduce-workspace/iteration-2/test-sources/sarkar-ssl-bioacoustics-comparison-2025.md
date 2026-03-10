---
description: "Compares SSL models pre-trained on human speech vs animal vocalizations for bioacoustics classification tasks"
source_type: paper
url: "https://arxiv.org/abs/2501.05987"
author: "Eklavya Sarkar, Mathew Magimai-Doss"
date_accessed: "2026-02-27"
status: unprocessed
---

# Comparing Self-Supervised Learning Models Pre-Trained on Human Speech and Animal Vocalizations for Bioacoustics Processing (Sarkar & Magimai-Doss, 2025)

Idiap Research Institute / EPFL. Published January 2025.

## Research Question

Do SSL foundation models pre-trained specifically on animal vocalizations outperform models pre-trained on human speech for bioacoustics classification? And does fine-tuning on automatic speech recognition (ASR) enhance or hurt cross-domain transfer to animal calls?

## Models Compared

1. **AVES-Bio**: Pre-trained on animal vocalizations (AudioSet animal subset). Based on HuBERT architecture but trained on bioacoustic data specifically.
2. **HuBERT**: Pre-trained on LibriSpeech (960h English). Self-supervised with masked prediction objective.
3. **Wav2Vec2**: Pre-trained on LibriSpeech, tested both base and ASR-fine-tuned variants.
4. **WavLM**: Pre-trained on 94k hours mixed audio data (LibriSpeech + GigaSpeech + VoxPopuli), tested base and ASR-fine-tuned.

All models use transformer architecture. Representations extracted from each of 12 layers independently — layer selection turns out to matter.

## Datasets

1. **Watkins Marine Mammal Sound Database** (1,697 samples, 295 min): 32 species. Task: species classification.
2. **InfantMarmosetsVox (IMV)** (72,920 samples, 464 min): Callithrix jacchus. Task: call-type classification (11 classes).
3. **Abzaliev Dog Barks** (8,034 samples, 137 min): Domestic dogs. Task: bark-type classification (14 classes).

## Evaluation Method

- Linear probing: freeze SSL model, train linear classifier on extracted features
- Layer-wise evaluation: test each transformer layer independently
- Metric: Unweighted Average Recall (UAR) — accounts for class imbalance
- 5-fold cross-validation

## Key Results (UAR % on test sets)

| Dataset | AVES-Bio | HuBERT | Wav2Vec2 | WavLM | Wav2Vec2-ft | WavLM-ft |
|---------|----------|--------|----------|-------|-------------|----------|
| IMV | 62.54 | **64.35** | 60.12 | 63.18 | 58.43 | 63.44 |
| Watkins | **94.95** | 94.18 | 91.47 | 93.81 | 89.73 | 93.93 |
| Abzaliev | **54.23** | 47.96 | 44.28 | 47.12 | 43.71 | 47.90 |

## Key Findings

### 1. Domain-specific pre-training provides minimal advantage
AVES (animal-pretrained) performed comparably to HuBERT (speech-pretrained) across all three datasets. The differences were small and inconsistent in direction — AVES won on Watkins and Abzaliev but lost on IMV. This suggests that general acoustic representations learned from human speech transfer effectively to animal vocalizations.

### 2. ASR fine-tuning hurts bioacoustic performance
Models fine-tuned on automatic speech recognition (ASR) consistently underperformed their base counterparts. "Fine-tuning on speech may push models to learn task-specific features that don't generalize" to animal calls. This is important: ASR fine-tuning makes models WORSE for bioacoustics, not better.

### 3. Layer selection matters more for fine-tuned models
Base models showed relatively stable performance across layers. Fine-tuned models showed more layer-dependent variance, suggesting ASR fine-tuning distorts upper layers while preserving lower-layer acoustic features.

### 4. Pre-trained representations are sufficient
No need for elaborate domain adaptation. Linear probing on frozen representations achieved strong results, suggesting the learned acoustic representations are already rich enough for bioacoustic classification.

## Implications for Our Pipeline

This paper suggests our approach of using pre-trained representations (currently VQ-VAE on spectrograms) is sound. Key takeaways:
- Don't assume animal-specific pre-training will help — general acoustic features transfer well
- If we consider using SSL models as feature extractors, avoid ASR-fine-tuned checkpoints
- Linear probing is a strong baseline — complex classifiers may be unnecessary
- Layer selection is an important hyperparameter to sweep

The finding that base HuBERT outperforms AVES-Bio on marmoset call-type classification is surprising and challenges the intuition that "more relevant training data = better features."

## Methodological Notes

- UAR (Unweighted Average Recall) is preferred over accuracy for imbalanced datasets — weights each class equally regardless of frequency
- 5-fold cross-validation provides robust estimates but doesn't test generalization to new recording conditions
- The study doesn't test fine-tuning the SSL models on bioacoustic data (only linear probing) — full fine-tuning might tell a different story
