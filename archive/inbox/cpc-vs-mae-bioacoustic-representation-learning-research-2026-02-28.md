---
description: "Comparative analysis of CPC vs MAE paradigms for bioacoustic feature learning, with BEANS/BirdSet benchmarks and USV gap analysis"
source_type: paper
url: "https://arxiv.org/abs/2508.11845, https://arxiv.org/abs/2501.05987, https://arxiv.org/abs/2504.12880, https://arxiv.org/abs/2508.01277"
author: "Miron et al. 2025; Sarkar & Magimai-Doss 2025; Schwinger et al. 2025 (review)"
date_accessed: "2026-02-28"
status: processed
research_tool: "claude-code-web-search"
research_query: "contrastive predictive coding vs masked autoencoder bioacoustic feature learning USV"
---

# CPC vs MAE for Bioacoustic Representation Learning (2024-2026 Landscape)

## Context
Three self-supervised learning (SSL) paradigms compete for audio representation learning: (1) Contrastive learning (CPC, wav2vec 2.0), (2) Masked prediction (HuBERT, AVES, WavLM), (3) Masked autoencoding (Audio-MAE, BirdMAE, BEATs). This research synthesizes recent benchmarks to determine which paradigm leads for bioacoustic applications, with specific attention to USV relevance.

## Key Points

### Paradigm Definitions
- **Contrastive Predictive Coding (CPC)**: Learns by predicting future latent representations from past context, training via InfoNCE loss to distinguish true future frames from negative samples. Wav2Vec 2.0 extends this with quantized latents and contrastive loss on masked positions.
- **Masked Autoencoder (MAE)**: Masks random patches of input spectrograms and trains the model to reconstruct masked regions. Audio-MAE, BirdMAE, and AudioMAE++ use this approach on mel/log-mel spectrograms with Vision Transformer architectures.
- **Masked Prediction (MP)**: Masks input positions and predicts discrete pseudo-labels (from k-means clustering) rather than reconstructing raw input. HuBERT, AVES, and WavLM use this approach. BEATs learns its own discrete audio tokenizer.

### BEANS Benchmark Results (AUROC, Attentive Probing) -- Schwinger et al. Aug 2025
| Model | Paradigm | BEANS AUROC | BirdSet AUROC |
|-------|----------|-------------|---------------|
| BEATsNLM | MP+generative | 98.14 | 82.45 |
| BEATs | MP (self-distilled) | 97.98 | 82.28 |
| EAT | MP | 97.51 | -- |
| BirdMAE | MAE (domain-specific) | 97.27 | 80.24 |
| AudioMAE | MAE (general) | 97.19 | 81.05 |
| AVES | MP (HuBERT-based) | 97.16 | -- |

Note: CPC and wav2vec 2.0 were NOT included in the BEANS benchmark comparison. The main studies focus on masked approaches.

### Linear Probing Tells a Different Story
With linear probing (simpler, less expressive), BEATs scored 94.10 and AudioMAE dropped to 84.47 on BEANS. Transformer models need attentive probing to unlock their representations; CNNs are more accessible with linear probes.

### Contrastive vs Masked Prediction: Direct Comparison (Sarkar & Magimai-Doss 2025)
| Model | Paradigm | IMV (Marmoset) | Watkins (Marine) | Abzaliev (Dogs) |
|-------|----------|----------------|-------------------|-----------------|
| HuBERT | Masked prediction | 64.35% UAR | 94.18% | 47.96% |
| AVES-Bio | Masked prediction | 62.54% | 94.95% | 54.23% |
| Wav2Vec2 | Contrastive | 62.40% | 94.25% | 48.95% |
| WavLM | Masked prediction | 58.98% | 94.78% | 43.97% |

Key finding: Wav2Vec2 (contrastive) did NOT outperform masked prediction models. HuBERT and AVES consistently matched or exceeded wav2vec2.

### Audio-MAE Domain Transfer Problem
- Generic Audio-MAE pretrained on AudioSet performs WORSE than simple supervised spectrogram features on bioacoustic tasks (Bird-MAE paper, ICLR 2025 area).
- Solution: Domain-specific MAE pretraining. Bird-MAE pretrained on bird audio improved from 44.69 to 55.28 MAP on BirdSet HSN dataset, a 10.6 point gain.
- Bird-MAE even outperformed fully supervised models (Perch: 41.12 MAP vs Bird-MAE: 55.26 MAP on POW).

### Combined Training is the New SOTA
- "What Matters for Bioacoustic Encoding" (Miron et al., Aug 2025): Self-supervised pretraining FOLLOWED BY supervised post-training on mixed bioacoustic + general audio yields best in-distribution AND out-of-distribution performance.
- OpenBEATs (pretrained on 20K hours: music + environmental + bioacoustic) outperformed BEATs and even Dasheng (1.2B params) on cross-domain evaluation.
- Data diversity in both pretraining stages matters more than model architecture choice.

### CPC Specifically for USV Data: Gap Confirmed
- **No published work applies CPC to rodent USV data.** Extensive search found zero results.
- **No published work applies ANY self-supervised foundation model specifically to mouse/rodent USVs.** The closest work uses these models on marmosets, dogs, marine mammals, bats, and birds.
- Mouse USV tools (DeepSqueak, VocalMat, BootSnap, MUPET, DAS) all use traditional supervised CNN/ML approaches.
- The 50-90 kHz USV frequency range and 300 kHz sample rate pose a unique domain shift challenge -- most SSL audio models are pretrained on 16-48 kHz sample rate data.

### TweetyBERT: An Alternative SSL Approach for Vocalizations
- TweetyBERT (2025) uses a transformer-based self-supervised approach for birdsong parsing that avoids the fixed 20ms kernel limitation of wav2vec2/HuBERT.
- Achieved V-measure of 0.88 for unsupervised canary song clustering in latent space.
- Relevant because it explicitly addresses the temporal resolution mismatch between human speech SSL models and shorter animal vocalizations.

### Few-Shot Learning Capability
- Bird-MAE with prototypical probing achieves competitive performance with just 10 labeled examples per class.
- This is highly relevant for USV research where labeled data is scarce.
- Prototypical probing improved frozen MAE representations by 37% MAP over linear probing.

## Raw Notes

### Why CPC is losing ground
CPC (van den Oord et al. 2018) was foundational but has been superseded by two waves:
1. **Masked prediction (2020-2022)**: HuBERT showed that predicting discrete pseudo-labels from k-means is more effective than contrastive future prediction. The key insight: contrastive loss requires careful negative sampling, while masked prediction avoids this.
2. **Masked autoencoding (2022-2024)**: Audio-MAE showed that reconstructing masked spectrogram patches learns rich visual-style representations. BEATs combined this with a learned tokenizer.

The field has largely moved past pure CPC. Wav2Vec 2.0 (which is CPC-derived) still appears in comparisons, but never wins. The "contrastive" component increasingly appears as a regularizer in hybrid approaches rather than the primary training objective.

### Sample rate mismatch problem for USV
Most SSL audio models operate at 16 kHz (speech) or 32 kHz (general audio). USV recordings at 300 kHz present a 10-19x sampling rate gap. Options:
1. Downsample USVs to the model's expected rate -- loses all ultrasonic content (defeats the purpose)
2. Retrain from scratch on 300 kHz data -- requires substantial compute
3. Train on spectrograms rather than raw waveforms -- our current approach, and what MAE-based methods actually do

This is why spectrogram-based approaches (MAE, our transformer) may be better suited for USV than waveform-based approaches (CPC, wav2vec2, HuBERT). Spectrograms abstract away the raw sample rate.

### Implications for our VQ-VAE pipeline
Our transformer -> VQ-MAE pipeline already follows the masked prediction paradigm family. The research validates this choice over contrastive approaches. Specific validations:
1. Masked approaches outperform contrastive on bioacoustic benchmarks
2. Domain-specific pretraining dramatically outperforms generic models
3. Our spectrogram-based approach avoids the sample rate mismatch problem
4. Combined SSL + supervised finetuning is the emerging best practice

### Key open question
Would pretraining a MAE on USV spectrograms (as Bird-MAE did for bird audio) provide a better backbone than our autoregressive transformer approach? The Bird-MAE results suggest potentially yes, but our approach has the added benefit of learning temporal/sequential structure through autoregressive prediction rather than just spatial reconstruction.

## Processing Notes
Processed 2026-03-01. 9 notes extracted covering SSL paradigm comparison, domain-specific MAE, combined training, USV gap, sample rate mismatch, BEATs benchmarks, few-shot probing, TweetyBERT, and CPC obsolescence.

### Existing vault overlap
- AVES note exists -- this research adds benchmark numbers and paradigm comparison context
- Speech SSL transfer note exists -- this adds wav2vec2 vs HuBERT direct comparison numbers
- Sarkar post-hoc VQ note exists -- no new information on that specific topic

### Novel claims for extraction
- Masked prediction outperforms contrastive learning for bioacoustic tasks (with benchmark numbers)
- Domain-specific MAE pretraining dramatically outperforms generic Audio-MAE
- Combined SSL + supervised post-training yields SOTA bioacoustic representations
- No self-supervised foundation model has been applied to rodent USVs (gap)
- Spectrogram-based SSL avoids sample rate mismatch affecting waveform-based models for USV
- BEATs self-distilled discrete tokenizer achieves highest BEANS score
- Few-shot prototypical probing with MAE features enables low-data bioacoustic classification
- TweetyBERT addresses temporal resolution limitations of speech SSL models for animal vocalizations
