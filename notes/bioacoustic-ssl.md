---
description: Self-supervised learning paradigms, foundation models, cross-species transfer, and the USV-specific gap at 300 kHz -- pretrained representations for bioacoustic analysis
type: moc
topics: "[[index]]"
---

# bioacoustic-ssl

Self-supervised and foundation model approaches for learning audio representations from bioacoustic data. Covers the paradigm landscape (masked prediction vs contrastive), domain-specific vs generic pretraining, cross-species transferability, and the specific challenges of applying these models to rodent USVs at 300 kHz sample rate.

## Self-Supervised Transfer Learning
- [[AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks]] -- potential alternative backbone for VQ-VAE
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] -- speech SSL models as bootstrap strategy
- [[phylogenetic proximity to humans does not influence transfer learning effectiveness from speech models to animal vocalizations]] -- acoustic features are universal across vocal production systems
- [[TweetyBERT self-supervised masked spectrogram prediction discovers birdsong syllable units matching biophysical models]] -- V-measure 0.88 birdsong clustering; discovered elliptical syllable trajectories
- [[self-supervised models detect seasonal vocal plasticity without temporal labels demonstrating unsupervised biological discovery]] -- TweetyBERT detected breeding season shifts without labels
- [[TweetyBERT self-supervised masked spectrogram prediction addresses temporal resolution limitations of speech SSL models for animal vocalizations]] -- 2.7ms resolution (10x finer than speech models)

## SSL Paradigm Comparison
- [[masked prediction outperforms contrastive learning for bioacoustic representation tasks]] -- BEATs/HuBERT beat wav2vec2; predicting pseudo-labels > contrastive future prediction
- [[CPC has been superseded by masked prediction and masked autoencoding for audio representation learning]] -- CPC foundational but wav2vec2 never wins modern benchmarks
- [[domain-specific MAE pretraining dramatically outperforms generic Audio-MAE for bioacoustic tasks]] -- Bird-MAE gained 10.6 MAP points over generic Audio-MAE
- [[combined self-supervised pretraining followed by supervised post-training yields best bioacoustic representations]] -- OpenBEATs: data diversity > architecture
- [[increasing pretext task difficulty improves embedding quality for downstream few-shot classification]] -- Perch 2.0: harder problems produce better embedding spaces

## Cross-Lab SSL Collaborations
- [[GmSLM is a London-Omer collaboration applying self-supervised speech models to marmoset vocalizations]] -- Sternberg et al. 2025 (EMNLP Findings); direct London-Omer collaboration using SSL on primate calls

## USV-Specific SSL Gap
- [[no self-supervised foundation model has been applied to rodent USV data]] -- 50-90 kHz range at 300 kHz sample rate remains untested
- [[spectrogram-based SSL avoids the sample rate mismatch that limits waveform-based models for USV analysis]] -- spectrograms abstract away sample rate
- [[the 300 kHz USV sample rate creates a domain shift challenge for applying audio foundation models]] -- 10-19x sampling rate gap with pretrained models
- [[frequency shifting USVs into the audible range could enable classification with standard audio foundation models]] -- pitch-shift 50-90 kHz to 2-10 kHz; untested but theoretically sound

## Foundation Models & Cross-Species Transfer
- [[supervised bioacoustic foundation models vastly outperform self-supervised for species-level clustering]] -- Muenster 2025: supervised 0.418 AMI vs self-supervised 0.256
- [[Perch 2.0 trained on 14795 species achieves state of the art bioacoustic embeddings that transfer across taxa]] -- EfficientNet-B3, AUROC 0.908 on BirdSet, transfers to marine
- [[NatureLM-audio combines BEATs encoder with Llama 3.1-8B for zero-shot bioacoustic species identification]] -- first audio-language model for bioacoustics; impractical for embeddings
- [[a generic cross-species autoencoder performs nearly as well as species-specific models suggesting shared vocalization structure]] -- universal acoustic features across vocal production systems
- [[transformer-based bioacoustic models require attentive probing not just linear probing to extract full representational power]] -- linear probes underestimate transformer capacity
- [[perceptual loss outperforms pixel-level MSE for autoencoder spectrogram representation learning]] -- VGG-based loss focuses on structural features
- [[BEATs self-distilled discrete tokenizer achieves the highest BEANS benchmark score among bioacoustic encoders]] -- 97.98 AUROC on BEANS
- [[prototypical probing with frozen MAE features enables bioacoustic classification with as few as 10 labeled examples]] -- 37% MAP improvement over linear probing
- [[foundation model embeddings enable few-shot classification via simple linear probes without end-to-end training]] -- Perch 2.0/BEATs/AVES provide off-the-shelf embeddings

## Open Questions
- Whether domain-specific MAE pretraining on USV spectrograms would bridge the performance gap
- How to evaluate SSL embeddings when ground-truth USV categories don't exist
- Whether TweetyBERT-style masked prediction at 2.7ms resolution would capture USV frequency modulations

## Related Areas
- [[representation-learning]] -- the VQ-VAE pipeline that could use SSL embeddings as input features
- [[model-adaptation]] -- LoRA/PEFT for efficient adaptation of frozen SSL models to USV tasks
- [[transformer-architecture]] -- the attention and MLP mechanics underlying these models
- [[classification]] -- few-shot classification using foundation model embeddings

---

Topics:
- [[index]]
