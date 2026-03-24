---
description: "Muenster et al 2025 tested 15 models — supervised achieved 0.418 AMI on birds vs 0.256 for self-supervised; Perch and BirdNET dominated top 6 slots"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
  - "[[classification]]"
---

# Supervised bioacoustic foundation models vastly outperform self-supervised for species-level clustering

Muenster et al. (2025, arXiv 2504.06710) conducted the most comprehensive evaluation to date, testing 15 bioacoustic deep learning models on clustering and novel class recognition. The results were striking: supervised models achieved 0.418 Adjusted Mutual Information (AMI) on bird data and 0.488 on frog data, while self-supervised models managed only 0.256 AMI (birds) and 0.414 (frogs). The top six models were all supervised, bird-trained models — Perch and BirdNET "vastly outperformed" all other feature extractors.

This challenges the assumption that self-supervised pretraining produces the best general-purpose embeddings. For species-level clustering, models trained on large taxonomic datasets with explicit labels produce embedding spaces that cluster novel species (never seen during training) more effectively than self-supervised models cluster anything. The key insight is that taxonomic structure provides a powerful inductive bias — the model learns to separate species, and this separation transfers to unseen species.

However, this finding may not generalize to within-species repertoire analysis (syllable types, individual identity), where the relevant variation is more subtle. Since [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]], the picture is nuanced: SSL models are good enough for many tasks, but supervised models excel specifically at categorical discrimination.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks]] -- tension: AVES claimed supervised-beating performance, but this comprehensive comparison reverses that finding at scale
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] -- consistent: SSL works, but supervised works better for clustering
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- could adapt SSL models toward supervised-like clustering quality without requiring large labeled taxonomic datasets

Topics:
- [[bioacoustic-ssl]]
- [[classification]]
