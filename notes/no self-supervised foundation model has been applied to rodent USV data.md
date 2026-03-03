---
description: "Despite SSL models being tested on marmosets, dogs, marine mammals, bats, and birds, the 50-90 kHz USV frequency range at 300 kHz sample rate remains untested territory"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
  - "[[signal-processing]]"
---

# No self-supervised foundation model has been applied to rodent USV data

An extensive literature search found zero published work applying CPC, masked prediction, or masked autoencoding to rodent USV data. The closest work uses these models on marmosets, dogs, marine mammals, bats, and birds — but never on mouse or rat ultrasonic vocalizations. Mouse USV tools (DeepSqueak, VocalMat, BootSnap, MUPET, DAS) all use traditional supervised CNN or ML approaches, none incorporating self-supervised pretraining.

The gap exists primarily because most SSL audio models are pretrained on 16-48 kHz sample rate data, creating a 10-19x sampling rate gap with 300 kHz USV recordings. Waveform-based SSL models cannot simply be applied to USV audio without either losing the ultrasonic content through downsampling or retraining from scratch on high-sample-rate data. Spectrogram-based approaches avoid this mismatch but have not yet been specifically adapted for the USV domain.

This positions our VQ-VAE transformer pipeline as novel — it is among the first approaches to apply self-supervised representation learning specifically to rodent USVs. The combination of spectrogram-based input (avoiding the sample rate mismatch), masked prediction paradigm (validated as superior to contrastive approaches on bioacoustic benchmarks), and discrete token learning (enabling sequence analysis) addresses a genuine gap in the literature rather than replicating existing work.

---

Source:
- cpc-vs-mae-bioacoustic-representation-learning-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[spectrogram-based SSL avoids the sample rate mismatch that limits waveform-based models for USV analysis]] — the technical reason for the gap
- [[QLoRA 4-bit quantization enables 7B model fine-tuning on consumer GPUs with 33 percent memory savings at 39 percent runtime cost]] -- addresses the HPC barrier for USV-specific model adaptation: consumer GPU fine-tuning could enable labs without datacenter access to adapt foundation models to the 300 kHz USV domain
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- LoRA offers a path to adapt existing speech SSL models to the USV domain with minimal parameters, potentially bridging this gap without training from scratch

Topics:
- [[bioacoustic-ssl]]
- [[signal-processing]]
