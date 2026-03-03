---
description: "Creating training data by mixing isolated single-source recordings simulates overlapping scenarios — viable for USVs since single-animal recordings provide clean individual call examples"
type: method
confidence: proven
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[signal-processing]]"
---

# Synthetic mixture training is the standard approach for training bioacoustic source separation networks

BioCPPNet and related source separation systems create training data by artificially mixing isolated single-source recordings to simulate overlapping scenarios. This approach is necessary because ground-truth separated sources are generally unavailable for real overlapping recordings. For USV research, this is viable because single-animal recordings provide clean individual call examples that can be mixed with controlled overlap parameters (timing, frequency, relative amplitude). The training data generator can systematically explore the space of overlap conditions: partial temporal overlap, full overlap, same-frequency crossing, and different-frequency co-occurrence. This synthetic approach has been validated across macaques, dolphins, and bats in BioCPPNet. For our pipeline, single-animal USV recordings already exist and could serve as source material for generating synthetic mixtures.

---

Source:
- overlapping-usv-source-separation-state-of-art-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[BioCPPNet U-Net architecture with permutation-invariant training enables single-channel bioacoustic source separation]] — the architecture trained with this data

Topics:
- [[detection]]
- [[signal-processing]]
