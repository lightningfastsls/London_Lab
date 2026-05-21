---
description: "Mean spectral power, tonality, and notch position differ systematically between recording chambers and dominate any biological signal in raw spectrograms — classifiers trained on raw input learn the room, not the mouse"
type: claim
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[wild-lab-vocal-comparison]]"
  - "[[signal-processing]]"
  - "[[classification-methodology]]"
---

# Cage acoustics drive between-cohort spectrogram separation more than biology

The 2026-05-18 VAE comparison memo (`docs/handoffs/2026-05-18_vae_comparison_memo.md`) trained two independent VAE architectures — DeepSqueak's published model and our own — on raw spectrograms from 5970 and lab 131204. Both architectures produced near-disjoint embeddings between cohorts. The dominant axes in latent space (DS z_17 d=+2.89; ours z_10 d=−2.72) correlated above |r|=0.3 with cage-acoustic descriptors — mean_power_db, tonality, notch position, ambient noise spectrum — well past the threshold that would call them genuine cohort discriminators. The biological signal that *should* differentiate wild 5970 from lab 131204 mice (call rate, repertoire entropy, acoustic feature distributions) is real but is smaller in magnitude than the cage signature, so it gets buried in raw input.

Because the effect replicates across two independent architectures with no shared code, this is a property of the **data**, not the model. Any classifier trained on raw spectrograms without explicit cage suppression will recapitulate it — the network learns to discriminate rooms because rooms are easier to discriminate than animals. This is why the cleaning stack (soft-notch + Boll baseline subtraction + global MAD + per-recording Z-score) exists, and why the Phase 1.0 cleaning validation gate is a blocking prerequisite to any downstream classifier training. Without that gate, a 12-class syllable classifier could trivially achieve high macro F1 on held-out data by exploiting recording-environment signatures while failing on any new cage.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[per-recording normalization compensates for varying noise floors across recording sessions]] — the layer that partially mitigates this
- [[recording groups 5970 3452 2379 are all wild mouse dyads not different strains]] — why between-cohort differences can't be reduced to mouse strain
- [[transient cage noises produce broadband vertical smears rejected by the minimum duration filter]] — a different cage artifact, post-detection

Topics:
- [[wild-lab-vocal-comparison]]
- [[signal-processing]]
- [[classification-methodology]]
