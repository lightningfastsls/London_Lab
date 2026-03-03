---
description: "Pitch-shifting 50-90 kHz USVs down to 2-10 kHz preserves relative spectral structure while matching foundation model input expectations -- an untested but theoretically sound approach"
type: method
confidence: speculative
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
  - "[[signal-processing]]"
  - "[[classification]]"
---

# frequency shifting USVs into the audible range could enable classification with standard audio foundation models

Standard audio foundation models (BEATs, Perch 2.0, AVES) are trained on audio at 16-48 kHz sample rates. One approach to applying them to 300 kHz USV recordings is frequency shifting: pitch-shift USVs from their native 50-90 kHz range down to 2-10 kHz, preserving relative spectral contours and temporal modulations while bringing them within the model's expected frequency range. The shifted audio could then be resampled to 16 kHz and processed through any standard audio embedding pipeline.

This approach preserves the relative structure -- frequency sweeps, modulations, harmonic relationships, and temporal durations -- that define syllable types. What it loses is absolute frequency information: a USV at 70 kHz and one at 50 kHz would be shifted to different absolute frequencies but would maintain their relative positions within the call. The tradeoff is acceptable if syllable types are defined primarily by contour shape rather than absolute frequency, which is consistent with how both human experts and existing tools like DeepSqueak classify USVs (by sweep direction, duration, and modulation pattern).

The implementation would be straightforward: apply a frequency shift in the time domain (multiply by a complex exponential to shift the spectrum) or equivalently decimate the spectrogram. However, this has not been tested in any published work, making it a novel experimental direction. The risk is that foundation models may have learned frequency-dependent features during pretraining that do not transfer to shifted content. But the Perch 2.0 finding that embeddings transfer across dramatically different acoustic domains (terrestrial to marine) suggests the learned features are more about spectrotemporal patterns than absolute frequencies.

---

Source:
- few-shot-learning-animal-sound-classification-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[the 300 kHz USV sample rate creates a domain shift challenge for applying audio foundation models]] -- the problem this method addresses

Topics:
- [[bioacoustic-ssl]]
- [[signal-processing]]
- [[classification-methodology]]
