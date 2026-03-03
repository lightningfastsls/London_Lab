---
description: Four augmentations (Gaussian noise, gain perturbation, frequency masking, time masking) applied at p=0.5 during transformer training only, inspired by Park et al. 2019.
type: method
confidence: experimental
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# spectrogram SpecAugment-style augmentation with frequency and time masking improves transformer generalization

SpecAugment (Park et al., 2019) demonstrated that directly augmenting the spectrogram input — rather than the raw waveform — can substantially improve generalization in speech recognition transformers. The technique masks contiguous bands in the frequency or time dimension, forcing the model to learn representations that are robust to partial information loss. This robustness mirrors real conditions where recordings may have frequency-dependent noise, brief interference, or microphone dropouts.

Four augmentation operations are applied independently with probability 0.5 during transformer training. Gaussian noise is added at a signal-to-noise ratio randomly drawn from 15-20 dB, simulating microphone self-noise and ambient interference. Gain perturbation uniformly shifts the spectrogram amplitude by ±3 dB, making the model invariant to recording-level differences. Frequency masking zeros out 1-2 rectangular bands of 20-30 bins each in the frequency axis. Time masking zeros out 1-2 contiguous spans each covering approximately 10% of the sequence length. Masked regions are set to zero (the post-normalization mean) to avoid introducing an artificial energy signal.

These augmentations are applied exclusively to transformer training inputs. The CNN pipeline uses [[constrained jittering generates diverse positive training examples by shifting detection boundaries within overlap constraints]] instead, which is better suited to the fixed-window CNN architecture. The two augmentation strategies are not interchangeable: jittering operates at the sample-selection level before spectrogram computation, while SpecAugment-style masking operates on the computed spectrogram tensor.

The confidence is rated "experimental" because SpecAugment benefits have been validated primarily in speech (human voice, ~1-8 kHz) rather than ultrasonic mouse vocalizations (20-120 kHz). The frequency structure and temporal statistics of mouse USVs differ from speech. Whether the same augmentation parameters transfer is an empirical question. The augmentation applies to [[bout-level spectrograms preserve inter-USV timing context for transformer training]], which carry the full temporal context needed for sequence-level classification. The masking values are set to zero specifically because [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] centers each frequency bin at zero mean, making zero the natural "uninformative" value.

---

Source: [ROADMAP](../ROADMAP.md), Phase 4

Relevant Notes:
- [[constrained jittering generates diverse positive training examples by shifting detection boundaries within overlap constraints]] -- CNN augmentation strategy; SpecAugment is the transformer counterpart
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- the input format these augmentations are applied to
- [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] -- why masked regions are set to zero (post-normalization mean)
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- Gaussian noise augmentation simulates the noisy recording environment

Topics:
- [[experimental-methods]]
- [[classification]]
