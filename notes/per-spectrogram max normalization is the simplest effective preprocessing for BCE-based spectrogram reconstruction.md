---
description: "divide each spectrogram by its own maximum value — discards absolute amplitude information but satisfies BCE 0-1 input requirement with no fitted parameters"
type: method
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[signal-processing]]"
---

# Per-spectrogram max normalization is the simplest effective preprocessing for BCE-based spectrogram reconstruction

AMVOC normalizes each extracted USV spectrogram by dividing by its own maximum value: `spec = spec / np.amax(spec)`. This is the simplest possible normalization that satisfies the [0, 1] input constraint for BCE loss. It is explicitly NOT min-max normalization (which would also subtract the minimum), NOT z-score normalization (which centers and scales by standard deviation), and NOT log-scale transformation.

The per-spectrogram nature is significant: each USV is normalized independently, so absolute amplitude information is deliberately discarded. A quiet USV and a loud USV with identical spectral shape will produce identical normalized inputs. This makes sense for classification (shape matters more than volume for USV type discrimination) but loses information that might be relevant for other analyses, such as arousal-level encoding in amplitude.

The simplicity has practical advantages: no fitted parameters (unlike StandardScaler, which needs training-set statistics), no risk of division-by-zero edge cases from min-max normalization on near-constant spectrograms, and trivial to implement. The main limitation is sensitivity to outlier pixels — a single bright noise pixel inflates the max and compresses the rest of the spectrogram's dynamic range. AMVOC mitigates this somewhat with a median filter (kernel 2×3) applied to the spectrogram before normalization.

For our pipeline, per-max normalization is a reasonable starting point if we adopt BCE loss, since [[BCE loss with sigmoid output treats spectrogram pixels as independent probabilities requiring input normalization to 0-1 range]]. If we use MSE or perceptual loss instead, different normalization schemes become preferable — MSE works fine with log-scale or z-score inputs.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[BCE loss with sigmoid output treats spectrogram pixels as independent probabilities requiring input normalization to 0-1 range]] — the loss function that requires this normalization
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] — our STFT parameters that would produce the spectrograms to be normalized
- [[per-caller normalization of AM and FM features to 0-1 prevents individual acoustic idiosyncrasies from dominating classification]] — contrasting normalization scope: per-spectrogram (AMVOC, discards absolute amplitude) vs per-caller (Oren, preserves relative variation within each individual); both normalize to [0,1] but for different reasons

Topics:
- [[signal-processing]]
