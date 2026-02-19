---
description: Each of 170 frequency bins normalized to zero mean/unit variance using training set statistics, preventing high-energy bins from dominating CNN input.
type: method
confidence: proven
topics:
  - "[[signal-processing]]"
  - "[[experimental-methods]]"
---

# per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input

Ultrasonic recordings contain systematic energy variation across the frequency axis. Low-frequency bins often carry more ambient energy than high-frequency bins, and this bias is unrelated to the presence or absence of USV calls. Without normalization, a CNN trained on raw spectrogram power would implicitly weight low-frequency evidence more heavily, not because it is more informative, but because it is more energetic by default.

The normalization formula is S_norm[f,t] = (S[f,t] - mean[f]) / (std[f] + 1e-8), applied independently for each of the 170 frequency bins spanning the 20-120 kHz range after STFT computation and frequency cropping. The epsilon term (1e-8) guards against division by zero in silent frequency bins. This per-bin approach is more precise than global normalization because it corrects for the frequency-dependent structure of the energy distribution rather than just centering the entire spectrogram.

Statistics are computed exclusively from the training split and saved to an npz file. This saved state is essential for reproducibility: the same mean and std vectors must be applied at inference time as were applied during training. Applying different statistics at inference would shift the input distribution and silently degrade model performance. See [[normalization statistics must be computed on training set only to prevent data leakage]] for the rationale behind this constraint.

The 170-bin resolution follows from [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] and the decision to crop to the 20-120 kHz USV-relevant band (see [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]]). At 586 Hz per bin, the 100 kHz band contains approximately 170 bins. This resolution is sufficient because [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]].

---

Source: [[ROADMAP.md]], Phase 2
