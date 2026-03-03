---
description: "Per-channel energy normalization adapts to local noise levels, giving the DCASE 2022 winning system an edge -- particularly important when support and query recordings have different noise profiles"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[classification]]"
---

# PCEN normalization is more robust than log-mel spectrograms for few-shot bioacoustic scenarios

Per-Channel Energy Normalization (PCEN) was a key component of the Surrey team's first-place DCASE 2022 few-shot bioacoustic detection system. Unlike log-mel spectrograms which apply a fixed logarithmic nonlinearity, PCEN adapts to local noise levels through automatic gain control, making it robust to varying recording conditions. This adaptive property matters critically for few-shot scenarios because the 5 support examples may come from different acoustic environments than the test recording.

PCEN operates by applying a frequency-dependent automatic gain control that normalizes each frequency channel based on its recent energy history. The result is that quiet sounds in noisy environments get boosted while loud transients get suppressed -- effectively performing a learned, per-channel noise reduction. Combined with delta MFCCs as input features and better utilization of negative data in the loss function, PCEN delivered the best DCASE 2022 performance.

For USV research, PCEN could improve generalization across recording sessions and cage configurations. Our current pipeline uses a fixed energy threshold at -60 dB, which works within a session but requires recalibration between sessions with different noise floors. PCEN would instead adapt continuously, potentially eliminating the per-session threshold tuning problem. However, there is a tradeoff: PCEN's temporal smoothing operates over a window that must be tuned for the signal characteristics. USVs are short (typically 5-100 ms), so the smoothing window would need to be shorter than typical speech applications. Whether PCEN's adaptive normalization provides sufficient benefit over our current per-recording normalization to justify the added complexity remains an empirical question.

---

Source:
- few-shot-learning-animal-sound-classification-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- our current approach uses fixed thresholds rather than adaptive normalization

Topics:
- [[signal-processing]]
- [[classification-methodology]]
