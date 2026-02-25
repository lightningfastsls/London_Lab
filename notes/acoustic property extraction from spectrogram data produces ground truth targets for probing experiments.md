---
description: "Seven properties extracted per frame — peak frequency, spectral centroid, energy, is_voiced, frequency direction, bout position, time since last USV"
type: method
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[signal-processing]]"
---

# acoustic property extraction from spectrogram data produces ground truth targets for probing experiments

Probing experiments require ground-truth labels that the probes attempt to predict from frozen transformer hidden states. These labels must be computed directly from the spectrogram input, because they serve as the "answers" that test whether the transformer has learned to encode specific acoustic properties at each layer. Seven properties are extracted per spectrogram frame, spanning a mix of regression (continuous) and classification (categorical) targets that together cover the essential acoustic dimensions of USV signals.

The first three properties are continuous and computed from individual spectrogram columns (170 frequency bins per ADR-002, spanning the 20-120 kHz detection range per ADR-001). **Peak frequency** is the frequency corresponding to the bin with maximum energy, converted to Hz. **Spectral centroid** is the energy-weighted mean frequency across all bins — it captures where the spectral mass is concentrated, which often differs from the peak when harmonics or noise are present. **Energy** is the total energy summed across all frequency bins in the column, providing an absolute loudness measure for each frame.

The next property is categorical: **is_voiced** is a binary label indicating whether a USV is present in that frame. This is derived from the detection pipeline output and serves as the most basic classification target — if the transformer cannot distinguish voiced from unvoiced frames, it has not learned the most fundamental acoustic distinction.

The remaining three properties capture temporal context. **Frequency direction** classifies each frame as rising, falling, or flat by comparing the peak frequency to the preceding frame — this tests whether the transformer encodes local frequency modulation, which is central to USV type discrimination. **Bout position** is normalized to 0-1 within each bout, where 0 is the bout onset and 1 is the bout offset — this tests whether the transformer has learned where it is within a vocalization sequence. **Time since last USV** measures the interval in milliseconds to the nearest preceding USV onset, capturing inter-event temporal context.

These seven properties are chosen because they span the space from purely spectral (peak frequency, centroid) to purely temporal (bout position, time since last USV), with mixed spectral-temporal properties in between (frequency direction). Therefore, the probing results can reveal not just whether the transformer encodes acoustic information, but which dimensions of that information are represented at which layers.

---

Source:
- [[vacation-master-plan-v2]]

Relevant Notes:
- [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] -- these seven properties are the targets that probes attempt to predict
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- the STFT parameters that determine the spectrogram columns from which properties are extracted

Topics:
- [[representation-learning]]
- [[signal-processing]]
