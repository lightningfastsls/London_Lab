---
description: "Oren 2024 rescales AM and FM independently to [0,1] per caller — prevents one loud/high-pitched individual from dominating feature space, but requires caller identity at inference"
type: method
confidence: proven
conditions:
  - per-caller normalization requires knowing caller identity; per-recording or per-session are alternatives for mouse USVs
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[classification]]"
---

# per-caller normalization of AM and FM features to 0-1 prevents individual acoustic idiosyncrasies from dominating classification

In the Oren et al. (2024) preprocessing pipeline, AM and FM features are independently rescaled to [0,1] **per caller** before classification:

```matlab
for i = 1:length(caller_id)
    idx = find(caller == caller_id(i));
    p(1:40, idx)  = rescale(p(1:40, idx),  0, 1);  % AM per caller
    p(41:end, idx) = rescale(p(41:end, idx), 0, 1); % FM per caller
end
```

This design choice means each monkey's calls are normalized to their own vocal range, so a consistently loud or high-pitched caller doesn't dominate the feature space. The classifier then operates on **relative** modulation patterns within each caller's range.

The normalization scope is a design decision with direct implications for mouse USV adaptation:

| Scope | Effect | When to use |
|-------|--------|-------------|
| **Per-caller** (Oren) | Removes individual baseline differences | When classifying within-caller targets (e.g., receiver identity) |
| **Per-recording** | Removes session-level variation (mic placement, room) | When classifying across recordings from the same animal |
| **Per-animal** | Removes individual differences | When comparing repertoire structure across animals |
| **Global** | Preserves all variation | When individual acoustic properties are the classification target |

For our wild mouse USV analysis, **per-recording** normalization is the natural analog — each WAV file represents one recording session, and mic placement or distance may vary between sessions. Per-animal normalization is also viable since our dyad design means each animal's calls come from multiple sessions.

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Oren, G. et al. (2024). Science, 385(6712), 996-1003.

Relevant Notes:
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- the full pipeline this normalization feeds into
- [[normalization statistics must be computed on training set only to prevent data leakage]] -- normalization scope and data leakage
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- our recording setup determines baseline acoustic properties
- [[per-spectrogram max normalization is the simplest effective preprocessing for BCE-based spectrogram reconstruction]] -- contrasting normalization scope: per-spectrogram (AMVOC) discards absolute amplitude entirely, while per-caller preserves relative variation across calls within one individual
- [[AMVOC SVM-smoothed frequency contour resampled to 90 dimensions is architecturally similar to peak-frequency vectorization]] -- AMVOC's frequency contour extraction does not normalize per-caller, making scope a confound when comparing AMVOC mode 3 against Omer ridge features

Topics:
- [[classification]]
