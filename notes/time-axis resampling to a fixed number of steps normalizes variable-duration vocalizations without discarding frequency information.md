---
description: "2D interpolation resamples the time axis to a fixed number of columns (e.g. 40) while preserving frequency resolution — solves variable-length call problem for vectorization"
type: method
confidence: proven
conditions:
  - Oren 2024 uses 40 steps; optimal count for mouse USVs is TBD
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[signal-processing]]"
  - "[[classification]]"
---

# time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information

Oren et al. (2024) solve the variable-duration problem by resampling the spectrogram time axis to exactly 40 columns via 2D interpolation, before ridge extraction. The frequency axis is preserved at native STFT resolution; only the time axis is normalized. This means a 50 ms call and a 500 ms call both become 40-column spectrograms, with each column representing a proportional time slice of the original call.

The MATLAB implementation uses `interp2` (bilinear interpolation):

```matlab
spect_dim = 40;
[x1, y1] = meshgrid(1:size(tf,2), 1:size(tf,1));
[x2, y2] = meshgrid(linspace(1, size(tf,2), spect_dim), 1:size(tf,1));
tf = interp2(x1, y1, tf, x2, y2);
```

This contrasts with other approaches to variable-length normalization:
- **Zero-padding / center-cropping** (AMVOC approach): pads short calls and crops long ones to a fixed width. Loses information at edges of long calls, adds zeros to short calls.
- **Temporal pooling** (our CNN approach): global average pooling over the time dimension collapses all temporal structure.
- **Duration-normalized time resampling** (this approach): every call is "stretched" or "compressed" to the same number of steps. Preserves the full temporal shape at the cost of losing absolute duration information.

The loss of absolute duration is acceptable because duration can be stored as a separate scalar feature if needed. The advantage is that the FM and AM trajectories extracted after resampling are always exactly n_steps long, enabling direct concatenation into a fixed-length vector for classification.

**For mouse USV type classification, duration is not optional — it is primary.** Scattoni's taxonomy distinguishes Short calls from longer Chevron and Complex types using absolute duration. Hertz 2020's ISI analysis treats duration as one of the primary distinguishing features driving iMSA's median-duration split (Simple-long / Simple-short). Time-resampling discards this information by construction. **Resolution:** After the 2 × n_steps trajectory features, append the original call duration as an explicit scalar — making the full vector 2×n_steps + 1 (e.g., 81 or 121 dimensions). This lets the clustering discover whether duration-based structure matters without forcing the choice upstream. The cost of omitting it is losing a primary axis of the Scattoni taxonomy.

For mouse USVs (typically 10-100 ms), the target step count may need adjustment from 40. Very short calls (10 ms) at our STFT parameters (hop ~0.4 ms) produce only ~25 time columns, so 40 steps would involve upsampling. A sweep over [20, 30, 40, 50] steps would determine the optimal resolution for the USV duration distribution.

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Oren, G. et al. (2024). Science, 385(6712), 996-1003.

Relevant Notes:
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- the full pipeline this step enables
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- alternative approach that preserves absolute timing between calls
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- uses zero-pad/center-crop instead of resampling
- [[symmetric zero-padding for short USVs and center-cropping for long ones standardizes variable-duration inputs to fixed dimensions]] -- the competing approach: resampling warps temporal dynamics but preserves full call shape, while zero-pad/crop preserves temporal dynamics but loses edges of long calls; AMVOC chose pad/crop targeting 64 frames (128 ms), which is a power-of-2 constraint from MaxPool layers

Topics:
- [[signal-processing]]
- [[classification]]
