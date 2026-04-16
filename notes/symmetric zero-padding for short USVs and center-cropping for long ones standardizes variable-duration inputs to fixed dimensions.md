---
description: "AMVOC pads or crops to 64 time frames (128ms) chosen as larger than mean and median USV duration and a power of 2 — generalizable method for any variable-length spectrogram input"
type: method
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[unsupervised-usv-discovery]]"
---

# Symmetric zero-padding for short USVs and center-cropping for long ones standardizes variable-duration inputs to fixed dimensions

Neural networks require fixed-size inputs, but USV durations vary widely (from <5 ms to >200 ms). AMVOC solves this with a two-rule strategy applied to each extracted USV spectrogram:

**Short USVs (< 64 frames):** Zero-pad symmetrically — center the USV and pad both sides equally. This preserves the temporal position of the USV within the frame and avoids the bias of left-aligning that would make the network learn position-dependent features.

**Long USVs (> 64 frames):** Center-crop — keep the central 64 frames, discard the edges. The rationale is that the most informative portion of a USV is typically in the middle, and edge regions often contain onset/offset transients that vary more across instances of the same type.

```python
if len(spec) > 64:
    spec = spec[int((len(spec)-64)/2) : int((len(spec)-64)/2)+64, :] / np.amax(spec)
elif len(spec) < 64:
    spec = np.pad(spec/np.amax(spec), ((pad_before, pad_after), (0, 0)))
```

The target of 64 frames (128 ms at 2 ms/frame) was chosen because it exceeds both mean and median USV duration in the training data, AND is a power of 2 (convenient for MaxPool layers that halve dimensions). This means most USVs are zero-padded rather than cropped, preserving full information for the majority of inputs.

This method is directly applicable to our pipeline. Since our STFT uses different parameters (0.427 ms hop vs. AMVOC's 2 ms), our frame count for 128 ms would be ~300 frames rather than 64 — but the same symmetric-pad/center-crop principle applies. The target duration should be set based on our own USV duration distribution.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression]] — the architecture that requires this fixed-size input
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] — different windowing strategy that preserves context between USVs
- [[time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information]] -- competing approach: resampling warps temporal dynamics but preserves full call shape, vs zero-pad/crop which preserves temporal dynamics but loses edges of long calls

Topics:
- [[signal-processing]]
- [[unsupervised-usv-discovery]]
