---
description: "ST_STEP equals ST_WIN equals 0.002s with median filter post-processing — contrasts sharply with our 75 percent overlap 512-point FFT and DeepSqueak 960-sample overlapping windows"
type: decision
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[signal-processing]]"
---

# AMVOC uses 2ms non-overlapping spectrogram windows giving 0.5 kHz frequency resolution at the expense of temporal smoothness

AMVOC computes spectrograms using 2 ms windows with no overlap (ST_STEP = ST_WIN = 0.002 s), computed via `pyAudioAnalysis.ShortTermFeatures.spectrogram()`. The frequency resolution is f_r = 1/w = 1/0.002 = 500 Hz = 0.5 kHz, covering 30–110 kHz (80 kHz bandwidth, yielding 160 frequency bins). A median filter with kernel (2, 3) is applied to the spectrogram as post-processing via `scipy.ndimage.median_filter`.

This is a strikingly different parameter choice from the two other major tools in the field:

| Tool | Window | Hop | Overlap | Freq Res | Time Res |
|------|--------|-----|---------|----------|----------|
| AMVOC | 2 ms | 2 ms | 0% | 0.5 kHz | 2.0 ms |
| Our pipeline | 1.7 ms (512@300kHz) | 0.43 ms (128@300kHz) | 75% | 586 Hz | 0.43 ms |
| DeepSqueak | 3.2 ms (960@300kHz) | 0.4 ms (overlap) | ~88% | 312 Hz | 0.4 ms |

AMVOC's zero-overlap choice means each time frame is independent — there's no spectral leakage between adjacent frames, and computation is ~4× faster than 75% overlap. However, it produces a temporally "blocky" spectrogram where rapid frequency transitions within 2 ms are averaged into a single frame. The median filter partially compensates by smoothing across the time-frequency grid.

Since [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]], AMVOC's trade-off (coarser time, reasonable frequency) may be acceptable for classification purposes even if it would be problematic for detection or precise temporal boundary estimation. For our autoencoder input, the choice of STFT parameters affects what features the network can learn — finer temporal resolution means more frames per USV, requiring either larger network inputs or coarser frequency bins.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] — our STFT parameters for comparison
- [[75 percent overlap with hop length 128 provides smooth temporal coverage for USV detection]] — why we chose high overlap
- [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]] — DeepSqueak's parameters for comparison
- [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]] — evidence supporting AMVOC's trade-off direction
- [[MUPET 2 ms frame duration with 80 percent overlap prioritizes temporal resolution for capturing rapid USV frequency modulations]] — convergent 2 ms window duration with AMVOC but opposite overlap strategy (80% vs 0%); MUPET gets temporal smoothness from overlap while AMVOC relies on post-hoc median filtering

Topics:
- [[signal-processing]]
