---
description: "time-based energy threshold combined with frequency-based peak-to-mean ratio both must pass — highest event detection among 6 tools tested on 245 syllables from 14 mice at 21.2x real-time"
type: finding
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[detection-landscape]]"
---

# AMVOC dual-criterion dynamic spectral thresholding achieved Event F1 90.5 percent outperforming DeepSqueak and VocalMat on the same benchmark

AMVOC's detection method (Stoumpou et al. 2022) uses a dual-criterion approach on 2 ms frames in the 30–110 kHz range. Both criteria must pass for a frame to be classified as vocalization:

**Criterion 1 — Time-based Thresholding (TT):** Spectral energy S_i (sum across 30–110 kHz) must exceed a dynamic threshold T_i = t × (0.5 × global_mean + 0.5 × moving_average_of_last_100_frames), with t = 0.5. The blended threshold adapts to local noise conditions via the moving average.

**Criterion 2 — Frequency-based Thresholding (FT):** Peak energy P_i must exceed f × mean energy M_i in a 60 kHz window around the peak frequency, with f = 3.5. This ensures the USV has a clear spectral peak rather than broadband energy.

**Post-processing:** 20 ms box filter (L=10 frames) for smoothing the binary decision sequence, concatenate segments separated by <11 ms, remove detections shorter than 5 ms.

Benchmark results on Dataset D1 (245 syllables, 14 mice):

| Tool | Event F1 | Temporal F1 | Speed |
|------|----------|-------------|-------|
| AMVOC offline | **90.5%** | 75.5% | 21.2× RT |
| DeepSqueak | 87.0% | 79.5% | 8.2× RT |
| MSA2 | 83.0% | 79.5% | — |
| VocalMat | 74.0% | 74.5% | 4.3× RT |
| MUPET | 75.0% | 69.0% | 32.4× RT |

AMVOC had the highest event F1 but lower temporal F1 than DeepSqueak and MSA2, suggesting it finds more events but with less precise boundaries. The dual-criterion approach — requiring BOTH time and frequency evidence — reduces false positives from broadband noise transients that would pass a single energy threshold.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] — architectural contrast in detection approach
- [[A-MUD classical signal processing detector outperforms USVSEG and MUPET in true positive rate for USV detection]] — another classical detector benchmark
- [[entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative]] — yet another classical approach with different trade-offs
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] — direct architectural contrast: AMVOC's single-stage dual-criterion achieves highest event F1 (90.5%) but lower temporal F1 than DeepSqueak; our two-stage pipeline decouples recall (energy detector) from precision (CNN), enabling independent tuning of each stage
- [[U-Net semantic segmentation exceeded 95 percent precision recall for USV detection in systematic DL comparison]] — DL-based detection ceiling is higher than AMVOC's classical ceiling: Ivanenko 2023 U-Net exceeded 95% precision/recall on the same task AMVOC achieved 90.5% event F1, suggesting semantic segmentation is the modern successor for high-precision detection
- [[SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines]] — another DL-based detection approach whose 90.2 Dice is comparable to AMVOC's 90.5 event F1 but at pixel-level resolution rather than event level, offering tighter temporal boundaries

Topics:
- [[detection-landscape]]
