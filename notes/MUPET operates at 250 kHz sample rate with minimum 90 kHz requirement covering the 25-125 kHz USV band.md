---
description: "250 kHz gives Nyquist at 125 kHz; minimum 90 kHz ensures coverage of 25-125 kHz USV range; 8th-order Chebyshev bandpass at 25 kHz isolates ultrasonic content"
type: baseline
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[signal-processing]]"
---

# MUPET operates at 250 kHz sample rate with minimum 90 kHz requirement covering the 25-125 kHz USV band

MUPET (Van Segbroeck et al. 2017) uses a 250 kHz sampling rate, giving a Nyquist frequency of 125 kHz — sufficient to cover the 25-125 kHz mouse USV range. The tool's minimum requirement is 90 kHz, which provides a Nyquist of 45 kHz — technically insufficient for the full USV band, suggesting this minimum is for low-frequency-only analysis or partial band coverage.

An 8th-order Chebyshev filter with a 25 kHz corner frequency isolates the ultrasonic content, effectively bandpassing the signal to remove low-frequency environmental noise. This is a more aggressive filtering approach than our pipeline, which uses a detection range of 20-120 kHz — since [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]], our lower bound (20 kHz) captures 5 kHz more than MUPET's filter.

Our 300 kHz sample rate provides Nyquist at 150 kHz — 25 kHz more headroom than MUPET's 125 kHz. Since [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]], the extra headroom accommodates occasional high-frequency events and prevents aliasing artifacts near the band edge. MUPET's 250 kHz represents a more conservative but widely accepted choice.

---

Source: mupet-sample-rate-usv-analysis-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- our higher sample rate provides 25 kHz more headroom
- [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]] -- our lower bound 20 kHz vs MUPET's 25 kHz
- [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] -- MUPET uses this standard band

Topics:
- [[signal-processing]]
