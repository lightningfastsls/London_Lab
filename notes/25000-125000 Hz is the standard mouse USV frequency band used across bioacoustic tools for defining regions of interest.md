---
description: "Cross-tool convention for mouse USV frequency bounds, slightly different from our 20-120 kHz detection band"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[classification]]"
  - "[[signal-processing]]"
---

# 25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest

The bioacoustic tool ecosystem (Raven Pro, DeepSqueak, VocalMat, BootSnap) converges on 25,000–125,000 Hz as the standard frequency band for mouse USV regions of interest. This band encompasses the full range of documented mouse ultrasonic vocalizations while excluding most audible-range noise below 25 kHz.

This differs from our detection pipeline's band of 20–120 kHz, which uses deliberate padding at both ends to avoid clipping edge-case calls (see [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]]). The discrepancy is minor but worth noting: our detection is slightly wider on the low end (20 vs 25 kHz) and narrower on the high end (120 vs 125 kHz). When exporting to Raven format for DeepSqueak, using the standard 25–125 kHz band follows cross-tool convention. Since [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]], the exact band choice is not critical — but consistency with cross-tool conventions is preferred.

---

Source:
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23)

Relevant Notes:
- [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]] -- our slightly different band choice
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- why exact bounds matter less for export
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the format where this band is specified

Topics:
- [[classification-tools]]
- [[signal-processing]]
