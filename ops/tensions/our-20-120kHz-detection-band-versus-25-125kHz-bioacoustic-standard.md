---
status: pending
created: 2026-02-24
---

# Our 20-120 kHz detection range versus the 25-125 kHz bioacoustic standard frequency band

Our detection pipeline uses 20-120 kHz (with deliberate padding at both ends to avoid clipping edge-case calls). The bioacoustic cross-tool standard (Raven Pro, DeepSqueak, BootSnap, VocalMat) uses 25,000-125,000 Hz. When exporting Raven selection tables, which band should we use?

## Conflicting Notes
- [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]] -- our padded band captures edge cases
- [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] -- cross-tool convention

## Analysis

The tension is mild because [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] — the exact band doesn't matter much for DeepSqueak import. However, the discrepancy means:
- Detections in the 20-25 kHz range (our extension) would still be exported as 25-125 kHz regions in Raven format — their time bounds are what matter
- We might detect calls DeepSqueak's own detector wouldn't, in the low-frequency tail
- Using 25-125 kHz in exports follows convention and ensures compatibility with any tool

## Possible Resolution
Use 25-125 kHz for Raven exports (follow convention), but document that our detection range is wider. The existing `raven_export.py` already uses configurable `low_freq_hz` and `high_freq_hz` parameters, so this is a configuration choice, not a code change.
