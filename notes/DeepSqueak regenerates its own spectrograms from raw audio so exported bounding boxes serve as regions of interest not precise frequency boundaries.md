---
description: "DeepSqueak re-computes spectrograms from the WAV file during import, so Raven export frequency bounds only need to be approximate regions of interest"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries

When importing detections via Raven selection tables, DeepSqueak does not use the exported spectrogram data or rely on the exact frequency boundaries provided. Instead, it requires access to the original WAV files and regenerates its own spectrograms using its internal STFT parameters. The bounding boxes from the Raven table serve only as regions of interest — pointers to where in time and frequency space to look.

This has a practical implication for our Raven export adapter: the Low Freq and High Freq columns do not need to match our detection pipeline's precise frequency estimates for each candidate. Using a fixed band (e.g., 25,000–125,000 Hz for the standard mouse USV range) is sufficient because DeepSqueak will recompute everything from the raw audio anyway. What matters is accurate time boundaries (Begin Time, End Time), since those determine which portion of the recording DeepSqueak will analyze. This is why [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] can be used as a fixed band in the export, rather than computing per-detection frequency bounds.

---

Source:
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23)

Relevant Notes:
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the format through which these bounding boxes are communicated
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- the strategy this enables
- [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]] -- DeepSqueak's own STFT parameters that it uses when regenerating spectrograms
- [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] -- the standard fixed band that suffices for export given that exact bounds are not needed
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- the re-association strategy needed because DeepSqueak's regenerated timestamps will not match exactly

Topics:
- [[classification-tools]]
- [[experimental-methods]]
