---
description: "Tab-separated .txt with Selection/View/Channel/Begin Time/End Time/Low Freq/High Freq columns, used by Raven Pro, DeepSqueak, Audacity, and others"
type: method
confidence: proven
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# Raven selection table format is the standard interchange format between bioacoustic analysis tools

The Raven selection table (.txt) is a tab-separated file with a fixed column schema that has become the de facto interchange format for passing detection results between bioacoustic tools. Raven Pro (Cornell Lab of Ornithology), DeepSqueak, Audacity label tracks, and other tools all support reading and/or writing this format. This makes it the natural bridge for feeding detections from one pipeline into the classification or visualization capabilities of another.

The mandatory columns are:

| Column | Content | Example |
|--------|---------|---------|
| Selection | 1-indexed row number | 1 |
| View | Always "Spectrogram 1" for single-view analysis | Spectrogram 1 |
| Channel | 1 for mono recordings | 1 |
| Begin Time (s) | Detection start time in seconds | 1.7006 |
| End Time (s) | Detection end time in seconds | 1.7420 |
| Low Freq (Hz) | Lower frequency bound in Hz (not kHz) | 25000 |
| High Freq (Hz) | Upper frequency bound in Hz (not kHz) | 125000 |

The standard naming convention is `{wav_stem}.Table.1.selections.txt`, tying each selection table to its source WAV file. This convention is recognized by DeepSqueak's "Import from Raven" function, which expects the original WAV files to be accessible for its own spectrogram regeneration. Since [[batch detection with skip-existing enables incremental processing of large WAV collections]], the Raven export adapter can operate on detection JSONs as they are produced, enabling incremental export that mirrors incremental detection.

---

Source:
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23)

Relevant Notes:
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- the strategy that uses Raven export as bridge
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- implication: frequency bounds need not be exact
- [[batch detection with skip-existing enables incremental processing of large WAV collections]] -- the detection pipeline whose outputs feed Raven export
- [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] -- the standard frequency band to use in the Low Freq / High Freq columns

Topics:
- [[classification-tools]]
- [[experimental-methods]]
