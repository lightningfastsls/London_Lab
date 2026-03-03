---
description: "DeepSqueak's Excel export gives 16 acoustic metrics per detected call — a rich feature set for downstream repertoire analysis without custom extraction code"
type: finding
confidence: proven
conditions:
  - via File → Export to Excel Log in DeepSqueak GUI
meta_state: current
source: "inbox/deepsqueak-usv-syllable-classification-practical-guide.md"
topics:
  - "[[classification]]"
---

# DeepSqueak Excel export provides 16 per-call metrics including principal frequency bandwidth slope and tonality

DeepSqueak's Excel export (`File → Export to Excel Log`) produces .xlsx files with 16 metrics per detected call: ID, label, begin time, end time, call length, principal frequency, low frequency, high frequency, bandwidth, frequency standard deviation, slope, sinuosity, mean power, tonality, and peak frequency.

This is the richest structured output DeepSqueak provides. The primary output format is MATLAB .mat files containing per-call structures with bounding box position `[Begin Time, Min Frequency, Duration, Frequency Range]`, detection confidence, raw audio snippets, accept/reject status, classification labels, and power. Additional exports include Raven selection tables and spectrogram images. There is no native CSV or SQLite output, though the .xlsx export is functionally equivalent for analysis workflows.

For our pipeline, the Excel export provides acoustic features we don't need to re-extract — particularly sinuosity (a measure of frequency modulation complexity) and tonality (signal-to-noise purity), which are expensive to compute from scratch.

---

Source:
- Compass synthesis: inbox/deepsqueak-usv-syllable-classification-practical-guide.md

Relevant Notes:
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- alternative export format focused on time-frequency boxes
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- what the bounding boxes actually mean
- [[Reading DeepSqueak mat outputs in Python uses scipy loadmat for v5 format or h5py for v7.3 HDF5 format]] -- the .mat format is the native output; Excel export is the structured alternative
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- strategic context for why we use these exports

Topics:
- [[classification-tools]]
