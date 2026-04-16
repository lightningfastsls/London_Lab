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

Oren et al. (2024) independently define a parallel set of 16 acoustic features (8 FM + 8 AM) for marmoset phee calls ([[Oren 2024 16 acoustic features for marmoset calls parallel DeepSqueak 16 Excel export metrics for rodent USVs]]). The feature categories are similar (freq_diff, freq_max, freq_mean vs principal frequency, slope, tonality) but the derivations differ: Oren's features come from ridge-extracted FM/AM trajectories (temporal dynamics), while DeepSqueak's come from bounding box statistics (summary measures). The convergence on exactly 16 features across species and tools supports this granularity as a natural representation level for single-call acoustic characterization.

---

Source:
- Compass synthesis: inbox/deepsqueak-usv-syllable-classification-practical-guide.md
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, April 2026) — parallel 16-feature finding

Relevant Notes:
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- alternative export format focused on time-frequency boxes
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- what the bounding boxes actually mean
- [[Reading DeepSqueak mat outputs in Python uses scipy loadmat for v5 format or h5py for v7.3 HDF5 format]] -- the .mat format is the native output; Excel export is the structured alternative
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- strategic context for why we use these exports
- [[Oren 2024 16 acoustic features for marmoset calls parallel DeepSqueak 16 Excel export metrics for rodent USVs]] -- cross-species convergence on 16-feature granularity
- [[AMVOC deep autoencoder features scored 37 percent higher than 4-feature handcrafted baselines in blinded human evaluation]] -- AMVOC's baseline was only 4 features; our 16 DeepSqueak metrics are a 4x richer starting point, so the 37% improvement from learned representations would likely be smaller over our feature set

Topics:
- [[classification-tools]]
