---
description: "DeepSqueak specifies FFT parameters in duration (3.2 ms window, 2.8 ms overlap) not samples, yielding 960/840 samples at 300 kHz and accepting any sample rate"
type: finding
confidence: proven
conditions:
  - "sample_rate: 300000"
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[classification]]"
---

# DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz

DeepSqueak specifies its internal spectrogram parameters in **seconds rather than sample counts**: 3.2 ms FFT windows with 2.8 ms overlap. This design means it accepts any sample rate -- "spectrograms are created using FFT windows of constant duration, rather than constant sample numbers, so other sample rates are accepted." **A 300 kHz WAV file will work** without modification.

At 300 kHz, these parameters translate to:
- FFT window: 3.2 ms x 300,000 = **960 samples** (~1024 padded to next power of 2)
- Overlap: 2.8 ms x 300,000 = **840 samples**
- Hop size: (3.2 - 2.8) ms x 300,000 = **120 samples** (0.4 ms)
- Frequency resolution: 300,000 / 960 = **312.5 Hz/bin** (or ~293 Hz with zero-padding to 1024)

Compare with our detection pipeline where [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- DeepSqueak uses nearly twice the FFT length, trading temporal resolution for finer frequency resolution. This is consistent with [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]], which suggests DeepSqueak's choice of longer FFT windows is well-suited for classification tasks even if our shorter windows are better for detection.

DeepSqueak accepts **WAV, FLAC, and Ultravox (.UVD)** files and processes only the first channel of multichannel recordings.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- DeepSqueak README and documentation

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- our STFT parameters for comparison
- [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]] -- justifies DeepSqueak's longer FFT windows
- [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] -- our coarser resolution vs DeepSqueak's ~312 Hz
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- sample rate context for the conversion
- [[chevron calls expose the STFT time-frequency tradeoff because they require simultaneous temporal and spectral precision]] -- the tradeoff DeepSqueak resolves toward frequency
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- these are the STFT parameters DeepSqueak applies when regenerating spectrograms from imported Raven selection tables
- [[AMVOC uses 2ms non-overlapping spectrogram windows giving 0.5 kHz frequency resolution at the expense of temporal smoothness]] -- three-way STFT parameter comparison: DeepSqueak 3.2 ms/88% overlap prioritizes frequency resolution (312 Hz), AMVOC 2 ms/0% overlap prioritizes speed and independence, our pipeline 1.7 ms/75% overlap balances temporal precision with smoothness

Topics:
- [[signal-processing]]
- [[classification-tools]]
