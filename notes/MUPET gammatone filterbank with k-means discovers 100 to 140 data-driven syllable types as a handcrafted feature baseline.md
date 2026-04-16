---
description: "64-channel Gammatone filterbank features with cosine-distance k-means — learned embeddings have since surpassed this baseline but it established data-driven syllable discovery"
type: baseline
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
  - "[[signal-processing]]"
---

# MUPET gammatone filterbank with k-means discovers 100 to 140 data-driven syllable types as a handcrafted feature baseline

MUPET (Mouse Ultrasonic Profile ExTraction) uses a 64-channel Gammatone filterbank to convert spectrograms into compact "GF-USV" representations, then applies unsupervised k-means clustering with cosine distance to discover 100-140 data-driven syllable types. This handcrafted-feature approach was an important historical baseline that learned embeddings have since surpassed.

The Gammatone filterbank is biologically inspired, mimicking the auditory periphery with filter bandwidths varying from 0.5 to 4 kHz and narrower filters concentrated in frequency regions containing the most acoustic events. This is a different philosophy from our direct STFT energy detection — MUPET transforms the spectrogram through an auditory model before analysis, while we operate on raw STFT output. Since [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]], the auditory-model-based representation may capture perceptually relevant features that raw STFTs miss.

The 100-140 syllable types discovered by k-means contrasts sharply with Goffinet's finding that since [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]], suggesting k-means may be imposing discrete structure that doesn't exist in the data.

⚠️ **MUPET vs iMUPET:** Hertz 2020 adapted MUPET as "iMUPET" using only **16 gammatone filters** (not 64) constrained to K=8. See [[iMUPET adapted for Hertz 2020 uses 16 gammatone filters producing 2016-dimensional feature vectors per syllable]].

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/), mupet-sample-rate-usv-analysis-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- gammatone as alternative spectrogram representation
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] -- k-means 100+ types vs GMM 2 types is a major tension
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- k-means forces discreteness onto a continuum
- [[iMUPET adapted for Hertz 2020 uses 16 gammatone filters producing 2016-dimensional feature vectors per syllable]] -- adapted version with fewer filters and K=8

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
- [[signal-processing]]
