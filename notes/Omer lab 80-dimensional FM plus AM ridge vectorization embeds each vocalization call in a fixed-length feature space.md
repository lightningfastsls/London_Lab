---
description: "Oren et al. 2024 extract FM ridge (40D) + AM along ridge (40D) = 80D fixed-length vector per call, enabling direct classification without variable-length handling"
type: method
confidence: proven
conditions:
  - implemented in MATLAB spectrogramming.m on Zenodo
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space

Oren et al. (2024, Science) developed a vectorization technique that represents each vocalization as an 80-dimensional feature vector by extracting two parallel trajectories from the spectrogram:

1. **FM trajectory (40D):** The dominant frequency at each of 40 resampled time steps, extracted via ridge detection (`tfridge` in MATLAB). This captures the pitch contour / frequency modulation of the call.
2. **AM trajectory (40D):** The spectrogram amplitude at the ridge location for each time step. This captures the amplitude envelope along the pitch contour.

The final vector is `[AM_1...AM_40, FM_1...FM_40]` = 80 dimensions.

The pipeline is: STFT (Hanning window, 50% overlap) -> frequency band crop (6-9 kHz for marmosets) -> 2D interpolation to 40 time columns -> ridge extraction (argmax per column with continuity constraints) -> amplitude extraction along ridge -> smoothing (median window=6 for AM, mean window=5 for FM) -> per-caller normalization to [0,1].

This technique is a **superset** of [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering|AMVOC feature mode 3]] (90D resampled frequency contour), since it additionally captures the amplitude trajectory. The key innovation over simple pitch contour extraction is the joint AM+FM representation: both shape and loudness dynamics are encoded.

For mouse USV adaptation, the frequency band changes to ~30-110 kHz, the window shrinks to ~1-5 ms (USVs are 10-100 ms, much shorter than phee calls), and `tfridge` is replaced by Python argmax or `scipy.signal` peak finding per column. See [[time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information]] for the normalization step.

The source code is available at Zenodo (CC-BY 4.0): [[Oren 2024 Zenodo repository provides complete MATLAB implementation of spectrogram ridge vectorization for adaptation]].

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Oren, G. et al. (2024). Science, 385(6712), 996-1003. DOI: 10.1126/science.adp3757

Relevant Notes:
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- AMVOC mode 3 is architecturally similar but FM-only (no AM component)
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- continuum finding means fixed-length vectorization must preserve continuous variation
- [[DeepSqueak Excel export provides 16 per-call metrics including principal frequency bandwidth slope and tonality]] -- alternative feature set (bounding-box statistics vs ridge trajectories)
- [[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory]] -- the core algorithmic step
- [[time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information]] -- the normalization step
- [[iMUPET adapted for Hertz 2020 uses 16 gammatone filters producing 2016-dimensional feature vectors per syllable]] -- alternative handcrafted vectorization (2016D gammatone vs 80D ridge); complementary feature philosophies
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- SIS could benchmark Omer-derived labels against iMUPET (0.13 bits) and iMSA (0.22 bits)
- [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]] -- Omer vectorization is a mid-complexity third option for resolving this open question
- [[AMVOC SVM-smoothed frequency contour resampled to 90 dimensions is architecturally similar to peak-frequency vectorization]] -- AMVOC mode 3 (90D resampled FM contour with SVM smoothing) is architecturally similar to Omer's FM component (40D); the key difference is Omer's additional 40D AM trajectory, making it a strict superset that captures both shape and loudness dynamics

Topics:
- [[unsupervised-usv-discovery]]
- [[classification]]
