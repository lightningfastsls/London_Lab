---
description: "iMUPET (Hertz 2020) uses 16 gammatone filters (not 64 as in full MUPET) yielding a 2016-dim vector; K-means with cosine distance, K=8, training on 5000 syllables"
type: method
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[classification-tools]]"
---

# iMUPET adapted for Hertz 2020 uses 16 gammatone filters producing 2016-dimensional feature vectors per syllable

Hertz et al. (2020) adapted MUPET as "iMUPET" for consistency with their evaluation framework. The adapted version uses:

- **16 gammatone filters** (not the 64 channels in full MUPET)
- Each syllable → vector of **length 2016** via the gammatone pipeline
- K-means clustering with **cosine distance**
- Training on **5000 syllables** subset
- **K = 8** clusters (inherited and held constant for cross-algorithm comparison)
- Remaining syllables assigned to nearest centroid

**Important distinction from original MUPET:** The full MUPET tool (Van Segbroeck et al. 2017) uses 64-channel gammatone filterbank and discovers 100-140 types by default. The iMUPET adaptation used for Hertz's comparison uses only 16 filters and is constrained to K=8 for comparability. The two notes in this vault about "MUPET" describe full MUPET, not iMUPET.

**Where 2016 comes from:** 16 filters × some temporal dimension (spectrogram frames per syllable). The exact construction references Van Segbroeck et al. (2017, MUPET paper). The gammatone filter details are noted with uncertainty in the source.

**Performance:** iMUPET achieves ~0.13 bits depth-1 SIS and produces the most uniform label distribution (H₀ ≈ 2.90 bits, close to max entropy log2(8) = 3.0) — reflecting that k-means with acoustic features distributes syllables evenly across 8 clusters.

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

⚠️ Uncertainty: The exact gammatone filter construction producing the 2016-dim vector references Van Segbroeck et al. 2017 (MUPET paper). The 16-filter count is from Hertz 2020 Methods; the full MUPET uses 64 channels. Verify in MUPET paper if precision is needed.

Relevant Notes:
- [[MUPET gammatone filterbank with k-means discovers 100 to 140 data-driven syllable types as a handcrafted feature baseline]] -- full MUPET (64-channel, K=100-140); this note describes iMUPET (16-channel, K=8)
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- also describes full MUPET
- [[Syntax Information Maximization SIM algorithm iteratively perturbs cluster centroids to maximize SIS on training sequences]] -- SIM starts from these iMUPET centroids
- [[Hertz 2020 quantitative benchmark iMSA achieves 0.22 bits depth-1 SIS versus iMUPET 0.13 and iVoICE 0.10 on C57BL-6 courtship data]] -- iMUPET's SIS performance
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- convergent evidence for gammatone usefulness
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- alternative handcrafted vectorization: 80D ridge-based (FM+AM) vs 2016D gammatone filterbank; complementary feature philosophies with known SIS for iMUPET (0.13 bits)

Topics:
- [[classification-methodology]]
- [[classification-tools]]
