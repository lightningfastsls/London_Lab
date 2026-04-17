---
description: "AMVOC (Giannakopoulos et al. 2022) is MIT-licensed Python/PyTorch using convolutional autoencoder for unsupervised USV clustering with both batch and real-time modes"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering

AMVOC (Giannakopoulos et al., 2022, *Bioacoustics*) at `github.com/tyiannak/amvoc` is the **best available open-source Python tool** for unsupervised USV analysis. It uses a convolutional autoencoder for feature extraction and clustering of USV spectrograms. Key properties:

- Pure Python 3.8 with PyTorch and scikit-learn
- MIT license
- Supports both offline batch processing and real-time analysis via a Dash web GUI
- Detection module outputs CSVs with onset/offset
- Clustering module processes detected USVs

AMVOC is potentially adaptable to accept externally detected segments if formatted correctly, which makes it relevant to our pipeline where USVs are already detected at F1 91.7%. Its unsupervised autoencoder approach is philosophically aligned with our VQ-VAE strategy -- both learn representations without predefined categories, since [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]]. However, AMVOC uses a standard autoencoder without the discretization step that our VQ-VAE provides.

**Architecture details** (from deep read): The autoencoder takes `(1, 64, 160)` inputs and compresses 8× to a 1,280-dimensional bottleneck via three Conv2d+MaxPool layers — see [[AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression]]. Training dataset: 22,409 syllables from 26 recordings across 9 male mice (B6D2F1/J + C57BL/6J lab strains, not wild mice). Code discrepancy: `training_task.py` uses 8 bottleneck filters (matches paper) while `conv_autoencoder.py` uses 4 (possibly for semi-supervised variant). Dependencies: PyTorch 1.7.1, scikit-learn 0.24, pyAudioAnalysis 0.3.6. Pre-trained models ship as full `torch.save(model)` objects (~121 KB), not state_dicts.

AMVOC feature mode 3 (90-dimensional resampled frequency contour, described in Stoumpou 2022) is architecturally similar to the [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space|Omer lab ridge vectorization]] (40-dimensional FM contour + 40-dimensional AM contour = 80D). The main difference is that the Omer method also extracts the amplitude trajectory along the ridge, making it a superset of AMVOC mode 3 — capturing both shape (FM) and loudness dynamics (AM). Whether the additional AM component reveals substructure that AMVOC mode 3 misses is [[whether Omer-style ridge vectorization applied to mouse USVs produces meaningfully different clustering than AMVOC autoencoder embeddings|an open empirical question]].

In the landscape of Python USV tools, AMVOC fills the unsupervised niche while [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] fills the supervised niche. The [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] strategy recommends using both approaches.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- inbox/amvoc-stoumpou-2022-deep-read-2026-04-15.md (deep read, April 2026) — full architecture + pipeline technical extraction
- Stoumpou et al. (2022), *Bioacoustics*, 32(2), 199-229
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, April 2026) — Omer vectorization comparison

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the continuum finding AMVOC's approach respects
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- supervised complement to AMVOC's unsupervised approach
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] -- similar unsupervised embedding approach across species
- [[No single Python tool cleanly accepts pre-detected USV segments and classifies them into syllable types as of 2026]] -- AMVOC partially addresses but doesn't fully solve this gap
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the strategy that combines tools like AMVOC and BootSnap
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- superset of AMVOC mode 3 (adds AM trajectory)
- [[AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression]] -- detailed architecture spec
- [[AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions]] -- post-processing: var threshold → scaler → PCA (95% var) → t-SNE (viz only)
- [[AMVOC uses 2ms non-overlapping spectrogram windows giving 0.5 kHz frequency resolution at the expense of temporal smoothness]] -- STFT parameters that feed the 64×160 input
- [[AMVOC trains for only 2 epochs deliberately because the undercomplete bottleneck acts as implicit regularizer]] -- training philosophy: lossy by design, 2 epochs suffice
- [[AMVOC deep autoencoder features scored 37 percent higher than 4-feature handcrafted baselines in blinded human evaluation]] -- quantitative evidence for learned representations
- [[AMVOC SVM-smoothed frequency contour resampled to 90 dimensions is architecturally similar to peak-frequency vectorization]] -- handcrafted baseline (feature mode 3) AMVOC compares against
- [[AMVOC semi-supervised retraining combines reconstruction KL divergence and pairwise constraint losses with uncertainty-based annotation priority]] -- extension of base autoencoder with human-in-the-loop refinement
- [[AMVOC dual-criterion dynamic spectral thresholding achieved Event F1 90.5 percent outperforming DeepSqueak and VocalMat on the same benchmark]] -- AMVOC's detection module (separate from feature extraction above)
- [[AMVOC t-SNE plus user-specified k versus field-standard UMAP plus HDBSCAN for bioacoustic clustering]] -- tension: AMVOC's pre-UMAP clustering still works but is paradigm-dated
- [[AMVOC lacks batch normalization dropout validation monitoring and VAE variant — all high-value improvements for our wild-mouse pipeline]] -- gap analysis for our design
- [[four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization]] -- AMVOC is the learned-features hypothesis (branch 3): testing whether sequential structure lives in reconstruction-relevant axes the autoencoder discovers
- [[autoencoder bottleneck plus PCA extracts concepts because reconstruction forces the model to preserve axes of variation that matter]] -- the mechanistic claim for why AMVOC's bottleneck + PCA pipeline actually discovers concepts rather than arbitrary compression
- [[low-dimensional intrinsic manifold argues for learned features rather than against them because bottleneck compression is how you find low-dim structure]] -- defends AMVOC's relevance against the inverted-evidence-chain rejection that uses our own HDBSCAN low-dim finding against autoencoder methods

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-tools]]
