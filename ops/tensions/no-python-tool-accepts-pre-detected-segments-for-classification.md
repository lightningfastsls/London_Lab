---
status: pending
created: 2026-02-23
---

# No Python USV tool cleanly accepts pre-detected segments for classification creating an integration gap

Our pipeline detects USVs at F1 91.7% using a two-stage energy detector + CNN approach. The natural next step is syllable classification -- but **no existing Python USV tool cleanly accepts pre-detected segments as input for classification**. This creates an integration gap between our detection pipeline and the available classification tools.

The landscape:
- **BootSnap**: Designed for pre-detected USVs, but [[whether BootSnap code is publicly available or must be requested from Abbasi Zala Penn at Vienna]] is unresolved
- **AMVOC**: Can potentially accept external detections if formatted correctly, but was designed with its own detection module -- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]]
- **DAS**: Requires raw audio input -- [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]]
- **WhisperSeg**: Also raw audio only -- [[WhisperSeg adapts OpenAI Whisper transformer for animal vocalization segmentation with positive cross-species transfer]]
- **DeepSqueak**: MATLAB only, and even its Raven import pathway requires audio access -- not pure classification
- **VocalMat**: MATLAB only, inactive since ~2021

## Conflicting Notes
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- we have good detection that needs a classification stage
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the planned strategy requires a classification tool that doesn't exist as-is

## Possible Resolution
Build a custom PyTorch CNN classifier on spectrogram patches extracted from detected segments. Use [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] for initial training. This is more work than integrating an existing tool, but ensures full control over the classification stage and avoids format/interface mismatches. The custom classifier can implement both supervised (Scattoni categories) and unsupervised (embedding extraction) branches.
