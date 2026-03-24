---
status: deferred
created: 2026-02-23
reviewed: 2026-03-21
reviewed_by: rethink-2026-03-21
review_note: "DeepSqueak bridge (Phase 3) provides immediate MATLAB pathway. Four Python-native resolution paths identified (custom CNN, few-shot, freq shifting, unsupervised). Revisit after DeepSqueak bridge operational."
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

## Possible Resolutions (updated 2026-03-01 after /reduce batch)

### Path A: Custom PyTorch CNN (original)
Build a custom PyTorch CNN classifier on spectrogram patches extracted from detected segments. Use [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] for initial training. This is more work than integrating an existing tool, but ensures full control over the classification stage and avoids format/interface mismatches. The custom classifier can implement both supervised (Scattoni categories) and unsupervised (embedding extraction) branches.

### Path B: Few-shot via foundation model embeddings (NEW)
Since [[foundation model embeddings enable few-shot classification via simple linear probes without end-to-end training]], extract embeddings from BEATs or AVES, then classify with k-NN or prototypical networks from ~10 labeled examples per type. Since [[prototypical networks are the dominant paradigm for few-shot bioacoustic event detection]], this is well-supported methodologically. The challenge is that [[the 300 kHz USV sample rate creates a domain shift challenge for applying audio foundation models]] — requires spectrogram-as-image or frequency shifting.

### Path C: Frequency shifting to audible range (NEW)
Since [[frequency shifting USVs into the audible range could enable classification with standard audio foundation models]], pitch-shift detected USV segments from 50-90 kHz to 2-10 kHz. This would unlock the entire ecosystem of speech/audio foundation models. Untested but theoretically sound.

### Path D: Unsupervised clustering (NEW)
Use [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] on spectrogram features from detected segments. Since [[unsupervised clustering as post-detection filtering eliminates 88 percent false positives while retaining 95 percent true positives]], this doubles as both classification and quality filtering. No labels needed.

### Current Strategy
DeepSqueak bridge (Phase 3, in progress) provides an immediate MATLAB-based classification pathway. Paths B-D are longer-term Python-native alternatives. The tension remains pending but has significantly more resolution options than when first identified.
