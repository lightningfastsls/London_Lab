---
description: "The Python USV tool landscape is fragmented — no tool directly accepts pre-detected segments for classification, motivating custom pipeline development"
type: baseline
confidence: likely
conditions:
  - as of February 2026 landscape survey
  - BootSnap was designed for this but code availability is uncertain
meta_state: superseded
superseded_by: "Three classification approaches built in-house by April 2026: rule-based traditional taxonomy (Python), UMAP+HDBSCAN unsupervised (Python), DeepSqueak bridge (MATLAB). The integration gap no longer exists."
source: "inbox/deepsqueak-usv-syllable-classification-practical-guide.md"
topics:
  - "[[classification]]"
---

# No single Python tool cleanly accepts pre-detected USV segments and classifies them into syllable types as of 2026

A comprehensive survey of the Python USV analysis landscape reveals no single tool that cleanly accepts pre-detected USV segments and classifies them into syllable types. The existing tools fall into three categories:

**Designed for pre-detected input but access-limited:** BootSnap (Abbasi et al., 2022) was explicitly built to classify pre-detected USVs into 12 syllable types using gammatone spectrograms and a CNN with snapshot ensemble learning. It achieved the best cross-generalization between wild and lab mice. However, no confirmed public GitHub repository exists — the code may need to be requested from the authors.

**Adaptable but not designed for it:** AMVOC (MIT-licensed, Python/PyTorch) uses a convolutional autoencoder for unsupervised clustering. Its detection module outputs CSVs with onset/offset, and its clustering module could potentially be adapted to accept externally detected segments if formatted correctly.

**Raw-audio-only (not adaptable):** DAS (98%/99% P/R), WhisperSeg, and VocalMat all process raw audio end-to-end and would require significant restructuring to work with pre-detected segments.

This gap is the primary motivation for our custom CNN classifier approach: fine-tuning a pretrained backbone (MobileNetV2/ResNet-18) on spectrogram patches extracted from detected USV segments.

---

Source:
- Compass synthesis: inbox/deepsqueak-usv-syllable-classification-practical-guide.md

Relevant Notes:
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- the best-fit tool but access-limited
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- most adaptable open-source option
- [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]] -- highest metrics but wrong interface
- [[whether BootSnap code is publicly available or must be requested from Abbasi Zala Penn at Vienna]] -- the unresolved access question
- [[DeepSqueak is fundamentally GUI-centric with no officially supported headless or scriptable operation]] -- the GUI-only constraint that compounds the Python tool gap
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- strategic workaround: use DeepSqueak via Raven bridge despite this gap
- [[three viable Python strategies for replacing DeepSqueak target segmentation-first unsupervised discovery and supervised classification]] -- the three compositional strategies that work around this tool gap
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the recommended dual approach requires a custom pipeline precisely because no off-the-shelf tool fills this gap
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- the tool gap exists partly because existing tools assume discrete Holy & Guo categories; the continuum finding invalidates that assumption but no tool has adapted

Topics:
- [[classification-tools]]
