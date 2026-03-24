---
description: "DeepSqueak lacks headless/CLI mode — batch processing exists via GUI buttons but programmatic control requires undocumented MATLAB function calls"
type: finding
confidence: proven
conditions:
  - as of DeepSqueak v3.2
meta_state: current
source: "inbox/deepsqueak-usv-syllable-classification-practical-guide.md"
topics:
  - "[[classification]]"
---

# DeepSqueak is fundamentally GUI-centric with no officially supported headless or scriptable operation

Despite supporting batch operations (Multi-Detect, batch clustering, batch export), DeepSqueak is fundamentally GUI-centric. There is no command-line interface, no Python API, and no officially documented way to run it headless. Advanced MATLAB users can call underlying functions like `SqueakDetect` programmatically, but this is undocumented and fragile across versions.

Batch operations available through the GUI include: multi-file detection, post-hoc denoising, threshold-based rejection, unsupervised clustering, supervised classification, and Excel export. These cover common workflows but require manual initiation.

A critical constraint: pure "classification only" mode without audio access is not possible. DeepSqueak regenerates spectrograms from raw audio during classification, so the original WAV files must remain accessible. This means you cannot ship just bounding boxes to DeepSqueak for classification — the audio files must be co-located.

For our pipeline, this reinforces the Raven selection table import as the practical bridge: export our detections as Raven tables, import into DeepSqueak with audio files present, run classification through the GUI, then export results back.

---

Source:
- Compass synthesis: inbox/deepsqueak-usv-syllable-classification-practical-guide.md

Relevant Notes:
- [[DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port]] -- the broader ecosystem constraint
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the workaround format
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- why audio must be present
- [[No single Python tool cleanly accepts pre-detected USV segments and classifies them into syllable types as of 2026]] -- the Python landscape gap this GUI constraint compounds
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- strategic workaround using Raven bridge despite GUI limitation

Topics:
- [[classification-tools]]
