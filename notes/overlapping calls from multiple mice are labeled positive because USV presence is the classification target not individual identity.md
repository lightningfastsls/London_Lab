---
description: "When two mice vocalize simultaneously, the label is USV-present — the system detects presence, not source attribution"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# Overlapping calls from multiple mice are labeled positive because USV presence is the classification target not individual identity

When two mice in the same cage vocalize simultaneously, producing overlapping calls in the spectrogram, the region is labeled positive (USV present). The detection system's classification target is USV presence/absence, not counting individual calls or attributing them to specific animals. This simplification is appropriate for the current research question — since [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]], what matters is detecting that vocalizations occur, not decomposing overlapping sources. Both stages of the [[two-stage detection uses permissive energy detector followed by CNN precision filter]] pipeline detect presence, not identity.

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- both stages detect presence, not identity
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question that shapes this labeling decision
- [[Ivanenko et al 2020 showed DNNs achieve 77-84 percent accuracy classifying emitter sex from spectrograms]] -- demonstrates that identity information IS present in spectrograms, though we deliberately ignore it for the presence-detection task
- [[wearable miniature microphones achieve 90 percent USV attribution from amplitude alone]] -- hardware-based attribution approach that could complement this presence-only labeling if individual identity becomes needed

Topics:
- [[detection]]
- [[classification]]
