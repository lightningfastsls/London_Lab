---
description: "Strategic decision: use DeepSqueak's existing classification for immediate repertoire comparison before the custom VQ-VAE pipeline is ready"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations

The core courtship degradation finding can likely be demonstrated before the VQ-VAE pipeline is complete, by using DeepSqueak's built-in classification to categorize each USV call and comparing repertoire distributions between wild and lab populations. This is a strategic decision: the most important scientific question (do wild and lab mice differ in USV repertoire?) does not require a custom pipeline. DeepSqueak, despite its limitations for detection since [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]], has a useful classification system that can provide immediate scientific value. The VQ-VAE + Transformer work then becomes a deeper investigation into whether sequential structure has language-like properties — a separate, more ambitious question.

---

Source:
- Researcher brain-dump on scientific hypotheses (2026-02-19)

Relevant Notes:
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- DeepSqueak's detection limitation vs classification utility
- [[VQ-VAE investigation of language-like sequential structure in USVs is a separate deeper question from courtship degradation]] -- the two-tier research strategy
- [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]] -- the finding this approach would formalize
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question this approach directly serves
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- DeepSqueak classification uses traditional types; VQ-VAE later tests whether continuum-based discretization is more informative

Topics:
- [[classification]]
- [[experimental-methods]]
