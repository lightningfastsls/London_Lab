---
description: "DeepSqueak (Coffey et al.) is the most widely used USV tool but its monolithic architecture doesn't separate recall from precision tuning"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision

DeepSqueak (Coffey et al.) is the most widely used tool for USV detection and analysis. It uses Faster R-CNN / YOLO-based object detection followed by k-means clustering for classification. However, its monolithic detection approach does not allow independent control of recall and precision — the detector either finds a region or it doesn't. Our pipeline was built specifically for more control: since [[two-stage detection uses permissive energy detector followed by CNN precision filter]], we can tune the energy detector for maximum recall (via [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]]) independently of the CNN's precision filtering. This separation is a deliberate architectural choice motivated by the specific needs of the wild vs lab mouse comparison study.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- Coffey et al. — DeepSqueak

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- our architectural response to DeepSqueak's limitation
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- the general pattern behind our approach
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- another detection tool comparison
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- pragmatic strategy: use DeepSqueak's classification immediately while building our VQ-VAE pipeline
- [[LMT USV Toolbox provides Python-based offline USV processing as a reference implementation]] -- another USV tool in the competitive landscape

Topics:
- [[detection]]
- [[classification]]
