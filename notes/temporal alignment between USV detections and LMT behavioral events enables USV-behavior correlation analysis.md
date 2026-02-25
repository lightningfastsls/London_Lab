---
description: "Correlating specific USV types with behavioral outcomes (e.g., mounting success) requires temporal alignment between acoustic and video data"
type: method
confidence: speculative
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
---

# Temporal alignment between USV detections and LMT behavioral events enables USV-behavior correlation analysis

A key analytical method for this research is correlating specific USV call types with specific behavioral outcomes. For example, does a certain type of USV make the female more likely to allow the male to mount her? This requires temporal alignment between USV detection timestamps and behavioral events from LMT/MiceCraft video analysis. The [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] provides the infrastructure, but the analytical pipeline for correlation analysis (event windowing, statistical testing, multiple comparisons correction) has not yet been built. This method would provide the strongest evidence for USV communicative function — moving beyond "USVs occur during courtship" to "specific USVs predict specific outcomes."

---

Source:
- Researcher brain-dump on scientific hypotheses (2026-02-19)

Relevant Notes:
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- the recording infrastructure that makes this possible
- [[USVs are one component of a multimodal courtship behavior suite including mounting approach and movement]] -- the multimodal framework
- [[whether specific USV call types predict specific courtship outcomes like female receptivity to mounting]] -- the open question this method addresses
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- the same engineering pattern of aligning timestamps across different analysis systems

Topics:
- [[experimental-methods]]
