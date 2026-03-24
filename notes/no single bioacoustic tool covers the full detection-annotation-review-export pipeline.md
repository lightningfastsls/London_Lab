---
description: "The ecosystem is fragmented — detection, annotation, review, and export are handled by different tools with format conversion friction between them"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[classification]]"
---

# No single bioacoustic tool covers the full detection-annotation-review-export pipeline

A survey of the bioacoustic annotation landscape in 2025 reveals that no single tool handles the complete pipeline from detection through annotation, review, and export. Detection tools (DAS, SqueakOut, VocalMat) focus on finding calls but lack human review workflows. Annotation platforms (Raven, Whombat, Sonic Visualiser) provide labeling interfaces but no built-in detection. Format interoperability tools (Crowsetta) bridge the gap but add a software dependency layer.

This fragmentation means researchers must compose multi-tool workflows with format conversion at each boundary — Raven selection tables to Python scripts to ML training frameworks. The practical consequence is significant friction: every tool transition risks data loss, format incompatibility, or manual re-entry. Therefore, building a pipeline that spans multiple stages requires deliberate interop design rather than relying on any single tool's built-in capabilities.

Our own pipeline reflects this pattern — energy detection produces candidates, which need manual review in a separate tool, followed by export for classification training. Because no tool owns the full workflow, the Raven selection table format has emerged as the lingua franca that stitches the ecosystem together, but even this requires custom adapters at each boundary.

---

Source:
- bioacoustic-annotation-tools-landscape-2025-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] — our pipeline is one instance of this fragmented pattern

Topics:
- [[detection]]
- [[classification]]
