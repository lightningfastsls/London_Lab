---
description: "The LMT project's Python toolbox for offline USV processing is a potential reference implementation and comparison point"
type: finding
confidence: likely
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
---

# LMT USV Toolbox provides Python-based offline USV processing as a reference implementation

The Live Mouse Tracker project provides a standalone Python toolbox for offline USV processing, available on GitHub. This is a natural comparison point for our custom detection pipeline — the LMT toolbox is developed by the same group (Institut Pasteur) that builds the behavioral tracking system we use. Comparing detection results, feature extraction approaches, and classification methods between our pipeline and the LMT toolbox could identify strengths and weaknesses in each approach. The toolbox is likely tuned for the same type of recordings (AviSoft Recorder output at 300 kHz from shared lab environments), making it a particularly relevant benchmark.

---

Source:
- Researcher brain-dump on lab conventions (2026-02-19)
- https://micecraft.org/lmt/

Relevant Notes:
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- the parent system
- [[AviSoft Recorder captures synchronized USV recordings within the LMT behavioral tracking system]] -- same recording pipeline
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- our pipeline to compare against

Topics:
- [[experimental-methods]]
