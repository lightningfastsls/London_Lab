---
description: "usv.pasteur.cloud offers browser-based USV detection testing, potentially useful as a comparison benchmark"
type: finding
confidence: likely
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
---

# Pasteur USV cloud platform enables online testing of detection methods without local infrastructure

The LMT project provides a cloud-based testing platform at usv.pasteur.cloud where researchers can test vocalization detection methods without local installation. This platform could serve as a reference benchmark for comparing our custom detection pipeline's performance against the LMT team's approach. The platform's existence alongside the local LMT USV Toolbox (a Python-based offline processing tool) provides two external reference points for detection quality assessment. Whether these tools use similar detection approaches or fundamentally different methods (e.g., template matching vs. CNN-based) would determine their value as comparison baselines.

---

Source:
- Researcher brain-dump on lab conventions (2026-02-19)
- https://micecraft.org/lmt/

Relevant Notes:
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- the parent system
- [[LMT USV Toolbox provides Python-based offline USV processing as a reference implementation]] -- the offline counterpart from the same team
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- our baseline to compare against
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- another external detection tool for comparison

Topics:
- [[experimental-methods]]
