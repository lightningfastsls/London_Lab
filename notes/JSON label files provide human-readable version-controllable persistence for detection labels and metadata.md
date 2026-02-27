---
description: "One JSON file per WAV stores detection metadata, boundaries, user labels, and probability curves -- human-readable, git-friendly, and inspectable"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[detection]]"
---

# JSON label files provide human-readable version-controllable persistence for detection labels and metadata

The desktop labeling app persists user labels, detection boundaries, and metadata as JSON files -- one per WAV file. Each file contains: metadata (WAV path, model file, timestamps, detection count, file-level label), detection parameters (high/low thresholds), individual detections (start/end times, durations, probabilities, user adjustments), and the full probability curve.

JSON was chosen over binary formats for three reasons: human readability (easy to inspect and debug), Git compatibility (meaningful diffs for version control), and simplicity (no schema migration overhead). File sizes are small enough that the verbosity of JSON is not a concern.

The schema includes both model-generated fields (probabilities, boundaries) and human-generated fields (user_adjusted, user_action), making each label file a complete record of both automated detection and human curation.

---

Source:
- DECISIONS.md (ADR-010) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- training data that labels help generate
- [[batch detection with skip-existing enables incremental processing of large WAV collections]] -- detection outputs that feed the labeling workflow

Topics:
- [[classification]]
- [[detection]]
