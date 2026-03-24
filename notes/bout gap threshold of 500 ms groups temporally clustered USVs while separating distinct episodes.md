---
description: "USVs within 500 ms of each other are grouped into the same bout, with inter-bout gaps treated as episode boundaries"
type: decision
confidence: likely
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# Bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes

The bout gap threshold of 500 ms determines how USV detections are grouped for transformer training. USVs occurring within 500 ms of each other are treated as part of the same behavioral episode and grouped into a single bout. Gaps longer than 500 ms are treated as episode boundaries, creating separate bouts. This parameter choice reflects the assumption that USVs clustered within half a second are likely part of the same communicative sequence, while longer gaps indicate distinct episodes. Since [[optimal bout gap threshold may vary across behavioral contexts and recording conditions]], this value may need adjustment for different experimental paradigms.

---

Source:
- DECISIONS.md (ADR-014) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- the bout extraction pipeline this feeds
- [[optimal bout gap threshold may vary across behavioral contexts and recording conditions]] -- acknowledged uncertainty in this parameter

Topics:
- [[detection]]
