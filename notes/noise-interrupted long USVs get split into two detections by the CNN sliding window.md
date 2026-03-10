---
description: "Long USVs with a noisy gap in the middle are split into two separate detections — inflates call counts but each fragment is a real USV"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# Noise-interrupted long USVs get split into two detections by the CNN sliding window

When a long USV has a noisy gap in the middle, the CNN's sliding window approach splits it into two separate detections. This inflates call counts, but each detected segment is still a real USV fragment — so it is "not necessarily wrong, but worth documenting." The energy detector has gap-bridging via [[segment continuity bridges brief amplitude dips that fragment single USVs]] with a 5 ms max gap, but this CNN-level splitting is a different mechanism: the CNN sees individual spectrogram patches and has no cross-patch continuity logic. If the gap exceeds the sliding window's receptive field, or if the noise is strong enough, the CNN treats each side as an independent detection. A post-CNN merging step could address this but is not currently implemented.

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- energy-detector-level bridging (pre-CNN), does not cover CNN-level splitting
- [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] -- bout-level grouping is a higher level of temporal structure
- [[recall versus precision tradeoff in two-stage USV detection]] -- splitting inflates recall counts without affecting precision per-fragment
- [[saved-previous ghost detections current editable and saved-current form three aligned detection state tiers in the app]] -- ghost detections help the user see previously saved segments that the current CNN pass may have split differently, providing visual context for split boundaries

Topics:
- [[detection]]
