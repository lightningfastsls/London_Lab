---
description: "Signals near the minimum duration threshold are an unresolved edge case for labeling — inclusion affects training data and energy detector parameters"
type: open-question
confidence: speculative
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[experimental-methods]]"
---

# Whether very short USV signals near the 8-10 ms boundary should be included or excluded from training

Very short signals near the 8-10 ms boundary are a recognized edge case in USV labeling. Including them as positives would increase training data diversity and potentially improve detection of brief calls, but risks introducing label noise if some are actually noise artifacts. Excluding them is conservative but may systematically miss a category of real USVs — particularly since [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]]. The minimum duration parameter in the energy detector interacts with this decision: if the energy detector has a minimum duration threshold, signals below it are never generated as candidates regardless of labeling policy. Notably, the minimum duration filter serves a dual purpose: [[transient cage noises produce broadband vertical smears rejected by the minimum duration filter]], so the 8-10 ms threshold exists partly for artifact rejection, not just USV definition. Resolution requires examining the distribution of signal durations and determining whether the 8-10 ms region contains genuine USVs or is dominated by artifacts.

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]] -- short signals are already a known blind spot
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- gap bridging may create or eliminate short fragments

Topics:
- [[detection]]
- [[experimental-methods]]
