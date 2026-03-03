---
status: archived
archived: 2026-03-02
archived_by: rethink-2026-03-02
created: 2026-02-19
resolved: 2026-02-19
---

# Segment continuity bridges gaps in the energy detector but noise-interrupted long USVs still get split at the CNN level

The energy detector has segment continuity bridging (5 ms max gap), but the researcher reports CNN-level splitting of long USVs with noisy gaps. The gap may exceed 5 ms, or the CNN's sliding window approach inherently cannot recombine what it sees as separate patches. This tension points to a gap in the pipeline: bridging happens pre-CNN but not post-CNN.

## Conflicting Notes
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- describes pre-CNN bridging
- [[noise-interrupted long USVs get split into two detections by the CNN sliding window]] -- describes post-CNN splitting

## Resolution
**Dissolved.** These operate at different pipeline stages on different phenomena:
- Energy detector bridging handles brief amplitude dips (sub-5ms) within a single USV
- CNN splitting happens at actual noise interruptions where the signal is genuinely absent

Both observations are correct. The tension was a framing issue, not a contradiction. Whether to add post-CNN merging is a feature decision, not a conflict — and the answer depends on whether noise-interrupted fragments should be treated as one call or two (a domain question, not an implementation bug).
