---
status: pending
created: 2026-02-19
---

# Segment continuity bridges gaps in the energy detector but noise-interrupted long USVs still get split at the CNN level

The energy detector has segment continuity bridging (5 ms max gap), but the researcher reports CNN-level splitting of long USVs with noisy gaps. The gap may exceed 5 ms, or the CNN's sliding window approach inherently cannot recombine what it sees as separate patches. This tension points to a gap in the pipeline: bridging happens pre-CNN but not post-CNN.

## Conflicting Notes
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- describes pre-CNN bridging
- [[noise-interrupted long USVs get split into two detections by the CNN sliding window]] -- describes post-CNN splitting

## Resolution Path
Could be resolved by: (a) accepting the split as correct behavior (each fragment is real), (b) adding post-CNN merging logic, or (c) widening the sliding window receptive field.
