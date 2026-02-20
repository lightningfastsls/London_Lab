---
description: "Bouts group nearby USV detections with 500 ms gap threshold and 200 ms padding, preserving temporal context"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# Bout-level spectrograms preserve inter-USV timing context for transformer training

The transformer architecture (Phase 1 of ADR-007) needs input that preserves inter-USV timing and context. Three options were considered: isolated USV crops (~40 ms windows, discards context), full WAV files (seconds to minutes, mostly silence), and bouts (clustered USV groups with padding). Bouts were chosen as the middle ground. Nearby USV detections are grouped using a [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]], with 200 ms context padding before the first and after the last USV. Minimum bout duration is 50 ms (shorter bouts are likely noise) and maximum is 10,000 ms (longer bouts are split). The transformer sees continuous acoustic stream at ~0.427 ms per frame, and can discover structure at whatever granularity is informative.

### ROADMAP Context

ROADMAP specifies the chunking strategy for feeding bouts into the transformer: max_seq_len=512 frames, overlap_ratio=0.5 (stride=256), with attention masks for padding (1=real frame, 0=padding). Spectrograms are transposed to (T × 170) where each row is one "token" for the transformer. For next-column prediction: input = frames[0:T-1], target = frames[1:T]. Bouts shorter than max_seq_len are zero-padded to fill the window; longer bouts are chunked with 50% overlap so no context is lost at chunk boundaries. See also [[length-bucketed batching minimizes padding waste when sequences vary in duration]].

---

Source:
- DECISIONS.md (ADR-014) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] -- the specific grouping parameter
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- the architecture consuming bout data
- [[75 percent overlap with hop length 128 provides smooth temporal coverage for USV detection]] -- determines frame rate within bouts
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- operates within individual USVs while bouts group between USVs, two complementary levels of temporal structure

Topics:
- [[detection]]
- [[representation-learning]]
