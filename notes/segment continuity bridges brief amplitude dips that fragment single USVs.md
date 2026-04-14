---
description: "Enabled by default with 5 ms max gap, 1.5 kHz frequency tolerance, and 15 dB energy tolerance to merge split segments"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# Segment continuity bridges brief amplitude dips that fragment single USVs

USVs can have brief amplitude dips that cause the energy detector to split a single vocalization into multiple fragments. Segment continuity analysis bridges these gaps by examining frequency and energy patterns in the gap region. It is enabled by default with these parameters: max_gap_ms=5.0 (bridge dips shorter than 5 ms), freq_tolerance_hz=1500 (adjacent segments must be within 1.5 kHz), energy_tolerance_db=15.0 (gap energy within 15 dB of segment energy), and gap_match_fraction=0.6 (at least 60% of gap frames must match criteria). This reduces over-segmentation but may occasionally merge genuinely separate USVs when the gap is less than 5 ms. Notably, the researcher observes that [[noise-interrupted long USVs get split into two detections by the CNN sliding window]] — segment continuity bridging operates at the energy detector level, but the CNN's sliding window approach can re-split long USVs with noisy gaps that exceed the bridging threshold. This downstream splitting is "not necessarily wrong, but worth documenting" since each fragment is still a real USV.

---

Source:
- DECISIONS.md (ADR-013) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- segment continuity operates within the first stage
- [[75 percent overlap with hop length 128 provides smooth temporal coverage for USV detection]] -- high overlap provides the temporal resolution needed for gap analysis
- [[recall versus precision tradeoff in two-stage USV detection]] -- over-bridging merges separate USVs (precision loss) while under-bridging fragments them (recall loss)
- [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] -- bouts group between USVs while segment continuity groups within USVs, two levels of temporal structure
- [[noise-interrupted long USVs get split into two detections by the CNN sliding window]] -- CNN-level splitting that occurs downstream of energy detector bridging
- [[collar-based evaluation with tolerance windows suits bioacoustics better than IoU-based overlap matching]] -- collar-based evaluation tolerates residual fragmentation that gap-bridging does not fully resolve, avoiding unfair IoU penalties on correctly identified calls

Topics:
- [[detection]]
