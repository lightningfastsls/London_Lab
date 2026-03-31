---
description: "±200ms onset/offset tolerance accommodates inherent boundary uncertainty in spectrograms — IoU penalizes fragmented detections that may still correctly identify vocalization presence"
type: method
confidence: likely
conditions: []
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
---

# Collar-based evaluation with tolerance windows suits bioacoustics better than IoU-based overlap matching

Two standard approaches exist for matching detected events to ground-truth annotations when evaluating bioacoustic detection systems. Collar-based matching, the standard in DCASE sound event detection challenges, considers a detection to match a ground-truth event if the detected onset and offset fall within a tolerance window (collar) around the true boundaries — typically ±200ms. IoU-based matching, borrowed from object detection in computer vision, requires that the temporal intersection between detection and ground truth divided by their union exceeds a threshold, typically 0.5.

For bioacoustic applications, collar-based matching is preferred for several reasons. First, spectrogram event boundaries are inherently fuzzy — the onset and offset of a vocalization do not have crisp temporal boundaries because the STFT window smears energy across time, and the vocalization itself may fade in and out gradually rather than switching on and off. A ±200ms tolerance accommodates this inherent uncertainty without penalizing detections that correctly identify the vocalization but disagree slightly on its boundaries.

Second, IoU-based matching penalizes fragmented detections disproportionately. If a single vocalization is detected as two adjacent segments with a brief gap between them, the IoU of each fragment against the ground truth may fall below the threshold even though the detector correctly identified the vocalization's presence and approximate timing. Collar-based matching handles this more gracefully because each fragment's onset or offset may still fall within the tolerance window.

Third, a detection that is slightly too long or too short still correctly identifies the vocalization for downstream behavioral analysis — the precise boundaries matter less than the fact that the call was detected. Our event scoring module implements collar-based matching with ±200ms tolerance and greedy one-to-one assignment to prevent double-counting. Kershenbaum et al. (2025, Biological Reviews) recommends transparent reporting of matching criteria, which is why we explicitly document our choice of collar-based evaluation with specified tolerance values.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[F2 score weights recall approximately 4x more than precision — standard for bioacoustic detection where missed calls bias statistics]] -- the metric computed after collar-based matching determines which detections count as true positives
- [[no existing mouse USV tool uses explicit hysteresis for event detection]] -- the detection method whose output is evaluated using collar-based matching

Topics:
- [[detection]]
