---
description: "A 1.17g headgear-mounted ultrasound microphone exploits the 10-20 dB amplitude advantage from acoustic proximity — 97% attribution when combined with video tracking"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[experimental-methods]]"
---

# Wearable miniature microphones achieve 90 percent USV attribution from amplitude alone

Cell Reports Methods (2025) reports a wearable ultrasound-sensitive microphone weighing only 1.17g mounted on mouse headgear. USVs are attributed based on relative amplitude difference between paired microphones — the acoustic port at ~0 degrees to the vocalizer's mouth creates a 10-20 dB advantage over the partner mouse's signal. This simple amplitude criterion achieves 90% attribution from acoustic data alone, rising to 97% when combined with video tracking. A distance correction algorithm accounts for varying animal-to-microphone geometry. However, the method was tested only on pairs (not groups of 3+) and requires physical attachment to animals, which is not suitable for all experimental designs. The approach solves attribution (which mouse vocalized) but not separation (extracting individual calls from an acoustic mixture).

---

Source:
- overlapping-usv-source-separation-state-of-art-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[HyVL hybrid beamforming achieves 3 to 5 mm USV localization precision with 91 percent source assignment]] — complementary hardware approach using spatial rather than amplitude cues

Topics:
- [[detection]]
- [[experimental-methods]]
