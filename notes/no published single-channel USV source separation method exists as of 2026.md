---
description: "BioCPPNet handles macaques, dolphins, and bats but has not been tested on ultrasonic frequencies — the cocktail party problem for rodent USVs remains unsolved"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[signal-processing]]"
---

# No published single-channel USV source separation method exists as of 2026

Despite bioacoustic source separation methods existing for audible-range species, no published work addresses single-channel source separation for ultrasonic vocalizations. BioCPPNet (Earth Species Project, 2021) demonstrated neural network source separation for macaques, dolphins, and Egyptian fruit bats with SI-SDR scores of ~10.3-10.6 dB, but was tested only on audible-range recordings. The 300 kHz sample rate and 25-120 kHz frequency range of mouse USVs present a domain that has not been explored. Labs routinely discard overlapping USV recordings because they cannot separate them, representing a significant data loss. Synthetic mixture training (mixing isolated single-source recordings) is the standard approach for training separation networks — viable for USVs from single-animal recordings.

---

Source:
- overlapping-usv-source-separation-state-of-art-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:

Topics:
- [[detection]]
- [[signal-processing]]
