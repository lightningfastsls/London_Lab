---
description: "AviSoft Recorder is the recording software, integrated with the LMT system for synchronized audio-behavior capture"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[signal-processing]]"
---

# AviSoft Recorder captures synchronized USV recordings within the LMT behavioral tracking system

The USV recordings are captured using AviSoft Recorder, which is integrated with the Live Mouse Tracker (LMT) system from Institut Pasteur. AviSoft Recorder is widely used in rodent ultrasonic vocalization research and supports the high sample rates required for USV capture. In this setup it records WAV files at 300 kHz, covering the 20-120 kHz frequency range of interest for mouse USVs. The integration with LMT means that since [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]], each USV recording has a temporal alignment with behavioral data — enabling correlation between specific call patterns and social interactions.

---

Source:
- Researcher brain-dump on lab conventions (2026-02-19)
- https://micecraft.org/lmt/

Relevant Notes:
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- the behavioral tracking system AviSoft integrates with
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- the sample rate AviSoft is configured to use
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- the recording environment

Topics:
- [[experimental-methods]]
- [[signal-processing]]
