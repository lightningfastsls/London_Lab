---
source: researcher brain-dump
topic: lab-specific conventions
date: 2026-02-19
method: structured interview (AskUserQuestion) + web research (micecraft.org/lmt/)
---

# Lab-Specific Conventions — Brain Dump

## Hardware
Researcher doesn't remember specific microphone/recorder hardware specs. The system is part of the LMT (Live Mouse Tracker) from Institut Pasteur (micecraft.org). LMT uses AviSoft Recorder for synchronized USV recording with behavioral events.

## Recording Software and Settings
- **Software**: AviSoft Recorder (integrated with LMT behavioral tracking)
- **Sample rate**: 300 kHz (confirmed — earlier references to 250 kHz are outdated/legacy)
- **File format**: WAV
- **Frequency range of interest**: 20-120 kHz
- **STFT parameters**: n_fft=512, hop_length=128 (at 300 kHz gives ~586 Hz frequency resolution)

### Sample Rate History
Both 250 kHz and 300 kHz appear in project documentation. 300 kHz is correct (ADR-001). The 250 kHz references are from earlier documentation that was not fully updated. SpectrogramConfig.expected_sample_rate_hz = 250_000 is explicitly marked as outdated in DECISIONS.md.

## Acoustic Environment
Shared lab space (not a dedicated sound-attenuated room). This implies:
- Higher background noise levels than ideal
- Potential electrical interference from other lab equipment
- The CNN and detection pipeline must be robust to real-world noise conditions
- This context explains why noise robustness is a key concern in training

## LMT System Integration
The Live Mouse Tracker (https://micecraft.org/lmt/) synchronizes USV recordings with behavioral events, enabling researchers to correlate vocalizations with specific social behaviors (e.g., approaches, contacts, chases). The LMT USV Toolbox (Python) is available for offline processing, and there is also an online testing platform at https://usv.pasteur.cloud.
