---
description: "The frequencies and bandwidths of cage equipment tonals are properties of specific recording chambers, not transferable defaults — a tonal library calibrated on lab 131204 is a no-op on any other cage and must be re-derived per environment"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[experimental-methods]]"
---

# Soft-notch tonal libraries are cage-specific empirical artifacts requiring per-cage calibration

The soft-notch tonal removal layer (`src/usv_spectrogram/app/core/notch.py`) suppresses cage equipment tonals — narrowband persistent frequencies emitted by microphone preamps, fluorescent lights, AC power harmonics, fan motors, and other ambient electronics in the recording environment. For our lab 131204 setup, the empirical tonal library at `data/lab_tonal_lines/lab_131204.json` lists the specific frequencies measured in that cage: a 51 kHz tone and a 46.58 kHz tone, with their bandwidths and persistence-across-recordings statistics. The notch layer uses this library to apply targeted spectral suppression at exactly those frequencies.

The library is calibrated for *one* recording chamber. The 51 kHz tone is not a universal mouse-USV-environment artifact; it's specific to the equipment in lab 131204. A different lab using different hardware will have different tonals at different frequencies. Our 5970 wild recordings use a different microphone setup and presumably different tonals (which we have not yet measured). VocalMat's data comes from 5 different lab environments across 30+ recording sessions — none of which match our lab 131204 tonal signature.

The implication: when running the cleaning stack on cohorts other than lab 131204, the soft-notch layer is configured to no-op (`tonal_library_path=None`). The classifier pipeline's cleaning stack relies on the other three layers — Boll baseline subtraction, global MAD, per-recording Z-score — to handle non-lab-131204 cohorts. The cleaning is asymmetrically applied: lab 131204 gets all 4 layers, everyone else gets 3.

To extend soft-notch coverage to a new cage, run `scripts/calibrate_lab_tonal_lines.py` on a representative recording from that cage. The script extracts persistent narrowband features above a configurable persistence threshold and outputs a JSON tonal library that the notch layer can load. This is empirical work, not configuration — each cage's tonal signature is its own dataset to characterize.

The general pattern: when a preprocessing component encodes properties of the physical world (which frequencies are noise in this room?), the encoding is environment-specific and the maintenance discipline is empirical re-calibration per environment, not parameter sharing across environments. Tonal libraries belong in `data/`, not `src/`, because they are facts, not code.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[cage acoustics drive between-cohort spectrogram separation more than biology]] — the broader confound soft-notch is one defense against
- [[per-recording normalization compensates for varying noise floors across recording sessions]] — different cleaning layer with different scope

Topics:
- [[signal-processing]]
- [[experimental-methods]]
