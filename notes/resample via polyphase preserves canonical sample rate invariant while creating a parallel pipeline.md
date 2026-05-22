---
description: "When integrating with an external dataset that uses a different sample rate, polyphase resampling at a fixed ratio creates a parallel internal pipeline without altering the codebase's canonical sample rate invariant — both pipelines can coexist"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[classification-methodology]]"
---

# Resample via polyphase preserves canonical sample rate invariant while creating a parallel pipeline

The USV detection codebase enforces a canonical sample rate invariant at `src/usv_spectrogram/corpus.py`: `SAMPLE_RATE_HZ = 300_000`. Every detection-pipeline module imports from corpus; the constant is asserted at startup; no module is allowed to redeclare it. This invariant is load-bearing because the production CNN (`models/hard_neg_retrain/best_model.pt`) was trained at 300 kHz on this exact frequency grid, and downstream code (ExtractionConfig, sliding_inference, hysteresis) operates on tensors with shapes derived from this sample rate. Changing corpus.SAMPLE_RATE_HZ would silently break inference.

VocalMat's published dataset uses 250 kHz audio. To use VocalMat for training, we must resample our own 300 kHz recordings to 250 kHz to match VocalMat's preprocessing conventions. The naive approach — change corpus.SAMPLE_RATE_HZ from 300_000 to 250_000 — would corrupt every downstream detection module.

The architectural fix is to create a parallel sample-rate pipeline that doesn't go through corpus. The `src/usv_spectrogram/classifier/resample.py` module defines its own constants (`TARGET_SAMPLE_RATE_HZ = 250_000`, `RESAMPLE_UP = 5`, `RESAMPLE_DOWN = 6`) and applies polyphase resampling via `scipy.signal.resample_poly(samples, up=5, down=6)` — the 5/6 ratio converts 300 kHz to 250 kHz exactly (300_000 * 5 / 6 = 250_000). The classifier package operates internally at 250 kHz; the detection package continues to operate at 300 kHz; neither imports the other's sample-rate constants.

Why polyphase: `resample_poly` uses a Kaiser-windowed FIR filter to handle the anti-aliasing implicitly. At a 5/6 ratio, the new Nyquist frequency is 125 kHz (vs 150 kHz at 300 kHz). Any audio energy between 125 kHz and 150 kHz would alias into the passband without anti-aliasing — the FIR filter suppresses it by >40 dB. For mouse USV recordings where signal energy is concentrated below 120 kHz, the loss is acceptable; the anti-aliasing is the only thing standing between us and band corruption.

The methodological principle: when integrating with an external dataset that uses different DSP conventions, build a parallel pipeline rather than changing the canonical invariant. Both pipelines coexist, each is internally consistent, and the boundary between them is one explicit conversion step. Changing the invariant to accommodate the external dataset would distribute its assumptions across the codebase irrecoverably.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[the 300 kHz USV sample rate creates a domain shift challenge for applying audio foundation models]] — the broader sample-rate-domain-shift concern
- [[cage acoustics drive between-cohort spectrogram separation more than biology]] — a different domain-shift confound in the same pipeline

Topics:
- [[signal-processing]]
- [[classification-methodology]]
