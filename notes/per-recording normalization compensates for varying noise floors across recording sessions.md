---
description: "Dynamic per-spectrogram vmin/vmax normalization based on per-recording statistics prevents model performance from varying across recordings"
type: method
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[classification]]"
---

# Per-recording normalization compensates for varying noise floors across recording sessions

Different recordings have different noise floors — a recording from one session might have a higher ambient noise level than another due to varying equipment states, environmental conditions, or microphone positioning. Without per-recording normalization, the model's performance varies across recordings because the same USV appears at different absolute energy levels in different recordings. The solution is dynamic per-spectrogram normalization using vmin/vmax based on per-recording statistics, plus per-sample normalization (min-max to [0,1] or standardization) for CNN input. This complements [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] — per-bin normalization corrects frequency-dependent energy bias, while per-recording normalization compensates for session-level noise floor variation. As noted in [[normalization statistics must be computed on training set only to prevent data leakage]], all normalization statistics must come from training data only.

---

Source:
- Researcher brain-dump on preprocessing insights (2026-02-19)

Relevant Notes:
- [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] -- complementary normalization on the frequency axis
- [[normalization statistics must be computed on training set only to prevent data leakage]] -- data leakage constraint on normalization
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- why noise floors vary
- [[electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs]] -- one source of varying noise floor: electrical interference intensity differs between sessions
- [[transient cage noises produce broadband vertical smears rejected by the minimum duration filter]] -- another source of session-varying noise that per-recording normalization helps compensate

Topics:
- [[signal-processing]]
- [[classification]]
