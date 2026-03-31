---
description: "Sustain threshold naturally keeps events alive across brief dips (gap-fill), while onset threshold filters isolated noise spikes (min-duration) — fewer parameters, more principled"
type: method
confidence: proven
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
---

# Hysteresis subsumes gap-filling and minimum duration as special cases of dual-threshold logic

The Schmitt trigger analogy clarifies why dual-threshold hysteresis replaces two separate post-hoc filters with a single unified mechanism. In a Schmitt trigger, the output switches high when the input exceeds an upper threshold and remains high until the input drops below a lower threshold. This creates built-in inertia — once an event starts, brief dips in signal strength do not terminate it, because the sustain threshold is lower than the onset threshold.

Gap-filling is therefore an emergent property of hysteresis rather than an add-on: when the CNN probability momentarily drops below the onset threshold but stays above the sustain threshold, the event simply continues. There is no need for a separate gap_fill_windows parameter to stitch together nearby fragments. Similarly, minimum duration filtering becomes largely unnecessary because isolated noise spikes must exceed the higher onset threshold, which they rarely do for multiple consecutive windows — the onset threshold itself acts as a noise gate.

Our optimized hysteresis configuration validates this empirically. The grid search in optimize_hysteresis.py converged on gap_fill_windows=0 and min_duration_windows=3, meaning the optimizer found that the post-hoc constraints added almost nothing when hysteresis was already handling the underlying problems. The min_duration_windows=3 value is minimal (representing roughly 50 ms at our hop size), suggesting it catches only the rarest edge cases that hysteresis misses. This is a principled reduction in parameter count: from four independent parameters (threshold, gap_fill, min_duration, smoothing) to two primary ones (onset_threshold, sustain_threshold) with the others effectively zeroed out by the optimizer.

The practical benefit is reduced hyperparameter search space and more interpretable configurations, because each parameter has a clear physical meaning rather than being one of several interacting post-hoc corrections.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[no existing mouse USV tool uses explicit hysteresis for event detection]] -- the landscape gap that motivates adopting this approach
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- the noise pattern that the onset threshold specifically addresses

Topics:
- [[detection]]
