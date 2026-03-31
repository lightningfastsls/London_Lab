---
description: "Cances et al 2019 WASPAA showed that tuning onset/offset thresholds per sound class via dichotomic search yields large gains in sound event detection"
type: finding
confidence: proven
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
---

# DCASE class-dependent post-processing parameters improved F1 from 37 to 44 percent

Cances et al. (2019, WASPAA) demonstrated that class-dependent post-processing — tuning onset thresholds, offset thresholds, and smoothing parameters independently for each sound class — improved macro-averaged F1 from 37.1% to 43.9% on the DCASE 2019 Task 4 sound event detection challenge. This is a substantial gain achieved purely through post-processing optimization, without changing the underlying neural network model.

Their method used dichotomic search, which operates as a coarse-to-fine grid search: first sweep a wide parameter range at coarse granularity, identify the region containing the optimum, then refine with finer steps within that region. This is more efficient than exhaustive grid search when the parameter space is large, because it avoids evaluating the full Cartesian product at fine resolution.

However, our hysteresis optimization problem has a smaller parameter space — four parameters (onset_threshold, sustain_threshold, gap_fill_windows, min_duration_windows) — which makes exhaustive grid search feasible. The optimize_hysteresis.py script evaluates all combinations within a defined grid, which is practical because each evaluation is fast (the CNN probabilities are precomputed, so scoring a configuration requires only array operations). Therefore, while dichotomic search is a useful reference technique, our problem is small enough that we do not need it.

The key insight from Cances et al. is not the search method itself but the finding that post-processing parameters matter enormously — a 7 percentage point F1 improvement from tuning alone. This validates our investment in careful hysteresis parameter optimization rather than treating post-processing as an afterthought with hand-tuned defaults.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[no existing mouse USV tool uses explicit hysteresis for event detection]] -- the landscape context showing that existing tools do not tune post-processing this carefully
- [[hysteresis subsumes gap-filling and minimum duration as special cases of dual-threshold logic]] -- our specific post-processing approach that benefits from this optimization philosophy

Topics:
- [[detection]]
