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
- [[collar-based evaluation with tolerance windows suits bioacoustics better than IoU-based overlap matching]] -- DCASE challenges use collar-based matching as the standard evaluation protocol for measuring these post-processing gains
- [[F2 score weights recall approximately 4x more than precision — standard for bioacoustic detection where missed calls bias statistics]] -- the recall-weighted metric that makes post-processing optimization especially impactful for bioacoustic applications
- [[scikit-maad implements double-threshold hysteresis binarization for ecological acoustics]] -- an established ecological acoustics library implementing one of the post-processing techniques (hysteresis) whose careful optimization DCASE validates
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- the general pattern that DCASE validates: each stage (model vs post-processing) can be independently optimized, and post-processing alone yields large gains
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- our specific pipeline whose post-processing layer (hysteresis optimization) is justified by Cances et al.'s finding that tuning post-processing independently yields 7pp F1 improvement
- [[Clarfeld 2025 secondary logistic regression on primary detections achieved 85-90 percent FP filtering accuracy]] -- cross-taxa validation of the same principle: what happens after the primary detector matters as much as the detector itself
- [[unsupervised clustering as post-detection filtering eliminates 88 percent false positives while retaining 95 percent true positives]] -- an alternative post-detection precision technique validating the same DCASE insight that post-processing investment pays off

Topics:
- [[detection]]
