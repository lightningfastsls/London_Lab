---
description: "Custom [32, 96, 192] filter config with dense_units=64 (~207K params) sits between small (101K) and medium (400K) presets, targeting the 14,700-sample matched-windows retrain with a ~71:1 sample-to-param ratio"
type: decision
confidence: likely
conditions:
  - "validate after matched-windows retrain: compare val loss curves against small preset to confirm capacity gain without overfitting"
meta_state: current
topics:
  - "[[classification]]"
  - "[[training-methodology]]"
---

# mid-C CNN balances capacity and inference speed for 14K samples

The matched-windows retrain has 14,700 samples (5,646 positives, 9,034 negatives). The existing small preset [32, 64, 128] with ~101K params was designed for 2K-10K samples and risks underfitting at this scale. The medium preset [64, 128, 256] with ~400K params targets 15K-20K samples -- overkill for 14.7K, and inference time roughly doubles (21.7ms vs 9.6ms per window).

The mid-C configuration [32, 96, 192] with dense_units=64 reaches ~207K params, yielding a sample-to-param ratio of approximately 71:1. This sits in the healthy range where the model has enough capacity to learn complex spectral patterns without memorizing the training set. The architecture rationale has three parts:

1. **First layer stays small (32 filters).** Early convolutional layers extract low-level features (edges, frequency bands). These are cheap and well-constrained -- 32 filters suffice, matching both presets. No reason to double early computation.

2. **Deeper layers widen (96, 192).** Higher layers combine low-level features into complex spectral-temporal patterns. The jump from 64 to 96 and from 128 to 192 gives the model more capacity where it matters -- at the level of spectral combination rather than raw feature extraction.

3. **Classifier head stays small (64 units).** The dense layer is the most overfitting-prone component. Keeping it at 64 (same as the small preset) constrains the decision boundary while the convolutional backbone does the heavy representational lifting.

Inference benchmarks: 12.1ms per window, versus 9.6ms (small) and 21.7ms (medium). For a batch of 100 recordings, this translates to approximately 6 minutes total inference versus 5 minutes (small) or 11 minutes (medium). The 25% slowdown relative to the small preset is acceptable given the capacity gain.

This decision refines the scaling guidance in [[model size should scale with labeled dataset size to balance underfitting and overfitting]], which defined three discrete tiers. The mid-C config demonstrates that interpolating between tiers is practical when the dataset size falls near a boundary. The three-block architecture template from [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] is preserved -- only the filter widths and dense head size change.

The confidence is "likely" rather than "proven" because the capacity advantage over the small preset has not yet been empirically validated on this specific dataset. The conditions field specifies the validation required: compare val loss curves between mid-C and small after the matched-windows retrain completes.

---

Source:
- CNN retrain plan (docs/plans/CNN_RETRAIN_PLAN.md) -- matched-windows architecture selection
- Inference benchmarks from local profiling during retrain planning

Relevant Notes:
- [[model size should scale with labeled dataset size to balance underfitting and overfitting]] -- the three-tier scaling principle that this decision refines by interpolating between small and medium
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- the architectural template preserved by mid-C (same 3-block structure, different filter widths)
- [[model size growth versus available labeled data at each training milestone]] -- documents the tension between capacity and data availability that motivated this choice
- [[recording-level splits reduce effective training set size but prevent data leakage]] -- effective diversity is lower than raw sample count, supporting a conservative capacity choice
- [[class weight boosting biases toward recall at the cost of precision]] -- the training loss weighting interacts with model capacity; mid-C provides more capacity to learn precision without sacrificing the recall bias

Topics:
- [[classification]]
- [[training-methodology]]
