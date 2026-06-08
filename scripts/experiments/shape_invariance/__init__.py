"""Shape-invariance benchmark package (handoff: usv_shape_invariance_handoff.md).

Foundation for testing by-construction time/frequency-invariant USV contour
representations against the incumbent registration-Euclidean and the soft-DTW
bar, using the STANDING human-anchored kNN-purity gate.

Modules
-------
loader   : load_labeled() / load_full() -> arrays aligned to the labeled rows.
harness  : benchmark(...) -> 4-setting nested purity dict + k-sweep.
reversal : reversal_test(encode_fn, contours50) -> direction-blindness verdict.
io       : save_features(...) -> features/shape_invariance/{method}__{hash}.npy.
methods  : per-method encoders (m5_turning, ...).

All five SPEC functions (build_join, group_family, bootstrap_purity_ci,
bootstrap_purity_ci_from_distance, loo_knn_purity / knn_purity_from_distance)
live in scripts/experiments/eval_shape_human_anchored.py and are REUSED, never
reimplemented.
"""
