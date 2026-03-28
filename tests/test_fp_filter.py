"""Tests for fp_filter module — written by test-architect BEFORE implementation.

Module under test: src/usv_spectrogram/postprocessing/fp_filter.py
Depends on:        src/usv_spectrogram/postprocessing/event_features.py (module 15.4)

Both modules are absent at test-write time, so import errors are expected
at collection. All tests will fail (ImportError or AttributeError) until
both modules are implemented.

ROADMAP test plan coverage (section 15.5):
  1. "Filter trained on labeled events achieves F2 > 0.80 in cross-validation"
     -> test_cross_validated_f2_above_threshold
  2. "Feature importances are non-zero for at least 5 features"
     -> test_feature_importances_nonzero_count
  3. "save/load round-trip produces identical predictions"
     -> test_save_load_roundtrip_identical_predictions
  4. "Filter with all-positive training data doesn't crash (edge case)"
     -> test_fit_all_positive_labels_no_crash
  5. "Balanced class weights handle imbalanced event counts"
     -> test_balanced_weights_handle_imbalanced_data

Additional coverage (recurring gap patterns):
  - fit/predict/predict_proba interface contract  -> test_fit_predict_basic_interface
  - predict_proba output shape and bounds         -> test_predict_proba_shape_and_bounds
  - feature_importances() key set matches spec    -> test_feature_importances_keys
  - single-item training set edge case            -> test_fit_single_positive_single_negative
  - predict returns List[bool] not np.ndarray     -> test_predict_returns_list_of_bool
  - predict_proba probabilities sum to 1.0        -> test_predict_proba_rows_sum_to_one
  - filter rejects positive example it was trained on (sanity check) -> test_fitted_filter_separates_synthetic_classes
  - load returns FalsePositiveFilter instance     -> test_load_returns_correct_type
  - fit with mismatched lengths raises            -> test_fit_mismatched_lengths_raises
  - unfitted filter raises on predict             -> test_unfitted_filter_raises_on_predict
  - feature_importances() keys are all feature field names -> test_feature_importances_keys

Total: 16 tests (5 from ROADMAP, 11 additional)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Both of these imports will fail until the modules are implemented.
# That is expected — tests should fail at collection with ImportError.
from usv_spectrogram.postprocessing.event_features import EventFeatures  # noqa: E402
from usv_spectrogram.postprocessing.fp_filter import FalsePositiveFilter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — synthetic EventFeatures construction
#
# These helpers are sufficient for tests because fp_filter must work on
# EventFeatures dataclass instances regardless of how they were produced.
# We hand-craft two archetypes:
#   - "usv_like":   high peak prob, low std, high tonality, tonal freq
#   - "noise_like": low peak prob, high std, low tonality, broadband freq
# ---------------------------------------------------------------------------

_USV_FEATURE_KWARGS = dict(
    peak_probability=0.95,
    mean_probability=0.88,
    prob_std=0.05,
    prob_kurtosis=2.1,
    prob_roughness=0.02,
    duration_windows=12,
    tonality=0.55,
    mean_peak_freq_bin=85.0,   # 20000 + 85 * 586 ≈ 69.8 kHz (in-band USV)
    freq_range_bins=8.0,
    freq_modulation_rate=1.5,
    snr_db=18.0,
)

_NOISE_FEATURE_KWARGS = dict(
    peak_probability=0.45,
    mean_probability=0.38,
    prob_std=0.22,
    prob_kurtosis=5.8,
    prob_roughness=0.15,
    duration_windows=4,
    tonality=0.08,
    mean_peak_freq_bin=30.0,   # broadband / out-of-band
    freq_range_bins=55.0,
    freq_modulation_rate=12.0,
    snr_db=3.5,
)


def _make_usv_feature(**overrides) -> EventFeatures:
    """Return a USV-like EventFeatures instance."""
    kwargs = {**_USV_FEATURE_KWARGS, **overrides}
    return EventFeatures(**kwargs)


def _make_noise_feature(**overrides) -> EventFeatures:
    """Return a noise-like EventFeatures instance."""
    kwargs = {**_NOISE_FEATURE_KWARGS, **overrides}
    return EventFeatures(**kwargs)


def _make_balanced_dataset(n_per_class: int = 30):
    """Return (features, labels) with n_per_class examples of each class.

    USV events are label=True, noise events are label=False.
    Random jitter is added per-feature to prevent a perfectly singular matrix.
    """
    rng = np.random.default_rng(42)

    features: List[EventFeatures] = []
    labels: List[bool] = []

    for _ in range(n_per_class):
        # USV-like: jitter each float field slightly
        features.append(_make_usv_feature(
            peak_probability=float(np.clip(rng.normal(0.95, 0.03), 0.6, 1.0)),
            mean_probability=float(np.clip(rng.normal(0.88, 0.04), 0.5, 1.0)),
            prob_std=float(np.clip(rng.normal(0.05, 0.01), 0.0, 0.3)),
            tonality=float(np.clip(rng.normal(0.55, 0.08), 0.0, 1.0)),
        ))
        labels.append(True)

    for _ in range(n_per_class):
        # Noise-like: jitter each float field slightly
        features.append(_make_noise_feature(
            peak_probability=float(np.clip(rng.normal(0.45, 0.05), 0.0, 0.8)),
            mean_probability=float(np.clip(rng.normal(0.38, 0.05), 0.0, 0.7)),
            prob_std=float(np.clip(rng.normal(0.22, 0.03), 0.0, 0.5)),
            tonality=float(np.clip(rng.normal(0.08, 0.03), 0.0, 0.3)),
        ))
        labels.append(False)

    return features, labels


# ---------------------------------------------------------------------------
# Expected feature field names from the 15.4 spec
# ---------------------------------------------------------------------------

_EXPECTED_FEATURE_KEYS = {
    "peak_probability",
    "mean_probability",
    "prob_std",
    "prob_kurtosis",
    "prob_roughness",
    "duration_windows",
    "tonality",
    "mean_peak_freq_bin",
    "freq_range_bins",
    "freq_modulation_rate",
    "snr_db",
}


# ===========================================================================
# Test 1 (ROADMAP): fit/predict/predict_proba basic interface contract
# ===========================================================================

def test_fit_predict_basic_interface():
    """FalsePositiveFilter must accept List[EventFeatures] and List[bool] in fit,
    and return List[bool] from predict. Verifies the core API is correct."""
    features, labels = _make_balanced_dataset(n_per_class=20)
    filt = FalsePositiveFilter()

    # Must not raise
    filt.fit(features, labels)

    predictions = filt.predict(features)
    assert isinstance(predictions, list), "predict() must return a list"
    assert len(predictions) == len(features), "predict() length must match input"
    assert all(isinstance(p, bool) for p in predictions), (
        "predict() must return List[bool], not List[int] or np.ndarray"
    )


# ===========================================================================
# Test 2 (ROADMAP): predict_proba shape and bounds
# ===========================================================================

def test_predict_proba_shape_and_bounds():
    """predict_proba must return shape (n_samples, 2) with all values in [0, 1]."""
    features, labels = _make_balanced_dataset(n_per_class=15)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    proba = filt.predict_proba(features)

    assert isinstance(proba, np.ndarray), "predict_proba() must return np.ndarray"
    assert proba.shape == (len(features), 2), (
        f"Expected shape ({len(features)}, 2), got {proba.shape}"
    )
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0), (
        "All probabilities must be in [0, 1]"
    )


# ===========================================================================
# Test 3 (ROADMAP): predict_proba rows sum to 1.0
# ===========================================================================

def test_predict_proba_rows_sum_to_one():
    """Each row of predict_proba must sum to 1.0 (binary classification).

    Hand-computed: P(class=False) + P(class=True) = 1 for logistic regression.
    """
    features, labels = _make_balanced_dataset(n_per_class=15)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    proba = filt.predict_proba(features)
    row_sums = proba.sum(axis=1)

    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
        err_msg="predict_proba rows must sum to 1.0")


# ===========================================================================
# Test 4 (ROADMAP item 5): Balanced class weights on imbalanced data
# ===========================================================================

def test_balanced_weights_handle_imbalanced_data():
    """When positive class is rare (5:1 imbalance), balanced weights should
    still produce non-trivial recall — the filter must not just predict all-False.

    Verifies exit criterion: 'Balanced class weights handle imbalanced event counts'.
    """
    # 5 positives, 25 negatives (5:1 imbalance)
    rng = np.random.default_rng(7)
    features = []
    labels = []

    for _ in range(5):
        features.append(_make_usv_feature(
            peak_probability=float(np.clip(rng.normal(0.95, 0.02), 0.7, 1.0)),
            tonality=float(np.clip(rng.normal(0.60, 0.05), 0.3, 1.0)),
        ))
        labels.append(True)

    for _ in range(25):
        features.append(_make_noise_feature(
            peak_probability=float(np.clip(rng.normal(0.35, 0.05), 0.0, 0.6)),
            tonality=float(np.clip(rng.normal(0.07, 0.02), 0.0, 0.2)),
        ))
        labels.append(False)

    filt = FalsePositiveFilter()
    filt.fit(features, labels)
    predictions = filt.predict(features)

    # With balanced weights and separable classes, positive examples must be found
    predicted_positives = sum(1 for p in predictions if p is True)
    assert predicted_positives > 0, (
        "Filter with balanced weights must predict at least one positive "
        "even when positives are a minority class"
    )


# ===========================================================================
# Test 5 (ROADMAP item 1): Cross-validated F2 above threshold
# ===========================================================================

def test_cross_validated_f2_above_threshold():
    """Filter trained and evaluated on separable synthetic data should achieve
    F2 > 0.80. This tests the core discriminative capability requirement.

    Hand-computed: with clean USV vs noise separation, logistic regression
    should approach perfect recall on training data. F2 at perfect prediction:
      TP=40, FP=0, FN=0 → F2 = 5*40/(5*40+4*0+0) = 1.0

    We allow F2 >= 0.80 as a lower bound matching the spec exit criterion.
    Uses leave-one-out style: train on 58 samples, test on 2 held-out.
    """
    from usv_spectrogram.postprocessing.event_scoring import compute_f_beta

    features, labels = _make_balanced_dataset(n_per_class=30)

    # Deterministic shuffle so each sequential fold contains both classes.
    # Without this, the ordered data (30 True then 30 False) creates
    # single-class validation folds where F2 is undefined (TP=0, FN=0).
    rng = np.random.default_rng(0)
    indices = rng.permutation(len(features))
    features = [features[i] for i in indices]
    labels = [labels[i] for i in indices]

    # 5-fold cross-validation (manual split to avoid sklearn dependency in test)
    n = len(features)
    fold_size = n // 5
    f2_scores = []

    for fold in range(5):
        val_start = fold * fold_size
        val_end = val_start + fold_size

        train_f = features[:val_start] + features[val_end:]
        train_l = labels[:val_start] + labels[val_end:]
        val_f = features[val_start:val_end]
        val_l = labels[val_start:val_end]

        filt = FalsePositiveFilter()
        filt.fit(train_f, train_l)
        preds = filt.predict(val_f)

        tp = sum(1 for p, t in zip(preds, val_l) if p is True and t is True)
        fp = sum(1 for p, t in zip(preds, val_l) if p is True and t is False)
        fn = sum(1 for p, t in zip(preds, val_l) if p is False and t is True)
        f2 = compute_f_beta(tp, fp, fn, beta=2.0)
        f2_scores.append(f2)

    mean_f2 = np.mean(f2_scores)
    assert mean_f2 >= 0.80, (
        f"Cross-validated F2 {mean_f2:.3f} is below the 0.80 exit criterion. "
        "The filter must be discriminative enough on separable synthetic data."
    )


# ===========================================================================
# Test 6 (ROADMAP item 2): Feature importances — at least 5 non-zero
# ===========================================================================

def test_feature_importances_nonzero_count():
    """feature_importances() must return at least 5 non-zero coefficients.

    Spec: 'Feature importances are non-zero for at least 5 features.'
    LogisticRegression on multiple features with StandardScaler should never
    produce fewer than 5 non-zero weights when trained on all 11 features.
    """
    features, labels = _make_balanced_dataset(n_per_class=30)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    importances = filt.feature_importances()
    assert isinstance(importances, dict), "feature_importances() must return dict"

    nonzero_count = sum(1 for v in importances.values() if abs(v) > 1e-10)
    assert nonzero_count >= 5, (
        f"Expected >= 5 non-zero feature importances, got {nonzero_count}. "
        f"Full importances: {importances}"
    )


# ===========================================================================
# Test 7 (ROADMAP item 2): Feature importances — key set matches spec
# ===========================================================================

def test_feature_importances_keys():
    """feature_importances() must return a dict keyed by all 11 EventFeatures
    field names from the 15.4 spec. Missing or extra keys indicate the
    implementer mapped features incorrectly."""
    features, labels = _make_balanced_dataset(n_per_class=20)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    importances = filt.feature_importances()
    actual_keys = set(importances.keys())

    missing = _EXPECTED_FEATURE_KEYS - actual_keys
    extra = actual_keys - _EXPECTED_FEATURE_KEYS

    assert not missing, f"feature_importances() missing keys: {missing}"
    assert not extra, f"feature_importances() has unexpected keys: {extra}"


# ===========================================================================
# Test 8 (ROADMAP item 3): save/load round-trip — identical predictions
# ===========================================================================

def test_save_load_roundtrip_identical_predictions(tmp_path: Path):
    """Serialised and deserialised filter must produce bit-identical predictions.

    Spec: 'save/load round-trip produces identical predictions'.
    Tests both predict() and predict_proba() outputs.
    """
    features, labels = _make_balanced_dataset(n_per_class=25)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    original_preds = filt.predict(features)
    original_proba = filt.predict_proba(features)

    model_path = tmp_path / "fp_filter.pkl"
    filt.save(model_path)

    assert model_path.exists(), "save() must create a file at the given path"

    loaded = FalsePositiveFilter.load(model_path)
    loaded_preds = loaded.predict(features)
    loaded_proba = loaded.predict_proba(features)

    assert original_preds == loaded_preds, (
        "predict() results differ after save/load round-trip"
    )
    np.testing.assert_array_equal(original_proba, loaded_proba,
        err_msg="predict_proba() results differ after save/load round-trip")


# ===========================================================================
# Test 9: load() returns a FalsePositiveFilter instance
# ===========================================================================

def test_load_returns_correct_type(tmp_path: Path):
    """load() must return a FalsePositiveFilter, not a bare sklearn Pipeline."""
    features, labels = _make_balanced_dataset(n_per_class=10)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    path = tmp_path / "filter.pkl"
    filt.save(path)
    loaded = FalsePositiveFilter.load(path)

    assert isinstance(loaded, FalsePositiveFilter), (
        f"load() must return FalsePositiveFilter, got {type(loaded)}"
    )


# ===========================================================================
# Test 10 (ROADMAP item 4): All-positive training labels — no crash
# ===========================================================================

def test_fit_all_positive_labels_no_crash():
    """fit() with all labels=True must not raise an exception.

    Edge case: single-class training data. LogisticRegression with all
    positive labels is degenerate but must not crash — the spec explicitly
    calls this out as an edge case to handle.
    """
    features = [_make_usv_feature() for _ in range(20)]
    labels = [True] * 20

    filt = FalsePositiveFilter()
    filt.fit(features, labels)  # Must not raise

    # After fitting on all-positive, predict must still return a list
    preds = filt.predict(features)
    assert isinstance(preds, list)
    assert len(preds) == 20


# ===========================================================================
# Test 11: All-negative training labels — no crash
# ===========================================================================

def test_fit_all_negative_labels_no_crash():
    """fit() with all labels=False must not raise an exception."""
    features = [_make_noise_feature() for _ in range(20)]
    labels = [False] * 20

    filt = FalsePositiveFilter()
    filt.fit(features, labels)  # Must not raise

    preds = filt.predict(features)
    assert isinstance(preds, list)
    assert len(preds) == 20


# ===========================================================================
# Test 12: Single positive + single negative — minimal training set
# ===========================================================================

def test_fit_single_positive_single_negative():
    """fit() with exactly 1 positive and 1 negative must not crash.

    Verifies the minimum viable training set. The pipeline must not crash
    when degenerate splits produce trivially small training sets.
    """
    features = [_make_usv_feature(), _make_noise_feature()]
    labels = [True, False]

    filt = FalsePositiveFilter()
    filt.fit(features, labels)  # Must not raise

    preds = filt.predict(features)
    assert len(preds) == 2


# ===========================================================================
# Test 13: predict() returns Python list of bool, not np.ndarray
# ===========================================================================

def test_predict_returns_list_of_bool():
    """predict() must return List[bool] exactly as specified.

    A common greenwashing mistake is returning np.bool_ or np.ndarray,
    which passes duck-typing checks but breaks isinstance(x, bool) checks
    and JSON serialisation.
    """
    features, labels = _make_balanced_dataset(n_per_class=10)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    result = filt.predict(features[:4])

    assert isinstance(result, list), "predict() must return list, not np.ndarray"
    for val in result:
        assert type(val) is bool, (
            f"Each element must be Python bool, got {type(val)}: {val}"
        )


# ===========================================================================
# Test 14: Unfitted filter raises on predict
# ===========================================================================

def test_unfitted_filter_raises_on_predict():
    """Calling predict() before fit() must raise an informative exception.

    sklearn raises sklearn.exceptions.NotFittedError; the wrapper must not
    silently swallow it or return nonsense.
    """
    filt = FalsePositiveFilter()
    features = [_make_usv_feature()]

    with pytest.raises(Exception):
        filt.predict(features)


# ===========================================================================
# Test 15: Fitted filter separates clean synthetic classes
# ===========================================================================

def test_fitted_filter_separates_synthetic_classes():
    """After fitting on clearly separable USV vs noise examples, the filter
    must correctly classify each class on the training set with at least 80%
    accuracy. This is a sanity check, not a generalisation test.

    Hand-computed: USV features have peak_prob≈0.95, tonality≈0.55.
    Noise features have peak_prob≈0.45, tonality≈0.08.
    These two clusters are well-separated in feature space; any linear
    classifier should achieve near-perfect accuracy on this data.
    """
    features, labels = _make_balanced_dataset(n_per_class=30)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    preds = filt.predict(features)
    correct = sum(1 for p, t in zip(preds, labels) if p == t)
    accuracy = correct / len(labels)

    assert accuracy >= 0.80, (
        f"Filter achieves only {accuracy:.1%} accuracy on separable training data. "
        "A correctly implemented logistic regression should exceed 80%."
    )


# ===========================================================================
# Test 16: fit() with mismatched feature/label lengths raises
# ===========================================================================

def test_fit_mismatched_lengths_raises():
    """fit() must raise ValueError when features and labels have different lengths.

    This prevents silent bugs where a zip() silently truncates to the shorter
    sequence without warning.
    """
    features = [_make_usv_feature() for _ in range(10)]
    labels = [True] * 7  # Deliberate mismatch

    filt = FalsePositiveFilter()

    with pytest.raises((ValueError, AssertionError, Exception)):
        filt.fit(features, labels)


# ===========================================================================
# ADVERSARIAL TESTS — added by test-hardener (15.5)
#
# Gaps found:
#   A. Empty-list paths on fitted (non-constant) filter — WARNING 1 from review
#   B. Empty-list paths on constant-label filter
#   C. predict_proba column ordering (col 0 = FP, col 1 = USV)
#   D. save/load roundtrip of constant-label filter
#   E. feature_importances() on constant-label filter (zero coef path)
#   F. feature_importances() values are non-negative (docstring contract)
#   G. Large input array (100+ features) — no crash
#   H. All-identical feature values — StandardScaler zero-variance path
#   I. NaN / Inf in feature values — numerical corruption
#   J. _features_to_array empty shape is exactly (0, 11)
#   K. load() raises TypeError on wrong pickle type
#   L. Unfitted filter raises on predict_proba
#   M. Unfitted filter raises on feature_importances
#   N. constant-False predict_proba column assignment (col 0 = 1.0)
# ===========================================================================

from usv_spectrogram.postprocessing.fp_filter import _features_to_array  # noqa: E402


# ---------------------------------------------------------------------------
# A. Empty-list predict / predict_proba on a normally-fitted filter
#    (WARNING 1 from master-reviewer: _features_to_array([]) produced
#     shape (0,) pre-fix, crashing StandardScaler.transform)
# ---------------------------------------------------------------------------

def test_predict_empty_list_fitted_filter():
    """predict([]) on a normally-fitted (two-class) filter must return [].

    Regression guard for the WARNING 1 fix: empty input was crashing
    StandardScaler.transform before the (0, n_features) shape guard was added.
    """
    features, labels = _make_balanced_dataset(n_per_class=15)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    result = filt.predict([])

    assert result == [], f"predict([]) must return [], got {result!r}"


def test_predict_proba_empty_list_fitted_filter():
    """predict_proba([]) on a normally-fitted filter must return shape (0, 2).

    Regression guard for WARNING 1. The return value must be an ndarray
    with exactly two columns so downstream column-0 / column-1 accesses
    don't raise IndexError.
    """
    features, labels = _make_balanced_dataset(n_per_class=15)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    proba = filt.predict_proba([])

    assert isinstance(proba, np.ndarray), "predict_proba([]) must return np.ndarray"
    assert proba.shape == (0, 2), (
        f"predict_proba([]) must return shape (0, 2), got {proba.shape}"
    )


# ---------------------------------------------------------------------------
# B. Empty-list paths on constant-label filter
# ---------------------------------------------------------------------------

def test_predict_empty_list_constant_label_filter():
    """predict([]) on a constant-label (single-class) filter must return [].

    The constant-label path uses `[self._constant_label] * len(features)`.
    With len([]) == 0 this should return [] without touching sklearn at all.
    """
    features = [_make_usv_feature() for _ in range(10)]
    labels = [True] * 10

    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    result = filt.predict([])
    assert result == [], f"predict([]) on constant-label filter must return [], got {result!r}"


def test_predict_proba_empty_list_constant_label_filter():
    """predict_proba([]) on a constant-label filter must return shape (0, 2)."""
    features = [_make_noise_feature() for _ in range(10)]
    labels = [False] * 10

    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    proba = filt.predict_proba([])

    assert proba.shape == (0, 2), (
        f"predict_proba([]) on constant-label filter must return shape (0, 2), "
        f"got {proba.shape}"
    )


# ---------------------------------------------------------------------------
# C. predict_proba column ordering
#    Col 0 = P(False / FP), Col 1 = P(True / USV).
#    Verified by training on perfectly separable data and checking that
#    USV-like inputs receive high values in column 1 (not column 0).
# ---------------------------------------------------------------------------

def test_predict_proba_column_ordering_usv_in_col1():
    """For a clear USV-like input, P(USV) must be in column 1 and be > 0.5.

    Column ordering convention: col 0 = P(false positive), col 1 = P(real USV).
    This test would catch a transposed output that swaps the columns.
    """
    features, labels = _make_balanced_dataset(n_per_class=30)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    usv_inputs = [_make_usv_feature() for _ in range(5)]
    proba = filt.predict_proba(usv_inputs)

    # Column 1 should dominate for USV-like inputs
    assert proba.shape[1] == 2, "predict_proba must have exactly 2 columns"
    assert np.all(proba[:, 1] > 0.5), (
        f"Column 1 (P(USV)) must be > 0.5 for USV-like inputs, "
        f"got col1={proba[:, 1]}"
    )


def test_predict_proba_column_ordering_noise_in_col0():
    """For a clear noise-like input, P(FP) must be in column 0 and be > 0.5.

    Complements the USV column test — both must be correct for the ordering
    convention to hold.
    """
    features, labels = _make_balanced_dataset(n_per_class=30)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    noise_inputs = [_make_noise_feature() for _ in range(5)]
    proba = filt.predict_proba(noise_inputs)

    assert np.all(proba[:, 0] > 0.5), (
        f"Column 0 (P(FP)) must be > 0.5 for noise-like inputs, "
        f"got col0={proba[:, 0]}"
    )


# ---------------------------------------------------------------------------
# N. predict_proba column ordering on constant-label filters
# ---------------------------------------------------------------------------

def test_predict_proba_constant_true_label_uses_col1():
    """Constant-True filter: col 1 must be 1.0, col 0 must be 0.0.

    The fit() code sets `col = 1 if self._constant_label else 0`.
    This test makes that logic observable and guards against a flip.
    """
    features = [_make_usv_feature() for _ in range(5)]
    labels = [True] * 5
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    test_features = [_make_usv_feature(), _make_noise_feature()]
    proba = filt.predict_proba(test_features)

    assert proba.shape == (2, 2)
    np.testing.assert_array_equal(proba[:, 0], [0.0, 0.0],
        err_msg="constant-True filter: col 0 (P(FP)) must be 0.0")
    np.testing.assert_array_equal(proba[:, 1], [1.0, 1.0],
        err_msg="constant-True filter: col 1 (P(USV)) must be 1.0")


def test_predict_proba_constant_false_label_uses_col0():
    """Constant-False filter: col 0 must be 1.0, col 1 must be 0.0.

    This is the more dangerous misassignment: a filter trained on all-FP
    data should assign P(FP)=1.0 in column 0, not column 1.
    """
    features = [_make_noise_feature() for _ in range(5)]
    labels = [False] * 5
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    test_features = [_make_usv_feature(), _make_noise_feature()]
    proba = filt.predict_proba(test_features)

    assert proba.shape == (2, 2)
    np.testing.assert_array_equal(proba[:, 0], [1.0, 1.0],
        err_msg="constant-False filter: col 0 (P(FP)) must be 1.0")
    np.testing.assert_array_equal(proba[:, 1], [0.0, 0.0],
        err_msg="constant-False filter: col 1 (P(USV)) must be 0.0")


# ---------------------------------------------------------------------------
# D. save/load roundtrip of a constant-label filter
#    Existing roundtrip test uses a normally-fitted filter only.
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_constant_label_filter(tmp_path: Path):
    """Constant-label filter must survive save/load and return identical results.

    The constant-label path stores state in `_constant_label` and partially
    sets up the pipeline's scaler + dummy classifier. This tests that the
    entire object is faithfully pickled (not just the pipeline).
    """
    features = [_make_usv_feature() for _ in range(15)]
    labels = [True] * 15

    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    original_preds = filt.predict(features)
    original_proba = filt.predict_proba(features)

    path = tmp_path / "constant_filter.pkl"
    filt.save(path)

    loaded = FalsePositiveFilter.load(path)

    assert isinstance(loaded, FalsePositiveFilter)
    assert loaded._constant_label is True, (
        "Loaded constant-label filter must preserve _constant_label=True"
    )
    assert loaded.predict(features) == original_preds
    np.testing.assert_array_equal(loaded.predict_proba(features), original_proba)


# ---------------------------------------------------------------------------
# E. feature_importances() on constant-label filter
# ---------------------------------------------------------------------------

def test_feature_importances_constant_label_filter():
    """feature_importances() on a constant-label filter must return all zeros.

    The fit() code sets clf.coef_ = np.zeros((1, n_features)) for the
    constant-label case. This test makes that invariant visible.
    """
    features = [_make_usv_feature() for _ in range(10)]
    labels = [True] * 10
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    importances = filt.feature_importances()

    assert set(importances.keys()) == _EXPECTED_FEATURE_KEYS, (
        "feature_importances() must return all expected keys even for "
        "constant-label filter"
    )
    for name, val in importances.items():
        assert val == 0.0, (
            f"Constant-label filter: importance for '{name}' must be 0.0, got {val}"
        )


# ---------------------------------------------------------------------------
# F. feature_importances() values are non-negative (absolute value contract)
# ---------------------------------------------------------------------------

def test_feature_importances_all_nonnegative():
    """feature_importances() must return absolute coefficient values (>= 0).

    The docstring states 'absolute logistic regression coefficients'. This
    test verifies no negative value slips through (e.g. if abs() is removed).
    """
    features, labels = _make_balanced_dataset(n_per_class=20)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    importances = filt.feature_importances()

    for name, val in importances.items():
        assert val >= 0.0, (
            f"feature_importances() must return non-negative values; "
            f"'{name}' = {val}"
        )


# ---------------------------------------------------------------------------
# G. Large input array — 100 features, no crash
# ---------------------------------------------------------------------------

def test_predict_large_input_no_crash():
    """predict() and predict_proba() must handle 100+ features without crashing.

    Verifies no stack overflow, memory error, or silent truncation.
    """
    rng = np.random.default_rng(99)
    features = []
    labels = []

    for _ in range(60):
        features.append(_make_usv_feature(
            peak_probability=float(np.clip(rng.normal(0.95, 0.03), 0.6, 1.0)),
            tonality=float(np.clip(rng.normal(0.55, 0.08), 0.0, 1.0)),
        ))
        labels.append(True)

    for _ in range(60):
        features.append(_make_noise_feature(
            peak_probability=float(np.clip(rng.normal(0.40, 0.05), 0.0, 0.7)),
            tonality=float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.3)),
        ))
        labels.append(False)

    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    # 120-item inference batch
    preds = filt.predict(features)
    proba = filt.predict_proba(features)

    assert len(preds) == 120
    assert proba.shape == (120, 2)


# ---------------------------------------------------------------------------
# H. All-identical feature values — StandardScaler zero-variance path
# ---------------------------------------------------------------------------

def test_fit_predict_all_identical_features():
    """fit/predict must not crash when all features are identical within each class.

    StandardScaler will produce zero variance for constant columns and
    must not raise a division-by-zero or set output to NaN.
    """
    # Both classes share identical values within each group, but groups differ
    usv_feat = _make_usv_feature()
    noise_feat = _make_noise_feature()

    features = [usv_feat] * 15 + [noise_feat] * 15
    labels = [True] * 15 + [False] * 15

    filt = FalsePositiveFilter()
    filt.fit(features, labels)  # Must not raise

    preds = filt.predict(features)
    proba = filt.predict_proba(features)

    assert len(preds) == 30
    assert proba.shape == (30, 2)
    # Probabilities must not be NaN even with zero-variance features
    assert not np.any(np.isnan(proba)), (
        "predict_proba() must not produce NaN even for constant feature columns"
    )


# ---------------------------------------------------------------------------
# I. NaN / Inf in feature values
# ---------------------------------------------------------------------------

def test_predict_nan_features_does_not_silently_succeed():
    """A feature vector containing NaN must either raise or produce NaN output.

    The filter must not silently produce a confident non-NaN prediction when
    given corrupted input — that would hide upstream bugs. Either raising an
    exception or propagating NaN is acceptable; returning a confident bool
    from NaN input without warning is not.

    Note: sklearn's LogisticRegression with StandardScaler typically propagates
    NaN through transform, resulting in NaN probabilities. We accept either
    NaN propagation or an explicit exception.
    """
    features, labels = _make_balanced_dataset(n_per_class=15)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    nan_feat = _make_usv_feature(peak_probability=float("nan"))

    try:
        proba = filt.predict_proba([nan_feat])
        # If it doesn't raise, NaN must propagate (not silently become 0.5)
        assert np.any(np.isnan(proba)), (
            "NaN input must either raise or produce NaN output — "
            f"got confident non-NaN result: {proba}"
        )
    except (ValueError, RuntimeError, FloatingPointError):
        pass  # Explicit rejection is also acceptable


def test_predict_inf_features_does_not_silently_succeed():
    """A feature vector containing Inf must either raise or produce inf/nan output.

    Same contract as the NaN test — corrupted input must not silently return
    a confident, seemingly-valid probability.
    """
    features, labels = _make_balanced_dataset(n_per_class=15)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    inf_feat = _make_usv_feature(snr_db=float("inf"))

    try:
        proba = filt.predict_proba([inf_feat])
        has_non_finite = np.any(~np.isfinite(proba))
        assert has_non_finite, (
            "Inf input must either raise or produce non-finite output — "
            f"got apparently valid result: {proba}"
        )
    except (ValueError, RuntimeError, FloatingPointError, OverflowError):
        pass  # Explicit rejection is also acceptable


# ---------------------------------------------------------------------------
# J. _features_to_array empty shape is exactly (0, 11)
# ---------------------------------------------------------------------------

def test_features_to_array_empty_shape():
    """_features_to_array([]) must return shape (0, 11), not (0,) or (0, 0).

    The StandardScaler inside the pipeline will call X.shape[1] on its input.
    Shape (0,) raises IndexError; (0, 11) is the correct sentinel that lets
    transform produce a valid (0, 11) output without touching the data.
    """
    import dataclasses
    from usv_spectrogram.postprocessing.event_features import EventFeatures

    n_fields = len(dataclasses.fields(EventFeatures))
    result = _features_to_array([])

    assert result.ndim == 2, (
        f"_features_to_array([]) must return 2-D array, got ndim={result.ndim}"
    )
    assert result.shape == (0, n_fields), (
        f"_features_to_array([]) must return shape (0, {n_fields}), "
        f"got {result.shape}"
    )


def test_features_to_array_single_item_shape():
    """_features_to_array([one_feature]) must return shape (1, 11)."""
    import dataclasses
    from usv_spectrogram.postprocessing.event_features import EventFeatures

    n_fields = len(dataclasses.fields(EventFeatures))
    result = _features_to_array([_make_usv_feature()])

    assert result.shape == (1, n_fields), (
        f"_features_to_array([one_item]) must return shape (1, {n_fields}), "
        f"got {result.shape}"
    )


def test_features_to_array_field_order_matches_feature_names():
    """_features_to_array must preserve field order matching _FEATURE_NAMES.

    If the astuple() order ever diverges from _FEATURE_NAMES, feature
    importances will be mapped to the wrong names. Verify by checking that
    the first column of the array equals peak_probability for all rows.
    """
    from usv_spectrogram.postprocessing.fp_filter import _FEATURE_NAMES

    usv = _make_usv_feature(peak_probability=0.99)
    noise = _make_noise_feature(peak_probability=0.11)
    arr = _features_to_array([usv, noise])

    peak_prob_col_idx = _FEATURE_NAMES.index("peak_probability")
    assert arr[0, peak_prob_col_idx] == pytest.approx(0.99), (
        "peak_probability value must appear in the correct column of the array"
    )
    assert arr[1, peak_prob_col_idx] == pytest.approx(0.11), (
        "peak_probability value must appear in the correct column of the array"
    )


# ---------------------------------------------------------------------------
# K. load() raises TypeError on wrong pickle type
# ---------------------------------------------------------------------------

def test_load_raises_type_error_on_wrong_pickle(tmp_path: Path):
    """load() must raise TypeError when the pickle file contains a non-FalsePositiveFilter.

    Protects against accidentally loading a bare sklearn Pipeline or a
    different versioned object that happens to be in the same file.
    """
    wrong_object = {"not": "a filter"}
    path = tmp_path / "wrong.pkl"
    with open(path, "wb") as f:
        pickle.dump(wrong_object, f)

    with pytest.raises(TypeError):
        FalsePositiveFilter.load(path)


# ---------------------------------------------------------------------------
# L. Unfitted filter raises on predict_proba
# ---------------------------------------------------------------------------

def test_unfitted_filter_raises_on_predict_proba():
    """predict_proba() before fit() must raise an informative exception.

    The existing test only checks predict(). The same guard covers
    predict_proba(), but it must be verified independently.
    """
    filt = FalsePositiveFilter()
    with pytest.raises(Exception):
        filt.predict_proba([_make_usv_feature()])


# ---------------------------------------------------------------------------
# M. Unfitted filter raises on feature_importances
# ---------------------------------------------------------------------------

def test_unfitted_filter_raises_on_feature_importances():
    """feature_importances() before fit() must raise an informative exception.

    The RuntimeError guard in feature_importances() mirrors the one in
    predict() and predict_proba(). Needs its own test.
    """
    filt = FalsePositiveFilter()
    with pytest.raises(Exception):
        filt.feature_importances()


# ---------------------------------------------------------------------------
# Additional: save() creates intermediate directories when they don't exist
# ---------------------------------------------------------------------------

def test_save_creates_parent_directories(tmp_path: Path):
    """save() must create parent directories that do not yet exist.

    The implementation calls path.parent.mkdir(parents=True, exist_ok=True).
    This tests that a deeply nested path doesn't raise FileNotFoundError.
    """
    features, labels = _make_balanced_dataset(n_per_class=10)
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    deep_path = tmp_path / "models" / "v1" / "filter.pkl"
    assert not deep_path.parent.exists(), "Setup: parent directory must not exist yet"

    filt.save(deep_path)  # Must not raise

    assert deep_path.exists(), "save() must create the file at the nested path"


# ---------------------------------------------------------------------------
# Additional: predict() on constant-label filter returns Python bool, not np.bool_
# ---------------------------------------------------------------------------

def test_constant_label_predict_returns_python_bool():
    """predict() on a constant-label filter must return Python bool, not np.bool_.

    The constant-label path uses `[self._constant_label] * len(features)`.
    _constant_label is set via `bool(y[0])` which should be a Python bool.
    Verify the type contract is met for this code path too.
    """
    features = [_make_usv_feature() for _ in range(5)]
    labels = [True] * 5
    filt = FalsePositiveFilter()
    filt.fit(features, labels)

    preds = filt.predict([_make_noise_feature()])
    assert len(preds) == 1
    assert type(preds[0]) is bool, (
        f"constant-label predict() must return Python bool, got {type(preds[0])}"
    )
