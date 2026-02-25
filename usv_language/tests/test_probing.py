"""Tests for the probing framework (probing.py).

18 test cases covering:
1.  Perfect linear encoding — R² > 0.95
2.  Random noise encoding — R² < 0.1
3.  Deeper layers = stronger signal — scores monotonically increase
4.  Perfectly separable classification — accuracy > 0.95
5.  Random classification labels — accuracy in [0.35, 0.65]
6.  CV fold count — len(fold_scores) == n_folds
7.  Heatmap shape — (n_layers, n_properties)
8.  best_overall_layer correctness — selects highest-scoring layer
9.  Config: invalid probe_type — ValueError
10. Config: n_folds < 2 — ValueError
11. Pipeline with missing properties — warns, runs on available subset
12. Subsampling respects max_samples — result.n_samples <= max_samples
13. Sentinel filter — -1.0 values excluded before regression
14. Too few samples — graceful zero result when n_samples < n_folds
15. MLP probe end-to-end — MLP regression on perfect data yields R² > 0.8
16. Three-class classification — StratifiedKFold with 3 classes works
+   plot_probing_heatmap smoke test
+   plot_layer_comparison smoke test

All tests use synthetic numpy data with deterministic seeds.
No real WAV files needed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Bootstrap
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from usv_language.analysis.probing import (
    ProbingConfig,
    ProbingResult,
    ProbingAnalysisResult,
    ProbingExperiment,
    ProbingAnalysisPipeline,
    plot_probing_heatmap,
    plot_layer_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> ProbingConfig:
    return ProbingConfig(probe_types=("linear",), n_folds=3)


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


def _make_linear_data(
    n: int = 200, d: int = 32, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """X with y = X @ w + small noise (perfect linear relationship)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = rng.randn(d)
    y = X @ w + rng.randn(n) * 0.01
    return X, y


def _make_noise_data(
    n: int = 200, d: int = 32, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """X and y are independent random — no linear relationship."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randn(n)
    return X, y


def _make_separable_classes(
    n: int = 200, d: int = 32, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Two classes with large margin separation."""
    rng = np.random.RandomState(seed)
    half = n // 2
    X0 = rng.randn(half, d) - 5.0
    X1 = rng.randn(n - half, d) + 5.0
    X = np.vstack([X0, X1])
    y = np.array(["class_a"] * half + ["class_b"] * (n - half))
    return X, y


def _make_random_classes(
    n: int = 200, d: int = 32, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random class labels with no relationship to X."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.choice(["cat", "dog"], size=n)
    return X, y


# ---------------------------------------------------------------------------
# Test 1: Perfect linear encoding
# ---------------------------------------------------------------------------

def test_perfect_linear_encoding(default_config: ProbingConfig) -> None:
    """Ridge regression on a perfectly linear relationship should get R² > 0.95."""
    X, y = _make_linear_data(n=300)
    exp = ProbingExperiment(default_config)
    result = exp.run(X, y, "test_prop", layer=0, probe_type="linear", task_type="regression")

    assert result.score > 0.95, f"Expected R² > 0.95, got {result.score:.3f}"
    assert result.task_type == "regression"
    assert result.n_samples == 300


# ---------------------------------------------------------------------------
# Test 2: Random noise encoding
# ---------------------------------------------------------------------------

def test_random_noise_encoding(default_config: ProbingConfig) -> None:
    """Ridge on independent X, y should yield R² < 0.1 (near zero)."""
    X, y = _make_noise_data(n=300)
    exp = ProbingExperiment(default_config)
    result = exp.run(X, y, "noise", layer=0, probe_type="linear", task_type="regression")

    assert result.score < 0.1, f"Expected R² < 0.1, got {result.score:.3f}"


# ---------------------------------------------------------------------------
# Test 3: Deeper layers = stronger signal
# ---------------------------------------------------------------------------

def test_deeper_layers_stronger_signal(default_config: ProbingConfig) -> None:
    """When later layers have more signal, probe scores increase monotonically.

    We use d=8 (low dimensionality) so Ridge regression can find the
    signal even under moderate noise, and a wide noise range [2.0 -> 0.1]
    so the SNR difference between layers is unambiguous.
    """
    rng = np.random.RandomState(99)
    n, d = 500, 8
    w = rng.randn(d)

    # Same underlying signal for all layers — only noise varies
    X_signal = rng.randn(n, d)
    y = X_signal @ w  # target based on the underlying signal

    # Noise decreases: 2.0, 1.0, 0.5, 0.1  (wide SNR spread)
    noise_scales = [2.0, 1.0, 0.5, 0.1]
    layers = []
    for scale in noise_scales:
        X = X_signal + rng.randn(n, d) * scale
        layers.append(X)

    exp = ProbingExperiment(default_config)
    scores = []
    for layer_idx, X in enumerate(layers):
        result = exp.run(X, y, "synth", layer=layer_idx, probe_type="linear", task_type="regression")
        scores.append(result.score)

    # Later layers should score higher (monotonically increasing)
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1] - 0.05, (
            f"Layer {i} score ({scores[i]:.3f}) should be >= "
            f"layer {i-1} score ({scores[i-1]:.3f})"
        )


# ---------------------------------------------------------------------------
# Test 4: Perfectly separable classification
# ---------------------------------------------------------------------------

def test_perfectly_separable_classification(default_config: ProbingConfig) -> None:
    """Large-margin separated classes should achieve accuracy > 0.95."""
    X, y = _make_separable_classes(n=200)
    exp = ProbingExperiment(default_config)
    result = exp.run(X, y, "is_voiced", layer=0, probe_type="linear", task_type="classification")

    assert result.score > 0.95, f"Expected accuracy > 0.95, got {result.score:.3f}"
    assert result.task_type == "classification"


# ---------------------------------------------------------------------------
# Test 5: Random classification labels
# ---------------------------------------------------------------------------

def test_random_classification_labels(default_config: ProbingConfig) -> None:
    """Random binary labels should give accuracy near 0.5 (chance)."""
    X, y = _make_random_classes(n=300)
    exp = ProbingExperiment(default_config)
    result = exp.run(X, y, "random", layer=0, probe_type="linear", task_type="classification")

    assert 0.35 <= result.score <= 0.65, (
        f"Expected accuracy in [0.35, 0.65], got {result.score:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 6: CV fold count
# ---------------------------------------------------------------------------

def test_cv_fold_count() -> None:
    """fold_scores length should match n_folds configuration."""
    config = ProbingConfig(probe_types=("linear",), n_folds=7)
    X, y = _make_linear_data(n=200)
    exp = ProbingExperiment(config)
    result = exp.run(X, y, "test", layer=0, probe_type="linear", task_type="regression")

    assert len(result.fold_scores) == 7


# ---------------------------------------------------------------------------
# Test 7: Heatmap shape
# ---------------------------------------------------------------------------

def test_heatmap_shape() -> None:
    """heatmap_data should return (n_layers, n_properties)."""
    config = ProbingConfig(
        probe_types=("linear",),
        n_folds=3,
        regression_properties=("peak_frequency", "energy"),
        classification_properties=("is_voiced",),
    )
    n, d = 200, 16
    rng = np.random.RandomState(42)

    hidden_states = [rng.randn(n, d) for _ in range(4)]
    properties = {
        "peak_frequency": rng.randn(n),
        "energy": rng.randn(n),
        "is_voiced": rng.choice([True, False], size=n),
    }

    pipeline = ProbingAnalysisPipeline(config)
    result = pipeline.run(hidden_states, properties)
    mat = result.heatmap_data("linear")

    assert mat.shape == (4, 3), f"Expected (4, 3), got {mat.shape}"


# ---------------------------------------------------------------------------
# Test 8: best_overall_layer correctness
# ---------------------------------------------------------------------------

def test_best_overall_layer() -> None:
    """best_overall_layer should select the layer with highest mean score."""
    config = ProbingConfig(
        probe_types=("linear",),
        n_folds=3,
        regression_properties=("prop_a",),
        classification_properties=(),
    )
    rng = np.random.RandomState(123)
    n, d = 300, 16
    w = rng.randn(d)

    # Layer 2 gets the clean signal, others get noise
    hidden_states = [rng.randn(n, d) for _ in range(3)]
    X_clean = rng.randn(n, d)
    y = X_clean @ w + rng.randn(n) * 0.01
    hidden_states[2] = X_clean  # layer 2 = clean signal

    properties = {"prop_a": y}

    pipeline = ProbingAnalysisPipeline(config)
    result = pipeline.run(hidden_states, properties)

    assert result.best_overall_layer == 2, (
        f"Expected layer 2, got {result.best_overall_layer}"
    )


# ---------------------------------------------------------------------------
# Test 9: Config — invalid probe_type
# ---------------------------------------------------------------------------

def test_config_invalid_probe_type() -> None:
    """ProbingConfig should reject unknown probe types."""
    with pytest.raises(ValueError, match="Invalid probe_type"):
        ProbingConfig(probe_types=("linear", "transformer"))


# ---------------------------------------------------------------------------
# Test 10: Config — n_folds < 2
# ---------------------------------------------------------------------------

def test_config_nfolds_too_small() -> None:
    """ProbingConfig should reject n_folds < 2."""
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        ProbingConfig(n_folds=1)


# ---------------------------------------------------------------------------
# Test 11: Pipeline with missing properties
# ---------------------------------------------------------------------------

def test_pipeline_missing_properties_warns() -> None:
    """Pipeline should warn about missing properties and run on the subset."""
    config = ProbingConfig(
        probe_types=("linear",),
        n_folds=3,
        regression_properties=("peak_frequency", "nonexistent_prop"),
        classification_properties=(),
    )
    rng = np.random.RandomState(42)
    n, d = 100, 8
    hidden_states = [rng.randn(n, d)]
    properties = {"peak_frequency": rng.randn(n)}

    pipeline = ProbingAnalysisPipeline(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipeline.run(hidden_states, properties)

    # Should have warned about the missing property
    warn_msgs = [str(w.message) for w in caught]
    assert any("nonexistent_prop" in msg for msg in warn_msgs), (
        f"Expected warning about nonexistent_prop, got: {warn_msgs}"
    )

    # Should still produce results for the available property
    assert len(result.property_names) == 1
    assert result.property_names[0] == "peak_frequency"
    assert len(result.results) == 1


# ---------------------------------------------------------------------------
# Test 12: Subsampling respects max_samples
# ---------------------------------------------------------------------------

def test_subsampling_respects_max_samples() -> None:
    """When max_samples is set, result.n_samples should not exceed it."""
    config = ProbingConfig(
        probe_types=("linear",),
        n_folds=3,
        max_samples=50,
    )
    X, y = _make_linear_data(n=200)
    exp = ProbingExperiment(config)
    result = exp.run(X, y, "test", layer=0, probe_type="linear", task_type="regression")

    assert result.n_samples <= 50, (
        f"Expected n_samples <= 50, got {result.n_samples}"
    )


# ---------------------------------------------------------------------------
# Test 13: Sentinel filter for time_since_last_usv == -1.0
# ---------------------------------------------------------------------------

def test_sentinel_filtered_before_regression(default_config: ProbingConfig) -> None:
    """time_since_last_usv -1.0 sentinels must be excluded before regression."""
    rng = np.random.RandomState(42)
    n_valid = 150
    n_sentinel = 50
    d = 16
    X = rng.randn(n_valid + n_sentinel, d)
    y = np.concatenate([rng.uniform(0, 500, n_valid), np.full(n_sentinel, -1.0)])

    exp = ProbingExperiment(default_config)
    result = exp.run(
        X, y, "time_since_last_usv", layer=0,
        probe_type="linear", task_type="regression",
    )

    assert result.n_samples == n_valid, (
        f"Expected {n_valid} valid samples after filtering, got {result.n_samples}"
    )


# ---------------------------------------------------------------------------
# Test 14: Too few samples returns graceful zero result
# ---------------------------------------------------------------------------

def test_too_few_samples_graceful_return(default_config: ProbingConfig) -> None:
    """When fewer valid samples than n_folds, return score=0.0 gracefully."""
    rng = np.random.RandomState(42)
    # Only 2 valid samples, but n_folds=3 — can't cross-validate
    X = rng.randn(2, 8)
    y = rng.randn(2)
    exp = ProbingExperiment(default_config)
    result = exp.run(X, y, "tiny", layer=0, probe_type="linear", task_type="regression")

    assert result.score == 0.0
    assert result.n_samples == 2
    assert result.fold_scores == ()


# ---------------------------------------------------------------------------
# Test 15: MLP probe end-to-end
# ---------------------------------------------------------------------------

def test_mlp_probe_regression() -> None:
    """MLP probe on a perfectly linear relationship should get R² > 0.8.

    Uses small max_iter to keep the test fast.  The threshold is lower than
    the linear probe test (0.8 vs 0.95) because MLP convergence in few
    iterations is less precise than Ridge's closed-form solution.
    """
    config = ProbingConfig(probe_types=("mlp",), n_folds=3, mlp_max_iter=500)
    X, y = _make_linear_data(n=300, d=16)
    exp = ProbingExperiment(config)
    result = exp.run(X, y, "mlp_test", layer=0, probe_type="mlp", task_type="regression")

    assert result.score > 0.8, f"Expected MLP R² > 0.8, got {result.score:.3f}"
    assert result.probe_type == "mlp"
    assert len(result.fold_scores) == 3


# ---------------------------------------------------------------------------
# Test 16: Three-class classification with StratifiedKFold
# ---------------------------------------------------------------------------

def test_three_class_classification() -> None:
    """StratifiedKFold with 3 classes should not crash, even with imbalance.

    Simulates frequency_direction with rising/falling/flat — flat is rare
    (10% of samples).  Verifies that StratifiedKFold handles the 3-class
    case and produces a valid accuracy.
    """
    config = ProbingConfig(probe_types=("linear",), n_folds=3)
    rng = np.random.RandomState(42)
    n, d = 300, 16

    # 3 well-separated classes with imbalance: 45% / 45% / 10%
    n_rising = 135
    n_falling = 135
    n_flat = 30
    X0 = rng.randn(n_rising, d) + np.array([3.0] + [0.0] * (d - 1))
    X1 = rng.randn(n_falling, d) + np.array([-3.0] + [0.0] * (d - 1))
    X2 = rng.randn(n_flat, d)  # centered at origin
    X = np.vstack([X0, X1, X2])
    y = np.array(
        ["rising"] * n_rising + ["falling"] * n_falling + ["flat"] * n_flat
    )

    exp = ProbingExperiment(config)
    result = exp.run(
        X, y, "frequency_direction",
        layer=0, probe_type="linear", task_type="classification",
    )

    # With well-separated classes, accuracy should be high
    assert result.score > 0.8, (
        f"Expected 3-class accuracy > 0.8, got {result.score:.3f}"
    )
    assert result.n_samples == n
    assert len(result.fold_scores) == 3


# ---------------------------------------------------------------------------
# Bonus: Visualization smoke tests (no assertions on content, just no crash)
# ---------------------------------------------------------------------------

def test_plot_probing_heatmap_runs(tmp_path: Path) -> None:
    """plot_probing_heatmap should produce a figure without crashing."""
    config = ProbingConfig(
        probe_types=("linear",),
        n_folds=3,
        regression_properties=("prop_a",),
        classification_properties=(),
    )
    rng = np.random.RandomState(42)
    n, d = 100, 8

    hidden_states = [rng.randn(n, d) for _ in range(3)]
    properties = {"prop_a": rng.randn(n)}

    pipeline = ProbingAnalysisPipeline(config)
    result = pipeline.run(hidden_states, properties)

    out_path = str(tmp_path / "heatmap.png")
    fig = plot_probing_heatmap(result, "linear", output_path=out_path)
    assert Path(out_path).exists()
    plt.close(fig)


def test_plot_layer_comparison_runs(tmp_path: Path) -> None:
    """plot_layer_comparison should produce a figure without crashing."""
    config = ProbingConfig(
        probe_types=("linear",),
        n_folds=3,
        regression_properties=("prop_a",),
        classification_properties=(),
    )
    rng = np.random.RandomState(42)
    n, d = 100, 8

    hidden_states = [rng.randn(n, d) for _ in range(3)]
    properties = {"prop_a": rng.randn(n)}

    pipeline = ProbingAnalysisPipeline(config)
    result = pipeline.run(hidden_states, properties)

    out_path = str(tmp_path / "comparison.png")
    fig = plot_layer_comparison(result, output_path=out_path)
    assert Path(out_path).exists()
    plt.close(fig)
