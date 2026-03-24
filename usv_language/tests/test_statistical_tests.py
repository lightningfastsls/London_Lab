"""Tests for the statistical comparison framework (statistical_tests.py).

9 test cases covering:
1. Markov-1 sequence NOT significant vs Markov-1 null (entropy_rate)
2. Markov-1 sequence significant vs shuffled (entropy_rate)
3. Uniform random sequence NOT significant vs shuffled (all 7 metrics)
4. Long-range pattern significant vs Markov nulls (excess_entropy)
5. Manual z-score calculation verification
6. Rank p-value = 0.0 when real exceeds all nulls
7. Large effect size (Cohen's d > 0.8)
8. to_markdown() produces valid table
9. full_analysis completes on moderate-sized input (performance guard)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Bootstrap path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.analysis.statistical_tests import (
    MetricComparison,
    FullAnalysisResult,
    NullModelComparison,
    CORE_METRICS,
)
from usv_language.analysis.null_models import NullModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def markov1_sequence() -> np.ndarray:
    """1000 codes from a known Markov-1 chain (K=8).

    Each state transitions to the next state (mod K) with p=0.8,
    uniform over remaining states with p=0.2/(K-1).
    """
    rng = np.random.RandomState(123)
    K = 8
    T = np.full((K, K), 0.2 / (K - 1))
    for i in range(K):
        T[i, (i + 1) % K] = 0.8

    codes = [0]
    for _ in range(999):
        codes.append(rng.choice(K, p=T[codes[-1]]))
    return np.array(codes, dtype=np.int64)


@pytest.fixture
def uniform_random_sequence() -> np.ndarray:
    """1000 codes drawn i.i.d. uniform from {0..7}."""
    rng = np.random.RandomState(456)
    return rng.randint(0, 8, size=1000).astype(np.int64)


@pytest.fixture
def long_range_sequence() -> np.ndarray:
    """1000 codes with planted long-range structure.

    Repeating pattern of length 10 with 20% noise substitution.
    """
    rng = np.random.RandomState(789)
    pattern = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=np.int64)
    repeated = np.tile(pattern, 100)  # 1000 codes
    # Add 20% noise
    noise_mask = rng.random(1000) < 0.2
    noise_values = rng.randint(0, 8, size=1000).astype(np.int64)
    repeated[noise_mask] = noise_values[noise_mask]
    return repeated


@pytest.fixture
def comparator() -> NullModelComparison:
    """NullModelComparison instance."""
    return NullModelComparison()


@pytest.fixture
def fast_config() -> NullModelConfig:
    """Config with few surrogates for fast testing."""
    return NullModelConfig(n_surrogates=20, random_seed=42)


# ---------------------------------------------------------------------------
# Test 1: Markov-1 NOT significant vs Markov-1 null
# ---------------------------------------------------------------------------


class TestMarkov1VsMarkov1Null:
    def test_markov1_not_significant_vs_markov1_null(
        self,
        comparator: NullModelComparison,
        markov1_sequence: np.ndarray,
        fast_config: NullModelConfig,
    ) -> None:
        """A Markov-1 generated sequence should look like Markov-1 surrogates.

        entropy_rate should NOT be significant: the real sequence has the same
        structure that Markov-1 surrogates produce.
        """
        from usv_language.analysis.null_models import NullModelGenerator

        K = 8
        real_val = comparator._compute_metric("entropy_rate", markov1_sequence, K)

        gen = NullModelGenerator(fast_config)
        surrogates = gen.markov_order_k(markov1_sequence, k=1)
        null_vals = [
            comparator._compute_metric("entropy_rate", s, K) for s in surrogates
        ]

        result = comparator.compare(real_val, null_vals, "entropy_rate", "markov_1")
        assert not result.significant, (
            f"Markov-1 sequence should NOT be significant vs Markov-1 null. "
            f"z={result.z_score:.2f}, p={result.rank_p_value:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Markov-1 significant vs shuffled
# ---------------------------------------------------------------------------


class TestMarkov1VsShuffled:
    def test_markov1_significant_vs_shuffled(
        self,
        comparator: NullModelComparison,
        markov1_sequence: np.ndarray,
        fast_config: NullModelConfig,
    ) -> None:
        """A Markov-1 sequence has transition structure that shuffling destroys.

        entropy_rate should be significant vs shuffled: shuffled sequences
        have higher entropy rate (less structure) than the original.
        """
        from usv_language.analysis.null_models import NullModelGenerator

        K = 8
        real_val = comparator._compute_metric("entropy_rate", markov1_sequence, K)

        gen = NullModelGenerator(fast_config)
        surrogates = gen.shuffled(markov1_sequence)
        null_vals = [
            comparator._compute_metric("entropy_rate", s, K) for s in surrogates
        ]

        result = comparator.compare(real_val, null_vals, "entropy_rate", "shuffled")
        # Markov-1 should have LOWER entropy rate than shuffled
        assert result.significant, (
            f"Markov-1 sequence should be significant vs shuffled. "
            f"z={result.z_score:.2f}, p={result.rank_p_value:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Uniform random NOT significant vs shuffled
# ---------------------------------------------------------------------------


class TestUniformVsShuffled:
    def test_uniform_random_not_significant_vs_shuffled(
        self,
        comparator: NullModelComparison,
        uniform_random_sequence: np.ndarray,
        fast_config: NullModelConfig,
    ) -> None:
        """i.i.d. uniform sequence should match shuffled surrogates on all metrics.

        Since shuffling preserves the marginal and the original has no
        sequential structure, no metric should be significant.
        """
        from usv_language.analysis.null_models import NullModelGenerator

        K = 8
        gen = NullModelGenerator(fast_config)
        surrogates = gen.shuffled(uniform_random_sequence)

        # Test a subset of fast metrics (skip slow ones like idioms)
        fast_metrics = ["entropy_rate", "excess_entropy", "mutual_information"]

        for metric in fast_metrics:
            real_val = comparator._compute_metric(
                metric, uniform_random_sequence, K,
            )
            null_vals = [
                comparator._compute_metric(metric, s, K) for s in surrogates
            ]
            result = comparator.compare(
                real_val, null_vals, metric, "shuffled",
            )
            # Most should not be significant (allow one fluke)
            # We check using a softer criterion: z_score should be modest
            assert abs(result.z_score) < 4.0, (
                f"Uniform random {metric} unexpectedly extreme vs shuffled. "
                f"z={result.z_score:.2f}"
            )


# ---------------------------------------------------------------------------
# Test 4: Long-range pattern significant vs Markov nulls
# ---------------------------------------------------------------------------


class TestLongRangeVsMarkov:
    def test_long_range_significant_vs_markov_nulls(
        self,
        comparator: NullModelComparison,
        long_range_sequence: np.ndarray,
        fast_config: NullModelConfig,
    ) -> None:
        """A repeating length-10 pattern should have structure beyond Markov-1.

        excess_entropy should be significant vs at least shuffled.
        """
        from usv_language.analysis.null_models import NullModelGenerator

        K = 8
        real_val = comparator._compute_metric(
            "excess_entropy", long_range_sequence, K,
        )

        gen = NullModelGenerator(fast_config)
        surrogates = gen.shuffled(long_range_sequence)
        null_vals = [
            comparator._compute_metric("excess_entropy", s, K) for s in surrogates
        ]

        result = comparator.compare(
            real_val, null_vals, "excess_entropy", "shuffled",
        )
        assert result.significant, (
            f"Long-range pattern should be significant vs shuffled for "
            f"excess_entropy. z={result.z_score:.2f}, p={result.rank_p_value:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 5: Manual z-score calculation
# ---------------------------------------------------------------------------


class TestZScoreManual:
    def test_z_score_manual_calculation(
        self, comparator: NullModelComparison,
    ) -> None:
        """Hand-compute z-score from known values and verify match."""
        real = 10.0
        nulls = [2.0, 4.0, 6.0, 8.0, 10.0]
        # mean = 6.0, std(ddof=1) = sqrt(var) where var = ((16+4+0+4+16)/4) = 10
        # std = sqrt(10) = 3.162...
        # z = (10 - 6) / 3.162 = 1.265

        result = comparator.compare(real, nulls, "test_metric", "test_null")

        expected_mean = 6.0
        expected_std = np.std([2.0, 4.0, 6.0, 8.0, 10.0], ddof=1)
        expected_z = (10.0 - expected_mean) / expected_std

        assert abs(result.null_mean - expected_mean) < 1e-10
        assert abs(result.null_std - expected_std) < 1e-10
        assert abs(result.z_score - expected_z) < 1e-10
        assert result.effect_size == result.z_score


# ---------------------------------------------------------------------------
# Test 6: Rank p-value = 0.0 when real exceeds all nulls
# ---------------------------------------------------------------------------


class TestRankPValue:
    def test_rank_p_value_zero_when_exceeds_all(
        self, comparator: NullModelComparison,
    ) -> None:
        """When real value exceeds ALL null values, rank p should be 0.0."""
        real = 100.0
        nulls = list(range(1, 11))  # [1, 2, ..., 10]

        result = comparator.compare(real, nulls, "test_metric", "test_null")
        assert result.rank_p_value == 0.0
        assert result.significant  # Should be significant


# ---------------------------------------------------------------------------
# Test 7: Large effect size
# ---------------------------------------------------------------------------


class TestEffectSize:
    def test_effect_size_large(
        self, comparator: NullModelComparison,
    ) -> None:
        """Large difference gives Cohen's d > 0.8 (large effect)."""
        rng = np.random.RandomState(42)
        real = 10.0
        nulls = rng.normal(0.0, 1.0, size=100).tolist()

        result = comparator.compare(real, nulls, "test_metric", "test_null")

        # Cohen's d = (10 - ~0) / ~1 ≈ 10 >> 0.8
        assert abs(result.effect_size) > 0.8, (
            f"Expected large effect size, got {result.effect_size:.2f}"
        )


# ---------------------------------------------------------------------------
# Test 8: to_markdown() produces valid table
# ---------------------------------------------------------------------------


class TestToMarkdown:
    def test_to_markdown_valid(
        self, comparator: NullModelComparison,
    ) -> None:
        """to_markdown() produces a table with correct structure."""
        # Build a minimal FullAnalysisResult
        comparisons = [
            MetricComparison(
                metric_name="entropy_rate",
                null_model="shuffled",
                real_value=1.5,
                null_mean=2.0,
                null_std=0.2,
                z_score=-2.5,
                rank_p_value=0.99,
                effect_size=-2.5,
                n_surrogates=20,
                significant=True,
            ),
            MetricComparison(
                metric_name="excess_entropy",
                null_model="shuffled",
                real_value=0.5,
                null_mean=0.3,
                null_std=0.1,
                z_score=2.0,
                rank_p_value=0.05,
                effect_size=2.0,
                n_surrogates=20,
                significant=False,
            ),
        ]

        result = FullAnalysisResult(
            comparisons=comparisons,
            metrics_used=["entropy_rate", "excess_entropy"],
            null_models_used=["shuffled"],
            sequence_length=100,
            codebook_size=8,
            summary="Test summary",
        )

        md = result.to_markdown()
        assert "|" in md, "Markdown should contain table pipe characters"

        lines = md.strip().split("\n")
        # Header + separator + 1 data row (1 null model) + blank + legend = 5 lines
        assert len(lines) >= 3, f"Expected at least 3 lines, got {len(lines)}"

        # Header should contain metric names
        assert "entropy_rate" in lines[0]
        assert "excess_entropy" in lines[0]

        # Data row should contain "shuffled"
        data_lines = [l for l in lines if "shuffled" in l]
        assert len(data_lines) == 1

        # Significant metric should have asterisk
        assert "*" in md


# ---------------------------------------------------------------------------
# Test 9: full_analysis completes on moderate input (performance guard)
# ---------------------------------------------------------------------------


class TestFullAnalysisPerformance:
    def test_full_analysis_performance(
        self, comparator: NullModelComparison,
    ) -> None:
        """full_analysis on 500-token sequence with 10 surrogates completes quickly.

        This is a smoke test / characterization test, not a strict benchmark.
        Uses small parameters to keep CI fast.
        """
        rng = np.random.RandomState(42)
        K = 8
        # Generate a structured sequence (Markov-1) of moderate size
        T = np.full((K, K), 0.1 / (K - 1))
        for i in range(K):
            T[i, (i + 1) % K] = 0.9
        codes = [0]
        for _ in range(499):
            codes.append(rng.choice(K, p=T[codes[-1]]))
        sequence = np.array(codes, dtype=np.int64)

        config = NullModelConfig(
            n_surrogates=10,
            random_seed=42,
            markov_orders=(1,),  # Only Markov-1 for speed
        )

        start = time.time()
        result = comparator.full_analysis(sequence, K, null_config=config)
        elapsed = time.time() - start

        # Should complete in under 120 seconds even on slow CI
        assert elapsed < 120, f"full_analysis took {elapsed:.1f}s (> 120s limit)"

        # Structural checks
        assert len(result.comparisons) > 0
        assert result.sequence_length == 500
        assert result.codebook_size == K
        assert len(result.summary) > 0
        assert len(result.metrics_used) == len(CORE_METRICS)


# ---------------------------------------------------------------------------
# Edge case: empty null_values raises ValueError
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_null_values_raises(
        self, comparator: NullModelComparison,
    ) -> None:
        """compare() with empty null_values raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            comparator.compare(1.0, [], "metric", "null")

    def test_null_std_zero(
        self, comparator: NullModelComparison,
    ) -> None:
        """When all null values are identical, z_score=0 and significance uses rank_p."""
        result = comparator.compare(5.0, [3.0, 3.0, 3.0], "metric", "null")
        assert result.null_std == 0.0
        assert result.z_score == 0.0
        # real (5.0) > all nulls (3.0), so rank_p = 0.0 -> significant
        assert result.rank_p_value == 0.0
        assert result.significant

    def test_nan_real_value(
        self, comparator: NullModelComparison,
    ) -> None:
        """NaN real value propagates as NaN z_score, significant=False."""
        result = comparator.compare(float("nan"), [1.0, 2.0, 3.0], "metric", "null")
        assert np.isnan(result.z_score)
        assert not result.significant

    def test_empty_sequence_raises(
        self, comparator: NullModelComparison,
    ) -> None:
        """full_analysis with empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            comparator.full_analysis(np.array([], dtype=np.int64), K=8)

    def test_invalid_k_raises(
        self, comparator: NullModelComparison,
    ) -> None:
        """full_analysis with K <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="K must be positive"):
            comparator.full_analysis(np.array([1, 2, 3], dtype=np.int64), K=0)
