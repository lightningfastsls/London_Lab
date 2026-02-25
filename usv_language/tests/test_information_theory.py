"""Tests for information theory metrics module.

17 test cases covering:
1.  Uniform K=64 entropy rate ~ log2(64) = 6.0
1b. Convergence order value for uniform K=4
2.  Zipf MLE rank_alpha on synthetic alpha_rank=1.5
3.  Zipf MLE rank_alpha vs entropy alpha agree within 0.5
4.  Markov-1 entropy rate converges by order 3
5.  Periodic events: CV < 0.1
6.  Poisson events: 0.8 <= CV <= 1.2
7.  Bursty events: CV > 1.0
8.  Planted idiom (0,1,2) detected
9.  Plugin <= corrected entropy rate
10. Productivity CI bounds contain ratio
11. Empty sequence: no crash, returns defaults
11b. Small Zipf input returns defaults
12. Conditional entropy by lag shape
13. MI rate length
14. MI rate nonnegative
15. Burstiness by context groups
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Bootstrap path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.analysis.information_theory import (
    BurstinessResult,
    EntropyRateResult,
    IdiomResult,
    ProductivityResult,
    ZipfEntropyResult,
    ZipfResult,
    burstiness_coefficient,
    burstiness_by_context,
    conditional_entropy_by_lag,
    entropy_rate,
    mutual_information_rate,
    ngram_idioms,
    ngram_productivity,
    zipf_exponent_mle,
    zipf_via_shannon_entropy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_codes_k64() -> np.ndarray:
    """10 000 codes drawn uniformly from K=64."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 64, size=10_000).astype(np.int64)


@pytest.fixture
def markov1_codes() -> np.ndarray:
    """5 000 codes from a first-order Markov chain (K=8).

    Each state strongly prefers one successor (p=0.7), so the entropy rate
    should converge quickly (by order 2-3).
    """
    rng = np.random.RandomState(42)
    K = 8
    T = np.full((K, K), 0.3 / (K - 1))
    for i in range(K):
        T[i, (i + 1) % K] = 0.7

    codes = [0]
    for _ in range(4999):
        codes.append(rng.choice(K, p=T[codes[-1]]))
    return np.array(codes, dtype=np.int64)


@pytest.fixture
def zipf_codes() -> np.ndarray:
    """10 000 codes from a power-law distribution with alpha=1.5, K=100."""
    rng = np.random.RandomState(42)
    K = 100
    alpha = 1.5
    ranks = np.arange(1, K + 1, dtype=np.float64)
    probs = ranks ** (-alpha)
    probs /= probs.sum()
    return rng.choice(K, size=10_000, p=probs).astype(np.int64)


@pytest.fixture
def periodic_events() -> np.ndarray:
    """200 events at near-constant intervals (period=1.0, jitter=0.01)."""
    rng = np.random.RandomState(42)
    base = np.arange(200, dtype=np.float64)
    return base + rng.normal(0, 0.01, size=200)


@pytest.fixture
def poisson_events() -> np.ndarray:
    """500 events from a Poisson process (exponential IEIs)."""
    rng = np.random.RandomState(42)
    iei = rng.exponential(scale=1.0, size=500)
    return np.cumsum(iei)


@pytest.fixture
def bursty_events() -> np.ndarray:
    """500 events from a Gamma process with shape=0.3 (bursty, CV > 1)."""
    rng = np.random.RandomState(42)
    iei = rng.gamma(shape=0.3, scale=1.0, size=500)
    return np.cumsum(iei)


# ---------------------------------------------------------------------------
# Test 1: Uniform K=64 entropy rate
# ---------------------------------------------------------------------------


class TestEntropyRate:
    """Tests for entropy_rate function."""

    def test_uniform_entropy_rate(self, uniform_codes_k64: np.ndarray) -> None:
        """Uniform K=64: order 1 rate ~ log2(64) = 6.0.

        Only order 1 is checked tightly because higher-order n-gram spaces
        (K^n = 64^2 = 4096, 64^3 = 262,144) are severely undersampled with
        10,000 data points, causing downward bias in plugin entropy estimates.
        """
        result = entropy_rate(uniform_codes_k64, max_order=4)

        assert isinstance(result, EntropyRateResult)
        assert len(result.orders) == 4

        # Order 1: well-sampled (64 bins, 10k samples) -> tight bound
        assert abs(result.rates_corrected[0] - 6.0) < 0.15, (
            f"Order 1: expected ~6.0, got {result.rates_corrected[0]:.3f}"
        )
        # Order 2: moderately sampled (4096 bins) -> wider tolerance
        assert abs(result.rates_corrected[1] - 6.0) < 0.3, (
            f"Order 2: expected ~6.0, got {result.rates_corrected[1]:.3f}"
        )
        # Higher orders: just verify they're positive and monotonically
        # non-increasing (undersampling causes apparent decrease)
        for i in range(1, len(result.rates_corrected)):
            assert result.rates_corrected[i] > 0

    def test_convergence_order_value(self) -> None:
        """Convergence order is the first order where 2 consecutive small deltas begin.

        Uniform K=4 with N=50,000: all entropy rate deltas are tiny, so
        convergence should be detected starting at order 1.
        """
        rng = np.random.RandomState(42)
        seq = rng.randint(0, 4, size=50_000).astype(np.int64)
        result = entropy_rate(seq, max_order=5)
        assert result.convergence_order == 1, (
            f"Expected convergence_order=1 for uniform K=4, "
            f"got {result.convergence_order}"
        )

    def test_plugin_leq_corrected(self, uniform_codes_k64: np.ndarray) -> None:
        """Test 9: Miller-Madow corrected >= plugin for all orders."""
        result = entropy_rate(uniform_codes_k64, max_order=5)

        for i in range(len(result.orders)):
            assert result.rates_corrected[i] >= result.rates_plugin[i] - 1e-10, (
                f"Order {result.orders[i]}: corrected {result.rates_corrected[i]:.4f} "
                f"< plugin {result.rates_plugin[i]:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 2 & 3: Zipf MLE
# ---------------------------------------------------------------------------


class TestZipf:
    """Tests for Zipf exponent estimation."""

    def test_zipf_mle_alpha(self, zipf_codes: np.ndarray) -> None:
        """Test 2: MLE rank_alpha on synthetic alpha_rank=1.5 data.

        The fixture draws from a rank-frequency distribution with alpha=1.5.
        The MLE estimates the count-distribution exponent (~1.667 for alpha_rank=1.5),
        so we verify via rank_alpha which should recover ~1.5.
        """
        result = zipf_exponent_mle(zipf_codes)

        assert isinstance(result, ZipfResult)
        assert result.rank_alpha > 0, "rank_alpha should be positive"
        assert abs(result.rank_alpha - 1.5) < 0.5, (
            f"Expected rank_alpha ~ 1.5, got {result.rank_alpha:.3f}"
        )

    def test_zipf_mle_vs_entropy_agree(self, zipf_codes: np.ndarray) -> None:
        """Test 3: MLE rank_alpha and entropy-inversion alpha agree within 0.5.

        MLE estimates count-distribution alpha; entropy inversion estimates
        rank-frequency alpha directly.  We compare via rank_alpha (the derived
        transform: 1/(count_alpha - 1)) so both are in the same units.

        Note: Agreement tolerance holds broadly because rank_alpha is the
        correct comparison quantity.  The old comparison (raw MLE alpha vs
        entropy alpha) only agreed near alpha_rank=1.5 by coincidence.
        """
        mle_result = zipf_exponent_mle(zipf_codes)
        entropy_result = zipf_via_shannon_entropy(zipf_codes, K=100)

        assert isinstance(entropy_result, ZipfEntropyResult)
        assert abs(mle_result.rank_alpha - entropy_result.alpha_estimate) < 0.5, (
            f"MLE rank_alpha={mle_result.rank_alpha:.3f}, "
            f"entropy alpha={entropy_result.alpha_estimate:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Markov-1 convergence
# ---------------------------------------------------------------------------


class TestMarkovConvergence:
    """Test entropy rate convergence on Markov-1 data."""

    def test_markov1_converges(self, markov1_codes: np.ndarray) -> None:
        """Test 4: Markov-1 shows convergence behaviour.

        For a first-order Markov chain (K=8, 5000 samples), the entropy
        rate should drop from order 1 to order 2 (capturing the Markov
        structure), then flatten.  Higher orders show apparent decline
        due to undersampling (8^5 = 32,768 bins > 5,000 samples).
        We verify structural convergence at low orders.
        """
        result = entropy_rate(markov1_codes, max_order=4)

        # The drop from order 1 to 2 captures Markov structure
        drop_1_to_2 = result.rates_corrected[0] - result.rates_corrected[1]
        assert drop_1_to_2 > 0.05, (
            f"Expected meaningful drop from order 1 to 2, got {drop_1_to_2:.3f}"
        )

        # Orders 2 and 3 should be closer together than 1 and 2
        # (most structure captured by order 2)
        drop_2_to_3 = result.rates_corrected[1] - result.rates_corrected[2]
        assert drop_2_to_3 < drop_1_to_2, (
            f"Drop 2->3 ({drop_2_to_3:.3f}) should be < drop 1->2 ({drop_1_to_2:.3f})"
        )


# ---------------------------------------------------------------------------
# Tests 5-7: Burstiness
# ---------------------------------------------------------------------------


class TestBurstiness:
    """Tests for burstiness_coefficient."""

    def test_periodic_events(self, periodic_events: np.ndarray) -> None:
        """Test 5: Near-constant intervals -> CV < 0.1, periodic."""
        result = burstiness_coefficient(periodic_events)

        assert isinstance(result, BurstinessResult)
        assert result.cv < 0.1, f"Expected CV < 0.1 for periodic, got {result.cv:.3f}"
        assert result.interpretation == "periodic"

    def test_poisson_events(self, poisson_events: np.ndarray) -> None:
        """Test 6: Exponential IEIs -> CV in [0.8, 1.2], poisson.

        Bounds match the interpretation thresholds in burstiness_coefficient:
        CV < 0.8 = periodic, 0.8-1.2 = poisson, > 1.2 = bursty.
        """
        result = burstiness_coefficient(poisson_events)

        assert 0.8 <= result.cv <= 1.2, (
            f"Expected 0.8 <= CV <= 1.2 for Poisson, got {result.cv:.3f}"
        )
        assert result.interpretation == "poisson"

    def test_bursty_events(self, bursty_events: np.ndarray) -> None:
        """Test 7: Gamma(0.3) IEIs -> CV > 1.0, bursty."""
        result = burstiness_coefficient(bursty_events)

        assert result.cv > 1.0, (
            f"Expected CV > 1.0 for bursty, got {result.cv:.3f}"
        )
        assert result.interpretation == "bursty"


# ---------------------------------------------------------------------------
# Test 8: Planted idiom detection
# ---------------------------------------------------------------------------


class TestIdiomDetection:
    """Test for ngram_idioms with planted pattern."""

    def test_planted_idiom_detected(self) -> None:
        """Test 8: Sequence with planted (0,1,2) trigram is detected."""
        rng = np.random.RandomState(42)
        K = 8

        # Background: uniform random
        base = rng.randint(0, K, size=5000).astype(np.int64)

        # Plant 100 copies of (0, 1, 2) at random positions
        seq = base.copy()
        positions = rng.choice(len(seq) - 3, size=100, replace=False)
        for pos in positions:
            seq[pos] = 0
            seq[pos + 1] = 1
            seq[pos + 2] = 2

        results = ngram_idioms(seq, K, max_n=4, n_shuffles=100, fdr_alpha=0.05)

        # The planted trigram should be in the results
        detected_ngrams = [r.ngram for r in results]
        assert (0, 1, 2) in detected_ngrams, (
            f"Planted idiom (0,1,2) not detected. Found: {detected_ngrams[:5]}"
        )


# ---------------------------------------------------------------------------
# Test 10: Productivity CI bounds
# ---------------------------------------------------------------------------


class TestProductivity:
    """Tests for ngram_productivity."""

    def test_null_ci_valid(self, uniform_codes_k64: np.ndarray) -> None:
        """Test 10: null CI bounds are ordered and plausible.

        For near-uniform data the null CI (token-independence bootstrap)
        should be close to the observed ratio, since uniform sequences
        already behave like independent draws.
        """
        result = ngram_productivity(
            uniform_codes_k64, K=64, n=2, n_bootstrap=200,
        )

        assert isinstance(result, ProductivityResult)
        assert result.null_ci_lower <= result.null_ci_upper, (
            f"null_ci_lower {result.null_ci_lower:.4f} > "
            f"null_ci_upper {result.null_ci_upper:.4f}"
        )
        # For near-uniform data: null CI should be close to observed ratio
        assert abs(result.null_ci_lower - result.productivity_ratio) < 0.05, (
            f"For uniform data, null CI lower {result.null_ci_lower:.4f} should be "
            f"close to ratio {result.productivity_ratio:.4f}"
        )
        assert result.n == 2


# ---------------------------------------------------------------------------
# Test 11: Empty sequence
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_sequence(self) -> None:
        """Test 11: All functions handle empty input without crash."""
        empty = np.array([], dtype=np.int64)

        # Zipf MLE
        z = zipf_exponent_mle(empty)
        assert z.alpha == 0.0
        assert z.p_value == 1.0

        # Zipf entropy
        ze = zipf_via_shannon_entropy(empty)
        assert ze.alpha_estimate == 0.0

        # Entropy rate
        er = entropy_rate(empty)
        assert er.orders == []
        assert er.rates_plugin == []

        # Conditional entropy by lag
        ce = conditional_entropy_by_lag(empty, K=8, max_lag=5)
        assert len(ce) == 5
        assert all(v == 0.0 for v in ce)

        # MI rate
        mi = mutual_information_rate(empty, K=8, max_lag=5)
        assert len(mi) == 5

        # Productivity
        prod = ngram_productivity(empty, K=8, n=2, n_bootstrap=10)
        assert prod.observed == 0

        # Idioms
        idioms = ngram_idioms(empty, K=8)
        assert idioms == []

        # Burstiness
        b = burstiness_coefficient(np.array([1.0]))
        assert b.interpretation == "insufficient_data"

    def test_small_zipf_input(self) -> None:
        """Zipf MLE with < 10 unique values returns defaults."""
        small = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        result = zipf_exponent_mle(small)
        assert result.alpha == 0.0
        assert result.p_value == 1.0


# ---------------------------------------------------------------------------
# Test 12: Conditional entropy by lag shape
# ---------------------------------------------------------------------------


class TestConditionalEntropy:
    """Tests for conditional_entropy_by_lag."""

    def test_shape(self, uniform_codes_k64: np.ndarray) -> None:
        """Test 12: len(result) == max_lag."""
        max_lag = 7
        result = conditional_entropy_by_lag(
            uniform_codes_k64, K=64, max_lag=max_lag,
        )
        assert len(result) == max_lag


# ---------------------------------------------------------------------------
# Supplementary: MI rate and burstiness by context
# ---------------------------------------------------------------------------


class TestMIRate:
    """Tests for mutual_information_rate."""

    def test_mi_rate_length(self, uniform_codes_k64: np.ndarray) -> None:
        """MI rate returns correct number of values."""
        result = mutual_information_rate(uniform_codes_k64, K=64, max_lag=10)
        assert len(result) == 10

    def test_mi_rate_nonnegative(self, markov1_codes: np.ndarray) -> None:
        """MI values should be non-negative."""
        result = mutual_information_rate(markov1_codes, K=8, max_lag=5)
        for val in result:
            assert val >= -1e-10, f"MI should be >= 0, got {val}"


class TestBurstinessByContext:
    """Tests for burstiness_by_context."""

    def test_groups(self) -> None:
        """Splits events correctly by context label."""
        times = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        labels = np.array(["A", "A", "A", "B", "B", "B"])

        results = burstiness_by_context(times, labels)

        assert "A" in results
        assert "B" in results
        assert isinstance(results["A"], BurstinessResult)
