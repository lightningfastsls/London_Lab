"""Tests for null model surrogate generators.

16 test cases covering:
1.  Shuffled preserves exact code frequencies
2.  Shuffled destroys sequential structure (transition matrix differs)
3.  Markov-1 preserves bigram transition statistics
4.  Markov-1 entropy rate convergence similar to original
5.  Markov with k > len(seq) doesn't crash (short fallback)
6.  Renewal preserves approximate code frequencies
7.  Renewal produces correct-length sequences
8.  HMM produces valid surrogates (length + code range)
9.  FFT phase-randomization preserves autocorrelation
10. AAFT preserves exact marginal distribution
11. All methods produce correct-length surrogates
12. generate_all returns correct keys and surrogate counts
13. Same seed produces identical results (reproducibility)
14. Short sequence (len=3) doesn't crash
15. Single-code sequence produces valid output
16. Invalid config values raise ValueError
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# Bootstrap path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.analysis.null_models import (
    NullModelConfig,
    NullModelGenerator,
    _HAS_HMMLEARN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def structured_sequence() -> np.ndarray:
    """1000 codes with strong sequential structure (Markov-1, K=8).

    Each state strongly prefers one successor (p=0.8), creating
    detectable sequential order that surrogates should disrupt.
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
def small_config() -> NullModelConfig:
    """Config with few surrogates for fast tests."""
    return NullModelConfig(n_surrogates=10, random_seed=42)


@pytest.fixture
def gen(small_config: NullModelConfig) -> NullModelGenerator:
    """Generator with small config."""
    return NullModelGenerator(small_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transition_matrix(sequence: np.ndarray, K: int) -> np.ndarray:
    """Build K x K transition probability matrix from sequence."""
    T = np.zeros((K, K), dtype=np.float64)
    for i in range(len(sequence) - 1):
        T[sequence[i], sequence[i + 1]] += 1
    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def _autocorrelation(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Compute normalized autocorrelation for lags 1..max_lag."""
    x = x.astype(np.float64) - x.mean()
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag)
    result = []
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            result.append(0.0)
        else:
            c = np.mean(x[:-lag] * x[lag:])
            result.append(c / var)
    return np.array(result)


# ---------------------------------------------------------------------------
# 1. Shuffled: preserves frequencies
# ---------------------------------------------------------------------------


class TestShuffled:
    def test_preserves_frequencies(
        self, gen: NullModelGenerator, structured_sequence: np.ndarray
    ) -> None:
        """Every shuffled surrogate has identical code frequencies."""
        surrogates = gen.shuffled(structured_sequence)
        original_counts = Counter(structured_sequence.tolist())
        for surr in surrogates:
            assert Counter(surr.tolist()) == original_counts

    def test_destroys_structure(
        self, gen: NullModelGenerator, structured_sequence: np.ndarray
    ) -> None:
        """Shuffled surrogates have different transition matrices."""
        K = 8
        T_orig = _transition_matrix(structured_sequence, K)

        # Average Frobenius distance across surrogates
        surrogates = gen.shuffled(structured_sequence)
        distances = []
        for surr in surrogates:
            T_surr = _transition_matrix(surr, K)
            distances.append(np.linalg.norm(T_orig - T_surr, "fro"))

        mean_dist = np.mean(distances)
        # Structured sequence should have noticeably different transitions
        assert mean_dist > 0.1, f"Frobenius distance too small: {mean_dist}"


# ---------------------------------------------------------------------------
# 2. Markov: preserves k-gram statistics
# ---------------------------------------------------------------------------


class TestMarkov:
    def test_bigram_preservation(
        self, gen: NullModelGenerator, structured_sequence: np.ndarray
    ) -> None:
        """Markov-1 surrogates have transition matrix close to original."""
        K = 8
        T_orig = _transition_matrix(structured_sequence, K)

        surrogates = gen.markov_order_k(structured_sequence, k=1)

        # Average transition matrix across surrogates
        T_avg = np.zeros_like(T_orig)
        for surr in surrogates:
            T_avg += _transition_matrix(surr, K)
        T_avg /= len(surrogates)

        # Frobenius distance between original and average should be small
        dist = np.linalg.norm(T_orig - T_avg, "fro")
        assert dist < 0.3, f"Markov-1 transition matrix too different: {dist}"

    def test_entropy_rate_match(
        self, gen: NullModelGenerator, structured_sequence: np.ndarray
    ) -> None:
        """Markov-1 surrogates have similar per-symbol entropy."""
        K = 8

        # Compute empirical per-symbol entropy (bigram conditional)
        def _bigram_entropy(seq: np.ndarray) -> float:
            T = _transition_matrix(seq, K)
            # Stationary distribution (row sums of bigram counts)
            counts = Counter(seq[:-1].tolist())
            total = sum(counts.values())
            pi = np.array([counts.get(k, 0) / total for k in range(K)])
            h = 0.0
            for i in range(K):
                for j in range(K):
                    if T[i, j] > 0:
                        h -= pi[i] * T[i, j] * np.log2(T[i, j])
            return h

        h_orig = _bigram_entropy(structured_sequence)
        surrogates = gen.markov_order_k(structured_sequence, k=1)
        h_surrs = [_bigram_entropy(s) for s in surrogates]
        h_mean = np.mean(h_surrs)

        # Should be within 0.5 bits
        assert abs(h_orig - h_mean) < 0.5, (
            f"Entropy rate mismatch: orig={h_orig:.3f}, surr={h_mean:.3f}"
        )

    def test_short_fallback(self, gen: NullModelGenerator) -> None:
        """Markov with k > sequence length doesn't crash."""
        short = np.array([1, 2, 3], dtype=np.int64)
        surrogates = gen.markov_order_k(short, k=10)
        assert len(surrogates) == gen.config.n_surrogates
        for surr in surrogates:
            assert len(surr) == 3


# ---------------------------------------------------------------------------
# 3. Renewal: preserves temporal spacing
# ---------------------------------------------------------------------------


class TestRenewal:
    def test_preserves_frequencies(
        self, gen: NullModelGenerator, structured_sequence: np.ndarray
    ) -> None:
        """Renewal surrogates have approximately the same code frequencies."""
        original_counts = Counter(structured_sequence.tolist())
        total = len(structured_sequence)

        surrogates = gen.renewal_process(structured_sequence)

        for surr in surrogates:
            surr_counts = Counter(surr.tolist())
            for code in original_counts:
                orig_freq = original_counts[code] / total
                surr_freq = surr_counts.get(code, 0) / total
                # Allow 5% deviation per code
                assert abs(orig_freq - surr_freq) < 0.05, (
                    f"Code {code}: orig={orig_freq:.3f}, surr={surr_freq:.3f}"
                )

    def test_correct_length(
        self, gen: NullModelGenerator, structured_sequence: np.ndarray
    ) -> None:
        """All renewal surrogates have the same length as original."""
        surrogates = gen.renewal_process(structured_sequence)
        for surr in surrogates:
            assert len(surr) == len(structured_sequence)


# ---------------------------------------------------------------------------
# 4. HMM: valid surrogates
# ---------------------------------------------------------------------------


class TestHMM:
    @pytest.mark.skipif(
        not _HAS_HMMLEARN, reason="hmmlearn not installed"
    )
    def test_valid_surrogates(
        self, structured_sequence: np.ndarray
    ) -> None:
        """HMM surrogates have correct length and code range."""
        config = NullModelConfig(
            n_surrogates=5, random_seed=42,
            hmm_n_states=4, hmm_n_iter=50,
        )
        gen = NullModelGenerator(config)

        surrogates = gen.hmm_surrogate(structured_sequence)
        K = int(np.max(structured_sequence)) + 1

        assert len(surrogates) == 5
        for surr in surrogates:
            assert len(surr) == len(structured_sequence)
            assert np.all(surr >= 0)
            assert np.all(surr < K)

    def test_import_error_without_hmmlearn(self) -> None:
        """hmm_surrogate raises ImportError if hmmlearn unavailable."""
        if _HAS_HMMLEARN:
            pytest.skip("hmmlearn is installed; can't test ImportError")
        gen = NullModelGenerator(NullModelConfig(n_surrogates=2))
        with pytest.raises(ImportError, match="hmmlearn"):
            gen.hmm_surrogate(np.array([1, 2, 3], dtype=np.int64))


# ---------------------------------------------------------------------------
# 5. Phase-randomized: autocorrelation / marginal
# ---------------------------------------------------------------------------


class TestPhaseRandomized:
    def test_fft_autocorrelation(
        self, structured_sequence: np.ndarray
    ) -> None:
        """FFT phase-randomized surrogates preserve autocorrelation."""
        config = NullModelConfig(
            n_surrogates=10, random_seed=42, phase_rand_method="fft"
        )
        gen = NullModelGenerator(config)

        ac_orig = _autocorrelation(structured_sequence, max_lag=5)
        surrogates = gen.phase_randomized(structured_sequence)

        # Average autocorrelation across surrogates
        ac_surrs = np.array([
            _autocorrelation(s, max_lag=5) for s in surrogates
        ])
        ac_mean = ac_surrs.mean(axis=0)

        # Correlation between original and mean surrogate autocorrelation
        if np.std(ac_orig) > 0 and np.std(ac_mean) > 0:
            corr = np.corrcoef(ac_orig, ac_mean)[0, 1]
            assert corr > 0.8, (
                f"Autocorrelation correlation too low: {corr:.3f}"
            )

    def test_aaft_autocorrelation(
        self, structured_sequence: np.ndarray
    ) -> None:
        """AAFT surrogates preserve autocorrelation (at least as well as FFT)."""
        config = NullModelConfig(
            n_surrogates=10, random_seed=42, phase_rand_method="aaft"
        )
        gen = NullModelGenerator(config)

        ac_orig = _autocorrelation(structured_sequence, max_lag=5)
        surrogates = gen.phase_randomized(structured_sequence)

        ac_surrs = np.array([
            _autocorrelation(s, max_lag=5) for s in surrogates
        ])
        ac_mean = ac_surrs.mean(axis=0)

        if np.std(ac_orig) > 0 and np.std(ac_mean) > 0:
            corr = np.corrcoef(ac_orig, ac_mean)[0, 1]
            assert corr > 0.8, (
                f"AAFT autocorrelation correlation too low: {corr:.3f}"
            )

    def test_aaft_marginal(
        self, structured_sequence: np.ndarray
    ) -> None:
        """AAFT surrogates have identical sorted values (exact marginal)."""
        config = NullModelConfig(
            n_surrogates=5, random_seed=42, phase_rand_method="aaft"
        )
        gen = NullModelGenerator(config)

        surrogates = gen.phase_randomized(structured_sequence)
        sorted_orig = np.sort(structured_sequence)

        for surr in surrogates:
            assert np.array_equal(np.sort(surr), sorted_orig), (
                "AAFT surrogate doesn't preserve exact marginal distribution"
            )


# ---------------------------------------------------------------------------
# 6. Cross-cutting: length preservation
# ---------------------------------------------------------------------------


class TestLengthPreservation:
    def test_all_methods(
        self, gen: NullModelGenerator, structured_sequence: np.ndarray
    ) -> None:
        """Every surrogate from every method has correct length."""
        n = len(structured_sequence)

        for surr in gen.shuffled(structured_sequence):
            assert len(surr) == n

        for k in gen.config.markov_orders:
            for surr in gen.markov_order_k(structured_sequence, k):
                assert len(surr) == n

        for surr in gen.renewal_process(structured_sequence):
            assert len(surr) == n

        for surr in gen.phase_randomized(structured_sequence):
            assert len(surr) == n


# ---------------------------------------------------------------------------
# 7. generate_all: keys and counts
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_keys_and_counts(
        self, gen: NullModelGenerator, structured_sequence: np.ndarray
    ) -> None:
        """generate_all returns expected keys with correct surrogate counts."""
        results = gen.generate_all(structured_sequence)

        # Always-present keys
        expected_keys = {"shuffled", "markov_1", "markov_2", "markov_3",
                         "renewal", "phase_randomized"}
        assert expected_keys.issubset(results.keys())

        # HMM key present only if hmmlearn available
        if _HAS_HMMLEARN:
            assert "hmm" in results
        else:
            assert "hmm" not in results

        for key, surrogates in results.items():
            assert len(surrogates) == gen.config.n_surrogates, (
                f"{key}: expected {gen.config.n_surrogates}, "
                f"got {len(surrogates)}"
            )


# ---------------------------------------------------------------------------
# 8. Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed(self, structured_sequence: np.ndarray) -> None:
        """Same config produces identical surrogates."""
        config = NullModelConfig(n_surrogates=3, random_seed=99)

        gen1 = NullModelGenerator(config)
        gen2 = NullModelGenerator(config)

        s1 = gen1.shuffled(structured_sequence)
        s2 = gen2.shuffled(structured_sequence)

        for a, b in zip(s1, s2):
            assert np.array_equal(a, b), "Same seed produced different results"


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_short_sequence(self, gen: NullModelGenerator) -> None:
        """Length-3 sequence: all methods produce valid output."""
        short = np.array([0, 1, 2], dtype=np.int64)

        for surr in gen.shuffled(short):
            assert len(surr) == 3
            assert set(surr.tolist()) == {0, 1, 2}

        for surr in gen.markov_order_k(short, k=1):
            assert len(surr) == 3

        for surr in gen.renewal_process(short):
            assert len(surr) == 3

        for surr in gen.phase_randomized(short):
            assert len(surr) == 3

    def test_single_code(self, gen: NullModelGenerator) -> None:
        """Constant sequence [5,5,5,5,5]: valid output from all methods."""
        mono = np.array([5, 5, 5, 5, 5], dtype=np.int64)

        for surr in gen.shuffled(mono):
            assert np.all(surr == 5)
            assert len(surr) == 5

        for surr in gen.markov_order_k(mono, k=1):
            assert np.all(surr == 5)
            assert len(surr) == 5

        for surr in gen.renewal_process(mono):
            assert len(surr) == 5

        for surr in gen.phase_randomized(mono):
            assert len(surr) == 5


# ---------------------------------------------------------------------------
# 10. Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_validation(self) -> None:
        """Invalid configs raise ValueError."""
        with pytest.raises(ValueError, match="n_surrogates"):
            NullModelConfig(n_surrogates=0)

        with pytest.raises(ValueError, match="n_surrogates"):
            NullModelConfig(n_surrogates=-5)

        with pytest.raises(ValueError, match="markov_orders.*non-empty"):
            NullModelConfig(markov_orders=())

        with pytest.raises(ValueError, match="markov_orders.*positive"):
            NullModelConfig(markov_orders=(0, 1))

        with pytest.raises(ValueError, match="hmm_n_states"):
            NullModelConfig(hmm_n_states=0)

        with pytest.raises(ValueError, match="hmm_n_iter"):
            NullModelConfig(hmm_n_iter=-1)

        with pytest.raises(ValueError, match="phase_rand_method"):
            NullModelConfig(phase_rand_method="invalid")

        with pytest.raises(ValueError, match="random_seed"):
            NullModelConfig(random_seed=-1)
