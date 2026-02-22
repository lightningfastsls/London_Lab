"""Tests for Phase 8.4 analysis and interpretation tools.

14 test cases covering:
1.  AnalysisConfig validation
2.  decode_hidden_to_spectrogram shape
3.  decode_all_entries shape
4.  Zipf identifies power law
5.  Transition matrix properties
6.  Entropy rate decreases on structured data
7.  Code extraction valid range
8.  Concept injection shape
9.  N-gram short sequence edge cases
10. Conditional entropy bounds
11. Mutual information non-negative
12. Bigram productivity range
13. Concept scan shape
14. Chi-squared valid p-value
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Bootstrap
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.analysis.config import AnalysisConfig
from usv_language.analysis.transformer_suffix import (
    decode_hidden_to_spectrogram,
    inject_and_continue,
)
from usv_language.analysis import codebook_viz
from usv_language.analysis import sequence_analysis
from usv_language.analysis import concept_manipulation
from usv_language.analysis import context_analysis
from usv_language.analysis import compositionality
from usv_language.models.transformer import SpectrogramTransformer, TransformerConfig
from usv_language.models.vqvae import HiddenStateVQVAE, VQVAEConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_transformer() -> SpectrogramTransformer:
    """Small transformer for fast testing: 2 layers, d_model=64, n_freq=16."""
    config = TransformerConfig(
        n_freq=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ffn=128,
        max_seq_len=64,
        dropout=0.0,
    )
    model = SpectrogramTransformer(config)
    model.eval()
    return model


@pytest.fixture
def small_vqvae() -> HiddenStateVQVAE:
    """Small VQ-VAE: K=8, d_model=64, codebook_dim=32, linear encoder."""
    config = VQVAEConfig(
        d_model=64,
        codebook_size=8,
        codebook_dim=32,
        use_conv_encoder=False,
    )
    model = HiddenStateVQVAE(config)
    model.eval()
    return model


@pytest.fixture
def synthetic_codes() -> np.ndarray:
    """Synthetic code sequence from a structured Markov chain.

    Generates 5000 codes from K=8 with non-uniform transitions,
    so entropy rate should decrease with n-gram order.
    """
    rng = np.random.RandomState(42)
    K = 8
    # Structured transition matrix: each state prefers 2-3 successors
    T = np.zeros((K, K))
    for i in range(K):
        # Each state has 2-3 preferred successors
        successors = [(i + 1) % K, (i + 2) % K, (i + 3) % K]
        probs = rng.dirichlet([5, 3, 1])
        for s, p in zip(successors, probs):
            T[i, s] = p

    # Generate sequence
    codes = [0]
    for _ in range(4999):
        current = codes[-1]
        next_code = rng.choice(K, p=T[current])
        codes.append(next_code)

    return np.array(codes, dtype=np.int64)


@pytest.fixture
def synthetic_hidden_states() -> np.ndarray:
    """Random hidden states: (200, 64)."""
    rng = np.random.RandomState(42)
    return rng.randn(200, 64).astype(np.float32)


@pytest.fixture
def analysis_config() -> AnalysisConfig:
    """Analysis config matching small model dimensions."""
    return AnalysisConfig(n_freq=16, source_layer=1)


# ---------------------------------------------------------------------------
# Test 1: AnalysisConfig validation
# ---------------------------------------------------------------------------


class TestAnalysisConfig:
    """Test 1: AnalysisConfig defaults and error cases."""

    def test_valid_defaults(self):
        config = AnalysisConfig()
        assert config.n_exemplars == 10
        assert config.max_ngram_order == 8
        assert config.source_layer == 4
        assert config.n_freq == 170

    def test_invalid_n_exemplars(self):
        with pytest.raises(ValueError, match="n_exemplars"):
            AnalysisConfig(n_exemplars=0)

    def test_invalid_source_layer(self):
        with pytest.raises(ValueError, match="source_layer"):
            AnalysisConfig(source_layer=0)

    def test_invalid_freq_range(self):
        with pytest.raises(ValueError, match="freq_min_hz"):
            AnalysisConfig(freq_min_hz=120_000, freq_max_hz=20_000)


# ---------------------------------------------------------------------------
# Test 2: decode_hidden_to_spectrogram shape
# ---------------------------------------------------------------------------


def test_decode_hidden_to_spectrogram_shape(
    small_transformer: SpectrogramTransformer,
) -> None:
    """Transformer suffix: (1, 5, d_model=64) -> (1, 5, n_freq=16)."""
    h = torch.randn(1, 5, 64)
    with torch.no_grad():
        output = decode_hidden_to_spectrogram(small_transformer, h, start_layer=1)
    assert output.shape == (1, 5, 16), f"Expected (1, 5, 16), got {output.shape}"


# ---------------------------------------------------------------------------
# Test 3: decode_all_entries shape
# ---------------------------------------------------------------------------


def test_decode_all_entries_shape(
    small_transformer: SpectrogramTransformer,
    small_vqvae: HiddenStateVQVAE,
) -> None:
    """Codebook decode: (K=8, n_freq=16) output."""
    profiles = codebook_viz.decode_all_entries(
        small_transformer, small_vqvae, source_layer=1,
    )
    assert profiles.shape == (8, 16), f"Expected (8, 16), got {profiles.shape}"


# ---------------------------------------------------------------------------
# Test 4: Zipf identifies power law
# ---------------------------------------------------------------------------


def test_zipf_identifies_power_law() -> None:
    """Synthetic power-law distribution: alpha ~ -1.0, R² > 0.9."""
    rng = np.random.RandomState(42)
    # Generate Zipf-distributed codes
    # Zipf: P(k) proportional to k^{-alpha}
    alpha = 1.0
    K = 50
    probs = np.arange(1, K + 1, dtype=np.float64) ** (-alpha)
    probs /= probs.sum()

    codes = rng.choice(K, size=10000, p=probs)
    result = sequence_analysis.zipf_analysis(codes, min_count=5)

    # Alpha should be close to -1.0 (slope of log-log plot)
    assert abs(result["alpha"] - (-1.0)) < 0.3, (
        f"Expected alpha ~ -1.0, got {result['alpha']:.3f}"
    )
    assert result["r_squared"] > 0.9, (
        f"Expected R² > 0.9, got {result['r_squared']:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Transition matrix properties
# ---------------------------------------------------------------------------


def test_transition_matrix_properties(synthetic_codes: np.ndarray) -> None:
    """K×K matrix, rows sum to 1.0."""
    K = 8
    T = sequence_analysis.compute_transition_matrix(synthetic_codes, K)

    assert T.shape == (K, K), f"Expected ({K}, {K}), got {T.shape}"

    # Active rows should sum to 1.0
    row_sums = T.sum(axis=1)
    for i in range(K):
        if row_sums[i] > 0:
            assert abs(row_sums[i] - 1.0) < 1e-10, (
                f"Row {i} sums to {row_sums[i]}, expected 1.0"
            )


# ---------------------------------------------------------------------------
# Test 6: Entropy rate decreases
# ---------------------------------------------------------------------------


def test_entropy_rate_decreases(synthetic_codes: np.ndarray) -> None:
    """Rates should decrease on structured Markov chain."""
    rates = sequence_analysis.entropy_rate(synthetic_codes, max_order=5)

    assert len(rates) == 5
    # First rate should be higher than last (structure reduces uncertainty)
    assert rates[0] > rates[-1], (
        f"Entropy rate should decrease: {rates[0]:.3f} > {rates[-1]:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 7: Code extraction valid range
# ---------------------------------------------------------------------------


def test_code_extraction_valid_range(
    synthetic_hidden_states: np.ndarray,
    small_vqvae: HiddenStateVQVAE,
) -> None:
    """Codes in [0, K-1], correct length."""
    codes = sequence_analysis.extract_code_sequences(
        synthetic_hidden_states, small_vqvae,
    )
    K = small_vqvae.config.codebook_size

    assert len(codes) == len(synthetic_hidden_states)
    assert codes.min() >= 0
    assert codes.max() < K


# ---------------------------------------------------------------------------
# Test 8: Concept injection shape
# ---------------------------------------------------------------------------


def test_concept_injection_shape(
    small_transformer: SpectrogramTransformer,
    small_vqvae: HiddenStateVQVAE,
) -> None:
    """Injected spectrogram (S, n_freq), future (N, n_freq)."""
    n_freq = small_transformer.config.n_freq
    S = 10
    x = torch.randn(1, S, n_freq)
    n_future = 5

    result = concept_manipulation.concept_injection(
        small_transformer, small_vqvae, x,
        injection_position=3, codebook_index=0,
        source_layer=1, n_future_steps=n_future,
    )

    assert result["injected_output"].shape == (S, n_freq), (
        f"Expected ({S}, {n_freq}), got {result['injected_output'].shape}"
    )
    assert result["future_columns"].shape == (n_future, n_freq), (
        f"Expected ({n_future}, {n_freq}), got {result['future_columns'].shape}"
    )
    assert result["original_output"].shape == (S, n_freq)


# ---------------------------------------------------------------------------
# Test 9: N-gram short sequence
# ---------------------------------------------------------------------------


def test_ngram_short_sequence() -> None:
    """Empty for n > len, correct for n <= len."""
    codes = np.array([1, 2, 3], dtype=np.int64)

    # n > len should give empty
    ngrams_4 = sequence_analysis.extract_ngrams(codes, 4)
    assert len(ngrams_4) == 0

    # n == len should give exactly 1
    ngrams_3 = sequence_analysis.extract_ngrams(codes, 3)
    assert len(ngrams_3) == 1
    assert (1, 2, 3) in ngrams_3

    # n < len
    ngrams_2 = sequence_analysis.extract_ngrams(codes, 2)
    assert len(ngrams_2) == 2
    assert ngrams_2[(1, 2)] == 1
    assert ngrams_2[(2, 3)] == 1


# ---------------------------------------------------------------------------
# Test 10: Conditional entropy bounds
# ---------------------------------------------------------------------------


def test_conditional_entropy_bounds(synthetic_codes: np.ndarray) -> None:
    """0 <= H(C_{t+1}|C_t) <= log2(K)."""
    K = 8
    h = sequence_analysis.conditional_entropy(synthetic_codes, K)

    assert h >= 0, f"Conditional entropy should be >= 0, got {h}"
    assert h <= np.log2(K) + 0.01, (
        f"Conditional entropy should be <= log2({K})={np.log2(K):.3f}, got {h}"
    )


# ---------------------------------------------------------------------------
# Test 11: Mutual information non-negative
# ---------------------------------------------------------------------------


def test_mutual_information_nonneg(synthetic_codes: np.ndarray) -> None:
    """MI >= 0."""
    K = 8
    mi = sequence_analysis.mutual_information_bigram(synthetic_codes, K)
    assert mi >= -1e-10, f"MI should be >= 0, got {mi}"


# ---------------------------------------------------------------------------
# Test 12: Bigram productivity range
# ---------------------------------------------------------------------------


def test_bigram_productivity_range(synthetic_codes: np.ndarray) -> None:
    """0 < ratio <= 1.0."""
    K = 8
    result = compositionality.bigram_productivity(synthetic_codes, K)

    assert result["productivity_ratio"] > 0, (
        f"Productivity should be > 0, got {result['productivity_ratio']}"
    )
    assert result["productivity_ratio"] <= 1.0, (
        f"Productivity should be <= 1.0, got {result['productivity_ratio']}"
    )
    assert result["observed"] <= result["possible"]


# ---------------------------------------------------------------------------
# Test 13: Concept scan shape
# ---------------------------------------------------------------------------


def test_concept_scan_shape(
    small_transformer: SpectrogramTransformer,
    small_vqvae: HiddenStateVQVAE,
) -> None:
    """(K, n_freq) matrix."""
    n_freq = small_transformer.config.n_freq
    K = small_vqvae.config.codebook_size
    x = torch.randn(1, 8, n_freq)

    scan = concept_manipulation.concept_scan(
        small_transformer, small_vqvae, x,
        scan_position=4, source_layer=1,
    )

    assert scan.shape == (K, n_freq), f"Expected ({K}, {n_freq}), got {scan.shape}"


# ---------------------------------------------------------------------------
# Test 14: Chi-squared valid p-value
# ---------------------------------------------------------------------------


def test_chi_squared_valid_pvalue() -> None:
    """0 <= p <= 1; same distribution -> p > 0.05."""
    rng = np.random.RandomState(42)
    K = 8

    # Two similar distributions
    freq_a = rng.dirichlet(np.ones(K) * 10)
    freq_b = freq_a.copy()  # Same distribution

    result = context_analysis.chi_squared_test(freq_a, freq_b)

    assert 0 <= result["p_value"] <= 1, (
        f"p-value should be in [0, 1], got {result['p_value']}"
    )
    # Same distribution should not be significantly different
    assert result["p_value"] > 0.05, (
        f"Same distribution should give p > 0.05, got {result['p_value']}"
    )
