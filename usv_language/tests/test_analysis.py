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

import json
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
from usv_language.analysis import information_theory
from usv_language.analysis.run_analysis import (
    build_analysis_sample_spectrogram,
    coerce_vqvae_config,
    load_analysis_metadata,
    resolve_device,
    run_analysis,
    validate_analysis_inputs,
    validate_vqvae_provenance,
)
from usv_language.analysis.information_theory import (
    BurstinessResult,
    EntropyRateResult,
    ProductivityResult,
    ZipfEntropyResult,
    ZipfResult,
)
from usv_language.models.transformer import SpectrogramTransformer, TransformerConfig
from usv_language.models.vqvae import HiddenStateVQVAE, VQVAEConfig
from usv_language.training.compare_layers import compare_layers, parse_args as parse_compare_layers_args
from usv_language.training.extract_hidden_states import (
    extract_hidden_states,
    parse_args as parse_extract_hidden_states_args,
)


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


def test_group_codes_by_legacy_frame_metadata() -> None:
    codes = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    metadata = {
        "frames": [
            {"frame_index": 0, "recording_id": "rec_a"},
            {"frame_index": 1, "recording_id": "rec_a"},
            {"frame_index": 2, "recording_id": "rec_a"},
            {"frame_index": 3, "recording_id": "rec_b"},
            {"frame_index": 4, "recording_id": "rec_b"},
            {"frame_index": 5, "recording_id": "rec_b"},
        ]
    }

    grouped = context_analysis.group_codes_by_metadata(codes, metadata)

    assert set(grouped.keys()) == {"rec_a", "rec_b"}
    np.testing.assert_array_equal(grouped["rec_a"], np.array([0, 1, 2]))
    np.testing.assert_array_equal(grouped["rec_b"], np.array([3, 4, 5]))


def test_extract_bout_code_sequences_from_legacy_metadata(
    synthetic_hidden_states: np.ndarray,
    small_vqvae: HiddenStateVQVAE,
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({
            "frames": [
                {"frame_index": i, "recording_id": "rec_a" if i < 100 else "rec_b"}
                for i in range(len(synthetic_hidden_states))
            ]
        }),
        encoding="utf-8",
    )

    sequences = sequence_analysis.extract_bout_code_sequences(
        synthetic_hidden_states, metadata_path, small_vqvae, batch_size=64, device="cpu",
    )

    assert len(sequences) == 2
    assert sum(len(seq) for seq in sequences) == len(synthetic_hidden_states)


def test_build_analysis_sample_spectrogram_shape(
    small_transformer: SpectrogramTransformer,
) -> None:
    hidden_states = np.random.randn(20, small_transformer.config.d_model).astype(np.float32)

    sample = build_analysis_sample_spectrogram(
        hidden_states, small_transformer, source_layer=1, device=torch.device("cpu"), sample_len=8,
    )

    assert sample.shape == (1, 8, small_transformer.config.n_freq)


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = resolve_device("cuda")

    assert device.type == "cpu"


def test_coerce_vqvae_config_accepts_dict() -> None:
    config = coerce_vqvae_config({
        "d_model": 64,
        "codebook_size": 8,
        "codebook_dim": 16,
        "use_conv_encoder": False,
    })

    assert isinstance(config, VQVAEConfig)
    assert config.d_model == 64


def test_build_analysis_sample_spectrogram_rejects_empty_hidden_states(
    small_transformer: SpectrogramTransformer,
) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        build_analysis_sample_spectrogram(
            np.empty((0, small_transformer.config.d_model), dtype=np.float32),
            small_transformer,
            source_layer=1,
            device=torch.device("cpu"),
        )


def test_load_analysis_metadata_auto_detects_adjacent_metadata(
    tmp_path: Path,
) -> None:
    hidden_states_path = tmp_path / "hidden_states_layer1.npy"
    hidden_states_path.write_bytes(b"")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"total_frames": 8, "d_model": 64}), encoding="utf-8")

    metadata, resolved_path, source = load_analysis_metadata(str(hidden_states_path))

    assert metadata == {"total_frames": 8, "d_model": 64}
    assert resolved_path == str(metadata_path.resolve())
    assert source == "adjacent"


def test_load_analysis_metadata_rejects_missing_explicit_path(
    tmp_path: Path,
) -> None:
    hidden_states_path = tmp_path / "hidden_states_layer1.npy"
    hidden_states_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        load_analysis_metadata(
            str(hidden_states_path),
            metadata_path=str(tmp_path / "missing_metadata.json"),
        )


def test_run_analysis_records_partial_failures_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = SpectrogramTransformer(
        TransformerConfig(
            n_freq=16,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ffn=128,
            max_seq_len=64,
            dropout=0.0,
        )
    )
    vqvae = HiddenStateVQVAE(
        VQVAEConfig(
            d_model=64,
            codebook_size=8,
            codebook_dim=32,
            use_conv_encoder=False,
        )
    )

    transformer_path = tmp_path / "transformer.pt"
    vqvae_path = tmp_path / "vqvae.pt"
    hidden_states_path = tmp_path / "hidden_states.npy"
    output_dir = tmp_path / "analysis_output"

    torch.save(
        {"model_state_dict": transformer.state_dict(), "config": transformer.config},
        transformer_path,
    )
    torch.save(
        {
            "model_state_dict": vqvae.state_dict(),
            "config": vqvae.config,
            "provenance": {
                "hidden_states_path": str(hidden_states_path.resolve()),
                "hidden_states_filename": hidden_states_path.name,
                "source_layer": 1,
            },
        },
        vqvae_path,
    )
    np.save(hidden_states_path, np.random.randn(32, 64).astype(np.float32))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"total_frames": 32, "d_model": 64, "layers_extracted": [1]}),
        encoding="utf-8",
    )

    def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(codebook_viz, "decode_all_entries", lambda *args, **kwargs: np.zeros((8, 16)))
    monkeypatch.setattr(codebook_viz, "plot_decoded_profiles", _noop)
    monkeypatch.setattr(codebook_viz, "plot_codebook_usage", _noop)
    monkeypatch.setattr(codebook_viz, "plot_codebook_projection", _noop)
    monkeypatch.setattr(codebook_viz, "find_exemplars", lambda *args, **kwargs: {})
    monkeypatch.setattr(sequence_analysis, "extract_code_sequences", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sequence boom")))
    monkeypatch.setattr(concept_manipulation, "concept_scan", lambda *args, **kwargs: np.zeros((8, 16)))
    monkeypatch.setattr(concept_manipulation, "plot_concept_scan", _noop)

    run_analysis(
        transformer_path=str(transformer_path),
        vqvae_path=str(vqvae_path),
        hidden_states_path=str(hidden_states_path),
        output_dir=str(output_dir),
        source_layer=1,
        device_name="cpu",
    )

    summary = json.loads((output_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["artifacts"]["metadata"] == str(metadata_path.resolve())
    assert summary["artifacts"]["metadata_source"] == "adjacent"
    assert summary["artifacts"]["vqvae_provenance"]["source_layer"] == 1
    assert summary["section_status"]["codebook_visualization"]["status"] == "completed"
    assert summary["section_status"]["sequence_analysis"]["status"] == "failed"
    assert "sequence boom" in summary["section_status"]["sequence_analysis"]["reason"]
    assert summary["section_status"]["concept_manipulation"]["status"] == "completed"
    assert summary["section_status"]["context_analysis"]["status"] == "skipped"
    assert summary["section_status"]["compositionality"]["status"] == "skipped"
    assert summary["section_status"]["information_theory"]["status"] == "skipped"


def test_run_analysis_rejects_missing_explicit_metadata_path(
    tmp_path: Path,
) -> None:
    transformer = SpectrogramTransformer(
        TransformerConfig(
            n_freq=16,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ffn=128,
            max_seq_len=64,
            dropout=0.0,
        )
    )
    vqvae = HiddenStateVQVAE(
        VQVAEConfig(
            d_model=64,
            codebook_size=8,
            codebook_dim=32,
            use_conv_encoder=False,
        )
    )

    transformer_path = tmp_path / "transformer.pt"
    vqvae_path = tmp_path / "vqvae.pt"
    hidden_states_path = tmp_path / "hidden_states_layer1.npy"

    torch.save(
        {"model_state_dict": transformer.state_dict(), "config": transformer.config},
        transformer_path,
    )
    torch.save(
        {"model_state_dict": vqvae.state_dict(), "config": vqvae.config},
        vqvae_path,
    )
    np.save(hidden_states_path, np.random.randn(16, 64).astype(np.float32))

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        run_analysis(
            transformer_path=str(transformer_path),
            vqvae_path=str(vqvae_path),
            hidden_states_path=str(hidden_states_path),
            output_dir=str(tmp_path / "analysis_output"),
            source_layer=1,
            metadata_path=str(tmp_path / "missing_metadata.json"),
            device_name="cpu",
        )


def test_tiny_artifact_chain_extract_compare_analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_freq = 8
    d_model = 16
    transformer = SpectrogramTransformer(
        TransformerConfig(
            n_freq=n_freq,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            d_ffn=32,
            max_seq_len=8,
            dropout=0.0,
        )
    )

    data_dir = tmp_path / "bout_data"
    hidden_states_dir = tmp_path / "hidden_states"
    compare_output_dir = tmp_path / "compare_output"
    analysis_output_dir = tmp_path / "analysis_output"
    transformer_path = tmp_path / "transformer.pt"

    data_dir.mkdir()
    for idx in range(3):
        spec = np.random.randn(n_freq, 20).astype(np.float32)
        np.save(data_dir / f"rec_{idx:02d}_bout0.npy", spec)

    torch.save(
        {"epoch": 0, "model_state_dict": transformer.state_dict(), "config": transformer.config},
        transformer_path,
    )

    extract_args = parse_extract_hidden_states_args([
        "--checkpoint", str(transformer_path),
        "--data-dir", str(data_dir),
        "--output-dir", str(hidden_states_dir),
        "--layers", "1",
        "--primary-layer", "1",
        "--batch-size", "2",
        "--num-workers", "0",
    ])
    extract_hidden_states(extract_args)

    hidden_states_path = hidden_states_dir / "hidden_states_layer1.npy"
    metadata_path = hidden_states_dir / "metadata.json"
    assert hidden_states_path.exists()
    assert metadata_path.exists()

    compare_args = parse_compare_layers_args([
        "--hidden-states-dir", str(hidden_states_dir),
        "--output-dir", str(compare_output_dir),
        "--layers", "1",
        "--epochs", "1",
        "--batch-size", "4",
        "--patience", "0",
        "--d-model", str(d_model),
        "--codebook-size", "4",
        "--codebook-dim", "8",
        "--window-size", "8",
        "--stride", "4",
        "--val-fraction", "0.25",
        "--num-workers", "0",
        "--seed", "123",
    ])
    compare_layers(compare_args)

    layer_dir = compare_output_dir / "layer_1"
    vqvae_checkpoint = layer_dir / "best.pt"
    comparison_report = compare_output_dir / "comparison_report.md"
    assert vqvae_checkpoint.exists()
    assert comparison_report.exists()

    def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(codebook_viz, "plot_decoded_profiles", _noop)
    monkeypatch.setattr(codebook_viz, "plot_codebook_usage", _noop)
    monkeypatch.setattr(codebook_viz, "plot_codebook_projection", _noop)
    monkeypatch.setattr(codebook_viz, "plot_exemplar_gallery", _noop)
    monkeypatch.setattr(sequence_analysis, "plot_zipf", _noop)
    monkeypatch.setattr(sequence_analysis, "plot_entropy_rate", _noop)
    monkeypatch.setattr(sequence_analysis, "plot_mi_decay", _noop)
    monkeypatch.setattr(sequence_analysis, "plot_transition_matrix", _noop)
    monkeypatch.setattr(concept_manipulation, "plot_concept_scan", _noop)
    monkeypatch.setattr(compositionality, "plot_bigram_productivity", _noop)
    monkeypatch.setattr(compositionality, "plot_positional_independence", _noop)
    monkeypatch.setattr(
        information_theory,
        "zipf_exponent_mle",
        lambda codes: ZipfResult(alpha=2.0, xmin=1.0, p_value=1.0, n_tail=len(codes), log_likelihood_ratio=0.0),
    )
    monkeypatch.setattr(
        information_theory,
        "zipf_via_shannon_entropy",
        lambda codes, K: ZipfEntropyResult(
            alpha_estimate=1.0,
            entropy_observed=1.0,
            entropy_ci=(0.9, 1.1),
            method="stub",
        ),
    )
    monkeypatch.setattr(
        information_theory,
        "entropy_rate",
        lambda codes, max_order=8: EntropyRateResult(
            orders=[1, 2],
            rates_plugin=[1.0, 0.5],
            rates_corrected=[1.0, 0.5],
            convergence_order=2,
        ),
    )
    monkeypatch.setattr(information_theory, "ngram_idioms", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        information_theory,
        "ngram_productivity",
        lambda *args, **kwargs: ProductivityResult(
            observed=1,
            possible=16,
            productivity_ratio=1 / 16,
            null_ci_lower=0.0,
            null_ci_upper=0.1,
            n=2,
        ),
    )
    monkeypatch.setattr(
        information_theory,
        "conditional_entropy_by_lag",
        lambda *args, **kwargs: [0.1] * 10,
    )
    monkeypatch.setattr(
        information_theory,
        "mutual_information_rate",
        lambda *args, **kwargs: [0.2] * 20,
    )
    monkeypatch.setattr(
        information_theory,
        "burstiness_coefficient",
        lambda *args, **kwargs: BurstinessResult(
            cv=1.0,
            mean_iei=0.1,
            std_iei=0.1,
            n_bursts=1,
            mean_burst_duration=0.1,
            mean_inter_burst_interval=0.0,
            interpretation="poisson",
        ),
    )

    run_analysis(
        transformer_path=str(transformer_path),
        vqvae_path=str(vqvae_checkpoint),
        hidden_states_path=str(hidden_states_path),
        output_dir=str(analysis_output_dir),
        source_layer=1,
        device_name="cpu",
    )

    summary = json.loads((analysis_output_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    report_text = comparison_report.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert summary["artifacts"]["hidden_states"] == str(hidden_states_path.resolve())
    assert summary["artifacts"]["metadata"] == str(metadata_path.resolve())
    assert summary["artifacts"]["metadata_source"] == "adjacent"
    assert summary["artifacts"]["vqvae_provenance"]["source_layer"] == 1
    assert summary["section_status"]["codebook_visualization"]["status"] == "completed"
    assert summary["section_status"]["sequence_analysis"]["status"] == "completed"
    assert "## Extraction Provenance" in report_text
    assert str(hidden_states_path.resolve()) in report_text
    assert str(metadata_path.resolve()) in report_text
    assert metadata["layers_extracted"] == [1]


def test_validate_vqvae_provenance_rejects_mismatched_hidden_states_path(
    tmp_path: Path,
) -> None:
    expected_path = tmp_path / "hidden_states_layer4.npy"
    expected_path.write_bytes(b"")
    provided_path = tmp_path / "other_hidden_states_layer4.npy"
    provided_path.write_bytes(b"")

    with pytest.raises(ValueError, match="different hidden-state artifact"):
        validate_vqvae_provenance(
            {"provenance": {"hidden_states_path": str(expected_path.resolve()), "source_layer": 4}},
            str(provided_path),
            source_layer=4,
        )


def test_validate_vqvae_provenance_rejects_mismatched_source_layer(
    tmp_path: Path,
) -> None:
    hidden_states_path = tmp_path / "hidden_states_layer4.npy"
    hidden_states_path.write_bytes(b"")

    with pytest.raises(ValueError, match="source_layer does not match"):
        validate_vqvae_provenance(
            {"provenance": {"hidden_states_path": str(hidden_states_path.resolve()), "source_layer": 4}},
            str(hidden_states_path),
            source_layer=2,
        )


def test_validate_analysis_inputs_rejects_filename_layer_mismatch(
    small_transformer: SpectrogramTransformer,
    tmp_path: Path,
) -> None:
    hidden_states_path = tmp_path / "hidden_states_layer4.npy"
    hidden_states_path.write_bytes(b"")

    with pytest.raises(ValueError, match="filename layer does not match"):
        validate_analysis_inputs(
            str(hidden_states_path),
            np.zeros((8, small_transformer.config.d_model), dtype=np.float32),
            source_layer=2,
            transformer=small_transformer,
        )


def test_validate_analysis_inputs_rejects_metadata_total_frames_mismatch(
    small_transformer: SpectrogramTransformer,
    tmp_path: Path,
) -> None:
    hidden_states_path = tmp_path / "hidden_states_layer1.npy"
    hidden_states_path.write_bytes(b"")

    with pytest.raises(ValueError, match="Metadata total_frames does not match"):
        validate_analysis_inputs(
            str(hidden_states_path),
            np.zeros((8, small_transformer.config.d_model), dtype=np.float32),
            source_layer=1,
            transformer=small_transformer,
            metadata={"total_frames": 9, "d_model": small_transformer.config.d_model},
        )


def test_validate_analysis_inputs_rejects_metadata_d_model_mismatch(
    small_transformer: SpectrogramTransformer,
    tmp_path: Path,
) -> None:
    hidden_states_path = tmp_path / "hidden_states_layer1.npy"
    hidden_states_path.write_bytes(b"")

    with pytest.raises(ValueError, match="Metadata d_model does not match"):
        validate_analysis_inputs(
            str(hidden_states_path),
            np.zeros((8, small_transformer.config.d_model), dtype=np.float32),
            source_layer=1,
            transformer=small_transformer,
            metadata={"total_frames": 8, "d_model": small_transformer.config.d_model + 1},
        )


def test_validate_analysis_inputs_rejects_out_of_range_source_layer(
    small_transformer: SpectrogramTransformer,
    tmp_path: Path,
) -> None:
    hidden_states_path = tmp_path / "hidden_states_layer3.npy"
    hidden_states_path.write_bytes(b"")

    with pytest.raises(ValueError, match="source_layer must be in"):
        validate_analysis_inputs(
            str(hidden_states_path),
            np.zeros((8, small_transformer.config.d_model), dtype=np.float32),
            source_layer=3,
            transformer=small_transformer,
        )
