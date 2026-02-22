"""Tests for Hidden State VQ-VAE (Phase 8.3).

Covers: config validation, forward pass, gradient flow, EMA updates,
dead code reset, k-means init, overfit, utilization, checkpointing,
and HiddenStateDataset windowing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from usv_language.models.vqvae import (
    HiddenStateVQVAE,
    VQVAEConfig,
    VectorQuantizerV2,
)


class TestVQVAEConfig:
    """Test 1: Config validation (valid defaults + invalid values)."""

    def test_valid_defaults(self):
        config = VQVAEConfig()
        assert config.d_model == 512
        assert config.codebook_size == 64
        assert config.codebook_dim == 64
        assert config.commitment_weight == 0.25
        assert config.ema_decay == 0.99
        assert config.use_conv_encoder is True
        assert config.conv_kernel_size == 5

    def test_invalid_d_model(self):
        with pytest.raises(ValueError, match="d_model must be positive"):
            VQVAEConfig(d_model=0)

    def test_invalid_codebook_size(self):
        with pytest.raises(ValueError, match="codebook_size must be positive"):
            VQVAEConfig(codebook_size=-1)

    def test_invalid_ema_decay_low(self):
        with pytest.raises(ValueError, match="ema_decay must be in"):
            VQVAEConfig(ema_decay=0.0)

    def test_invalid_ema_decay_high(self):
        with pytest.raises(ValueError, match="ema_decay must be in"):
            VQVAEConfig(ema_decay=1.0)

    def test_invalid_even_kernel(self):
        with pytest.raises(ValueError, match="conv_kernel_size must be odd"):
            VQVAEConfig(conv_kernel_size=4)


class TestForwardPass:
    """Test 2: Forward pass output shapes."""

    def test_conv_encoder_shapes(self):
        config = VQVAEConfig(d_model=128, codebook_size=32, codebook_dim=32)
        model = HiddenStateVQVAE(config)
        x = torch.randn(2, 16, 128)
        out = model(x)

        assert out["recon"].shape == (2, 16, 128)
        assert out["indices"].shape == (2, 16)
        assert out["recon_loss"].ndim == 0
        assert out["commit_loss"].ndim == 0
        assert out["total_loss"].ndim == 0
        assert out["perplexity"].ndim == 0
        assert out["codebook_usage"].ndim == 0

    def test_linear_encoder_shapes(self):
        config = VQVAEConfig(
            d_model=128, codebook_size=32, codebook_dim=32,
            use_conv_encoder=False,
        )
        model = HiddenStateVQVAE(config)
        x = torch.randn(2, 16, 128)
        out = model(x)

        assert out["recon"].shape == (2, 16, 128)
        assert out["indices"].shape == (2, 16)


class TestCodebookIndices:
    """Test 3: Codebook indices valid range [0, K-1]."""

    def test_indices_valid_range(self):
        K = 32
        config = VQVAEConfig(d_model=64, codebook_size=K, codebook_dim=16)
        model = HiddenStateVQVAE(config)
        x = torch.randn(4, 8, 64)
        out = model(x)

        assert out["indices"].min() >= 0
        assert out["indices"].max() < K


class TestStraightThrough:
    """Test 4: Straight-through gradient flow to encoder."""

    def test_gradient_flows_to_input(self):
        config = VQVAEConfig(d_model=64, codebook_size=16, codebook_dim=16)
        model = HiddenStateVQVAE(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = model(x)
        out["total_loss"].backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Gradients should be non-trivial (not all zeros)
        assert x.grad.abs().sum() > 0

    def test_encoder_params_receive_gradients(self):
        config = VQVAEConfig(d_model=64, codebook_size=16, codebook_dim=16)
        model = HiddenStateVQVAE(config)
        x = torch.randn(2, 8, 64)
        out = model(x)
        out["total_loss"].backward()

        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for encoder.{name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for encoder.{name}"


class TestEMAUpdates:
    """Test 5: EMA updates modify codebook during training."""

    def test_ema_changes_codebook(self):
        config = VQVAEConfig(
            d_model=64, codebook_size=16, codebook_dim=16, ema_decay=0.5,
        )
        model = HiddenStateVQVAE(config)
        model.train()
        initial_weights = model.quantizer.embedding.weight.clone()

        x = torch.randn(4, 32, 64)
        _ = model(x)

        assert not torch.allclose(
            model.quantizer.embedding.weight, initial_weights,
        )

    def test_eval_no_ema(self):
        config = VQVAEConfig(d_model=64, codebook_size=16, codebook_dim=16)
        model = HiddenStateVQVAE(config)
        model.eval()
        initial_weights = model.quantizer.embedding.weight.clone()

        x = torch.randn(4, 32, 64)
        _ = model(x)

        assert torch.allclose(
            model.quantizer.embedding.weight, initial_weights,
        )


class TestDeadCodeReset:
    """Test 6: Dead code reset triggers below threshold."""

    def test_dead_codes_get_reset(self):
        K = 16
        vq = VectorQuantizerV2(
            codebook_size=K, codebook_dim=8, dead_code_threshold=0.5,
        )
        vq.train()

        # Create data that clusters tightly — most codes will be "dead"
        # Use a narrow distribution so only a few codes get assigned
        z = torch.randn(4, 32, 8) * 0.01  # very tight cluster
        z = torch.nn.functional.normalize(z, dim=-1)

        initial_weights = vq.embedding.weight.clone()

        # Run forward several times to trigger dead code resets
        for _ in range(5):
            _ = vq(z)

        # Some codes should have been reset (changed from initial)
        changed = not torch.allclose(
            vq.embedding.weight, initial_weights, atol=1e-3,
        )
        assert changed, "Expected dead codes to be reset"


class TestKMeansInit:
    """Test 7: K-means initialization sets codebook from data."""

    def test_kmeans_initializes_codebook(self):
        config = VQVAEConfig(d_model=64, codebook_size=16, codebook_dim=16)
        model = HiddenStateVQVAE(config)
        initial_weights = model.quantizer.embedding.weight.clone()

        # Generate some structured data
        data = torch.randn(100, 64)
        model.initialize_codebook_from_data(data)

        # Codebook should have changed
        assert not torch.allclose(
            model.quantizer.embedding.weight, initial_weights, atol=1e-3,
        )

        # Codebook should be L2-normalized
        norms = model.quantizer.embedding.weight.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestOverfit:
    """Test 8: Single-batch overfit (recon_loss < 0.01 after 200 steps)."""

    def test_single_batch_overfit(self):
        # Use codebook_dim = d_model so the only bottleneck is discretization
        # (no dimensionality reduction). With K=64 codes and only 32 frames,
        # the model has enough capacity to memorize.
        config = VQVAEConfig(
            d_model=64, codebook_size=64, codebook_dim=64,
            commitment_weight=0.1, use_conv_encoder=False,
        )
        model = HiddenStateVQVAE(config)
        model.train()

        # Fixed batch for overfitting — small and deterministic
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64)

        # Initialize codebook from data for faster convergence
        model.initialize_codebook_from_data(x.reshape(-1, 64))

        # Exclude codebook from optimizer (EMA-updated)
        params = [
            p for n, p in model.named_parameters()
            if "quantizer.embedding" not in n and p.requires_grad
        ]
        optimizer = torch.optim.Adam(params, lr=3e-3)

        for step in range(500):
            out = model(x)
            out["total_loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            out = model(x)

        assert out["recon_loss"].item() < 0.01, (
            f"Expected recon_loss < 0.01, got {out['recon_loss'].item():.4f}"
        )


class TestUtilization:
    """Test 9: Codebook utilization > 50% on synthetic data."""

    def test_utilization_above_50_percent(self):
        K = 32
        config = VQVAEConfig(
            d_model=64, codebook_size=K, codebook_dim=32,
        )
        model = HiddenStateVQVAE(config)
        model.train()

        # Diverse data: multiple distinct clusters
        x = torch.randn(8, 64, 64) * 2.0

        # Initialize codebook from data
        model.initialize_codebook_from_data(x.reshape(-1, 64))

        # Train briefly
        params = [
            p for n, p in model.named_parameters()
            if "quantizer.embedding" not in n and p.requires_grad
        ]
        optimizer = torch.optim.Adam(params, lr=1e-3)

        for _ in range(50):
            out = model(x)
            out["total_loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            out = model(x)

        usage = out["codebook_usage"].item()
        assert usage > 0.5, f"Expected utilization > 50%, got {usage:.1%}"


class TestCheckpointing:
    """Test 10: Checkpoint save/resume preserves state + EMA buffers."""

    def test_checkpoint_roundtrip(self):
        config = VQVAEConfig(d_model=64, codebook_size=16, codebook_dim=16)
        model = HiddenStateVQVAE(config)
        model.train()

        # Run a forward pass to update EMA state
        x = torch.randn(2, 8, 64)
        _ = model(x)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
            }
            torch.save(checkpoint, str(path))

            # Load into fresh model
            loaded = torch.load(str(path), map_location="cpu", weights_only=False)
            model2 = HiddenStateVQVAE(loaded["config"])
            model2.load_state_dict(loaded["model_state_dict"])

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                model.state_dict().items(), model2.state_dict().items(),
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"

            # Verify EMA buffers specifically
            assert torch.allclose(
                model.quantizer.ema_cluster_size,
                model2.quantizer.ema_cluster_size,
            )
            assert torch.allclose(
                model.quantizer.ema_embedding_sum,
                model2.quantizer.ema_embedding_sum,
            )

            # Verify same output on same input
            model.eval()
            model2.eval()
            with torch.no_grad():
                out1 = model(x)
                out2 = model2(x)
            assert torch.allclose(out1["recon"], out2["recon"], atol=1e-6)
            assert torch.equal(out1["indices"], out2["indices"])


class TestHiddenStateDataset:
    """Test 11: HiddenStateDataset windowing correctness."""

    def test_windowing(self):
        from usv_language.training.train_vqvae import HiddenStateDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test .npy file
            total_frames = 300
            d_model = 64
            data = np.random.randn(total_frames, d_model).astype(np.float32)
            path = Path(tmpdir) / "hidden_states.npy"
            np.save(str(path), data)

            ds = HiddenStateDataset(str(path), window_size=128, stride=64)

            try:
                # Check number of windows
                # stride-based: floor((300-128)/64) + 1 = 3
                expected_windows = (total_frames - 128) // 64 + 1
                assert len(ds) == expected_windows

                # Check shape of each item
                item = ds[0]
                assert item.shape == (128, d_model)

                # Check content matches source data
                assert np.allclose(item.numpy(), data[0:128])

                # Second window starts at stride offset
                item2 = ds[1]
                assert np.allclose(item2.numpy(), data[64:192])
            finally:
                # Release mmap file handle before temp dir cleanup (Windows)
                ds.close()

    def test_short_file(self):
        """File shorter than window_size should produce one padded window."""
        from usv_language.training.train_vqvae import HiddenStateDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            total_frames = 50
            d_model = 64
            data = np.random.randn(total_frames, d_model).astype(np.float32)
            path = Path(tmpdir) / "hidden_states.npy"
            np.save(str(path), data)

            ds = HiddenStateDataset(str(path), window_size=128, stride=64)

            try:
                assert len(ds) == 1
                item = ds[0]
                assert item.shape == (128, d_model)
                # First 50 frames should match, rest should be zero-padded
                assert np.allclose(item[:50].numpy(), data)
                assert np.allclose(item[50:].numpy(), 0.0)
            finally:
                # Release mmap file handle before temp dir cleanup (Windows)
                ds.close()


class TestConfigValidationExtended:
    """Test W6: commitment_weight and dead_code_threshold validation."""

    def test_invalid_commitment_weight(self):
        with pytest.raises(ValueError, match="commitment_weight must be >= 0"):
            VQVAEConfig(commitment_weight=-0.1)

    def test_invalid_dead_code_threshold(self):
        with pytest.raises(ValueError, match="dead_code_threshold must be >= 0"):
            VQVAEConfig(dead_code_threshold=-1.0)

    def test_zero_commitment_weight_allowed(self):
        config = VQVAEConfig(commitment_weight=0.0)
        assert config.commitment_weight == 0.0

    def test_zero_dead_code_threshold_allowed(self):
        config = VQVAEConfig(dead_code_threshold=0.0)
        assert config.dead_code_threshold == 0.0


class TestCountParameters:
    """Verify parameter count is in expected range (~820K)."""

    def test_param_count(self):
        config = VQVAEConfig()  # Default: d_model=512, codebook_dim=64
        model = HiddenStateVQVAE(config)
        n_params = model.count_parameters()
        # Expected ~820K params. Allow range 700K - 1M.
        assert 700_000 < n_params < 1_000_000, (
            f"Expected ~820K params, got {n_params:,}"
        )


class TestCompareLayers:
    """Test W4: score_layer and generate_report coverage."""

    def test_score_layer_ranking(self):
        """Better metrics should produce higher scores."""
        from usv_language.training.compare_layers import score_layer

        # Good metrics: high perplexity, high utilization, low recon_loss
        good = {"perplexity": 50.0, "codebook_usage": 0.9, "recon_loss": 0.01}
        # Bad metrics: low perplexity, low utilization, high recon_loss
        bad = {"perplexity": 5.0, "codebook_usage": 0.1, "recon_loss": 0.9}
        # Medium metrics
        medium = {"perplexity": 30.0, "codebook_usage": 0.5, "recon_loss": 0.3}

        s_good = score_layer(good, codebook_size=64)
        s_bad = score_layer(bad, codebook_size=64)
        s_medium = score_layer(medium, codebook_size=64)

        assert s_good > s_medium > s_bad, (
            f"Expected good ({s_good:.3f}) > medium ({s_medium:.3f}) > bad ({s_bad:.3f})"
        )

    def test_score_layer_range(self):
        """Score should be approximately in [0, 1]."""
        from usv_language.training.compare_layers import score_layer

        perfect = {"perplexity": 64.0, "codebook_usage": 1.0, "recon_loss": 0.0}
        worst = {"perplexity": 0.0, "codebook_usage": 0.0, "recon_loss": 1.0}

        assert 0.9 <= score_layer(perfect, codebook_size=64) <= 1.0
        assert 0.0 <= score_layer(worst, codebook_size=64) <= 0.1

    def test_generate_report_4_layers(self):
        """Report should contain table header, all 4 layers, and recommendation."""
        from usv_language.training.compare_layers import generate_report

        results = {
            2: {"recon_loss": 0.1, "commit_loss": 0.05, "perplexity": 40.0, "codebook_usage": 0.8},
            4: {"recon_loss": 0.05, "commit_loss": 0.03, "perplexity": 55.0, "codebook_usage": 0.9},
            6: {"recon_loss": 0.08, "commit_loss": 0.04, "perplexity": 45.0, "codebook_usage": 0.85},
            8: {"recon_loss": 0.15, "commit_loss": 0.06, "perplexity": 30.0, "codebook_usage": 0.7},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            generate_report(results, report_path, codebook_size=64)

            content = report_path.read_text(encoding="utf-8")

            # Check table header
            assert "| Layer | Recon Loss |" in content
            # Check all 4 layers present
            for layer in [2, 4, 6, 8]:
                assert f"| {layer:5d} " in content
            # Check recommendation section exists
            assert "## Recommendation" in content
            # Layer 4 should be recommended (best metrics)
            assert "Layer 4" in content
