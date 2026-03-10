"""Tests for Hidden State VQ-VAE (Phase 8.3).

Covers: config validation, forward pass, gradient flow, EMA updates,
dead code reset, k-means init, overfit, utilization, checkpointing,
and HiddenStateDataset windowing.
"""

from __future__ import annotations

import json
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
from usv_language.training.train_vqvae import (
    HiddenStateDataset,
    build_hidden_state_provenance,
    build_dataloaders,
    coerce_vqvae_config,
    load_checkpoint,
    save_checkpoint,
)
from usv_language.training.compare_layers import validate_hidden_state_artifact


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
        torch.manual_seed(42)
        config = VQVAEConfig(
            d_model=64, codebook_size=64, codebook_dim=64,
            commitment_weight=0.1, use_conv_encoder=False,
        )
        model = HiddenStateVQVAE(config)
        model.train()

        # Fixed batch for overfitting — small and deterministic
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
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test .npy file
            total_frames = 300
            d_model = 64
            data = np.random.randn(total_frames, d_model).astype(np.float32)
            path = Path(tmpdir) / "hidden_states.npy"
            np.save(str(path), data)

            ds = HiddenStateDataset(str(path), window_size=128, stride=64)

            try:
                # Tail coverage should add a final window anchored to the end.
                assert ds.starts == [0, 64, 128, 172]
                assert len(ds) == 4

                # Check shape of each item
                item = ds[0]
                assert item.shape == (128, d_model)

                # Check content matches source data
                assert np.allclose(item.numpy(), data[0:128])

                # Second window starts at stride offset
                item2 = ds[1]
                assert np.allclose(item2.numpy(), data[64:192])

                # Final window is anchored to the end instead of dropping tail frames
                tail_item = ds[-1]
                assert np.allclose(tail_item.numpy(), data[172:300])
            finally:
                # Release mmap file handle before temp dir cleanup (Windows)
                ds.close()

    def test_short_file(self):
        """File shorter than window_size should produce one padded window."""
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

    def test_build_dataloaders_drops_overlap_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            total_frames = 300
            d_model = 64
            data = np.random.randn(total_frames, d_model).astype(np.float32)
            path = Path(tmpdir) / "hidden_states.npy"
            np.save(str(path), data)

            ds = HiddenStateDataset(str(path), window_size=128, stride=64)

            try:
                train_loader, val_loader = build_dataloaders(
                    ds,
                    batch_size=2,
                    val_fraction=0.25,
                    num_workers=0,
                )

                train_indices = train_loader.dataset.indices
                val_indices = val_loader.dataset.indices

                assert train_indices == [0]
                assert val_indices == [3]

                train_last_end = ds.starts[train_indices[-1]] + ds.window_size
                val_first_start = ds.starts[val_indices[0]]
                assert train_last_end <= val_first_start
            finally:
                ds.close()

    def test_empty_hidden_states_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.npy"
            np.save(str(path), np.empty((0, 64), dtype=np.float32))

            with pytest.raises(ValueError, match="empty"):
                HiddenStateDataset(str(path), window_size=128, stride=64)

    def test_non_2d_hidden_states_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.npy"
            np.save(str(path), np.zeros((10, 4, 3), dtype=np.float32))

            with pytest.raises(ValueError, match="shape"):
                HiddenStateDataset(str(path), window_size=128, stride=64)

    def test_invalid_window_or_stride_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hidden_states.npy"
            np.save(str(path), np.zeros((10, 64), dtype=np.float32))

            with pytest.raises(ValueError, match="window_size"):
                HiddenStateDataset(str(path), window_size=0, stride=64)
            with pytest.raises(ValueError, match="stride"):
                HiddenStateDataset(str(path), window_size=128, stride=0)


class TestTrainVQVAEHelpers:
    def test_build_dataloaders_rejects_invalid_val_fraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hidden_states.npy"
            np.save(str(path), np.random.randn(300, 64).astype(np.float32))
            ds = HiddenStateDataset(str(path), window_size=128, stride=64)

            try:
                with pytest.raises(ValueError, match="val_fraction"):
                    build_dataloaders(ds, batch_size=2, val_fraction=0.0, num_workers=0)
            finally:
                ds.close()

    def test_coerce_vqvae_config_accepts_dict(self):
        config = coerce_vqvae_config({
            "d_model": 64,
            "codebook_size": 16,
            "codebook_dim": 16,
            "use_conv_encoder": False,
        })

        assert isinstance(config, VQVAEConfig)
        assert config.d_model == 64

    def test_load_checkpoint_coerces_dict_config(self):
        config_dict = {
            "d_model": 64,
            "codebook_size": 16,
            "codebook_dim": 16,
            "use_conv_encoder": False,
        }
        model = HiddenStateVQVAE(VQVAEConfig(**config_dict))
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3,
        )
        from usv_language.training.train_transformer import CosineWarmupScheduler
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=2, max_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save({
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": config_dict,
            }, str(path))

            model2 = HiddenStateVQVAE(VQVAEConfig(**config_dict))
            optimizer2 = torch.optim.AdamW(
                [p for p in model2.parameters() if p.requires_grad], lr=1e-3,
            )
            scheduler2 = CosineWarmupScheduler(optimizer2, warmup_steps=2, max_steps=10)
            ckpt = load_checkpoint(path, model2, optimizer2, scheduler2)

            assert isinstance(ckpt["config"], VQVAEConfig)

    def test_build_hidden_state_provenance_reads_neighbor_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hs_path = Path(tmpdir) / "hidden_states_layer4.npy"
            np.save(str(hs_path), np.zeros((8, 64), dtype=np.float32))
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_path.write_text(
                json.dumps({"d_model": 64, "total_frames": 8, "layers_extracted": [2, 4]}),
                encoding="utf-8",
            )

            provenance = build_hidden_state_provenance(hs_path)

            assert provenance["hidden_states_filename"] == "hidden_states_layer4.npy"
            assert provenance["source_layer"] == 4
            assert provenance["metadata_d_model"] == 64
            assert provenance["metadata_total_frames"] == 8
            assert provenance["layers_extracted"] == [2, 4]

    def test_save_checkpoint_persists_provenance(self):
        config = VQVAEConfig(d_model=64, codebook_size=16, codebook_dim=16)
        model = HiddenStateVQVAE(config)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3,
        )
        from usv_language.training.train_transformer import CosineWarmupScheduler
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=2, max_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            provenance = {
                "hidden_states_path": str((Path(tmpdir) / "hidden_states_layer4.npy").resolve()),
                "hidden_states_filename": "hidden_states_layer4.npy",
                "source_layer": 4,
            }
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch=1,
                config=config,
                best_val_loss=0.1,
                history=[],
                provenance=provenance,
            )

            loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            assert loaded["provenance"] == provenance


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

    def test_generate_report_includes_failures(self):
        from usv_language.training.compare_layers import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            generate_report(
                {},
                report_path,
                codebook_size=64,
                failures={2: "missing hidden state file", 4: "shape mismatch"},
            )

            content = report_path.read_text(encoding="utf-8")
            assert "No layer completed successfully" in content
            assert "Layer 2: missing hidden state file" in content
            assert "Layer 4: shape mismatch" in content

    def test_generate_report_includes_provenance_sections(self):
        from usv_language.training.compare_layers import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            generate_report(
                {
                    4: {
                        "recon_loss": 0.05,
                        "commit_loss": 0.03,
                        "perplexity": 55.0,
                        "codebook_usage": 0.9,
                    },
                },
                report_path,
                codebook_size=64,
                extraction_metadata={
                    "checkpoint": "transformer_best.pt",
                    "checkpoint_epoch": 12,
                    "layers_extracted": [2, 4, 6],
                    "primary_layer": 4,
                    "total_frames": 512,
                    "d_model": 64,
                },
                layer_run_configs={
                    4: {
                        "dataset": {
                            "hidden_states": "hidden_states_layer4.npy",
                            "window_size": 128,
                            "stride": 64,
                        },
                        "training": {"epochs": 20, "patience": 5},
                        "provenance": {
                            "hidden_states_path": "C:/tmp/hidden_states_layer4.npy",
                            "metadata_path": "C:/tmp/metadata.json",
                            "source_layer": 4,
                        },
                    },
                },
            )

            content = report_path.read_text(encoding="utf-8")
            assert "## Extraction Provenance" in content
            assert "transformer_best.pt" in content
            assert "## Layer Artifact Provenance" in content
            assert "C:/tmp/hidden_states_layer4.npy" in content
            assert "C:/tmp/metadata.json" in content

    def test_validate_hidden_state_artifact_rejects_layer_missing_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hidden_states_layer4.npy"
            np.save(str(path), np.zeros((8, 64), dtype=np.float32))

            with pytest.raises(ValueError, match="layers_extracted"):
                validate_hidden_state_artifact(
                    path,
                    layer=4,
                    metadata={"layers_extracted": [2], "total_frames": 8, "d_model": 64},
                )

    def test_validate_hidden_state_artifact_rejects_metadata_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hidden_states_layer4.npy"
            np.save(str(path), np.zeros((8, 64), dtype=np.float32))

            with pytest.raises(ValueError, match="does not match metadata d_model"):
                validate_hidden_state_artifact(
                    path,
                    layer=4,
                    metadata={"layers_extracted": [4], "total_frames": 8, "d_model": 32},
                )


def test_compare_layers_report_surfaces_layer_run_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usv_language.training import compare_layers as compare_layers_module

    hidden_states_dir = tmp_path / "hidden_states"
    hidden_states_dir.mkdir()
    output_dir = tmp_path / "comparison_output"

    hidden_states_path = hidden_states_dir / "hidden_states_layer4.npy"
    np.save(str(hidden_states_path), np.random.randn(32, 64).astype(np.float32))
    (hidden_states_dir / "metadata.json").write_text(
        json.dumps({
            "checkpoint": "transformer_best.pt",
            "checkpoint_epoch": 7,
            "layers_extracted": [4],
            "primary_layer": 4,
            "total_frames": 32,
            "d_model": 64,
        }),
        encoding="utf-8",
    )

    def _fake_train(args):
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "config.json").write_text(
            json.dumps({
                "dataset": {
                    "hidden_states": args.hidden_states,
                    "window_size": 128,
                    "stride": 64,
                },
                "training": {"epochs": args.epochs, "patience": args.patience},
                "provenance": {
                    "hidden_states_path": str(Path(args.hidden_states).resolve()),
                    "metadata_path": str((Path(args.hidden_states).parent / "metadata.json").resolve()),
                    "source_layer": 4,
                },
            }),
            encoding="utf-8",
        )
        return {
            "recon_loss": 0.05,
            "commit_loss": 0.02,
            "perplexity": 12.0,
            "codebook_usage": 0.5,
        }

    monkeypatch.setattr(compare_layers_module, "train_vqvae", _fake_train)

    args = compare_layers_module.parse_args([
        "--hidden-states-dir", str(hidden_states_dir),
        "--output-dir", str(output_dir),
        "--layers", "4",
        "--d-model", "64",
        "--epochs", "3",
        "--patience", "2",
    ])
    compare_layers_module.compare_layers(args)

    report = (output_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "## Extraction Provenance" in report
    assert "transformer_best.pt" in report
    assert "## Layer Artifact Provenance" in report
    assert str(hidden_states_path.resolve()) in report

