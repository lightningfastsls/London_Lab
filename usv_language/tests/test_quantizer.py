"""Tests for VectorQuantizer."""

from __future__ import annotations

import torch

from usv_language.src.model.quantizer import VectorQuantizer


class TestVectorQuantizer:
    def test_output_shape(self):
        vq = VectorQuantizer(codebook_size=64, d_model=32)
        z = torch.randn(2, 16, 32)
        out = vq(z)
        assert out["z_q"].shape == (2, 16, 32)
        assert out["indices"].shape == (2, 16)
        assert out["vq_loss"].ndim == 0
        assert out["perplexity"].ndim == 0
        assert out["codebook_usage"].ndim == 0

    def test_straight_through_gradient(self):
        """Gradients should pass through quantization to encoder."""
        vq = VectorQuantizer(codebook_size=64, d_model=32)
        z = torch.randn(2, 16, 32, requires_grad=True)
        out = vq(z)
        out["z_q"].sum().backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape

    def test_indices_valid_range(self):
        vq = VectorQuantizer(codebook_size=64, d_model=32)
        z = torch.randn(4, 8, 32)
        out = vq(z)
        assert out["indices"].min() >= 0
        assert out["indices"].max() < 64

    def test_decode_indices(self):
        vq = VectorQuantizer(codebook_size=64, d_model=32)
        indices = torch.randint(0, 64, (2, 16))
        vectors = vq.decode_indices(indices)
        assert vectors.shape == (2, 16, 32)

    def test_ema_update_changes_codebook(self):
        vq = VectorQuantizer(codebook_size=16, d_model=8, ema_decay=0.5)
        vq.train()
        initial_weights = vq.embedding.weight.clone()

        z = torch.randn(4, 32, 8)
        _ = vq(z)

        # EMA should have updated the codebook
        assert not torch.allclose(vq.embedding.weight, initial_weights)

    def test_perplexity_range(self):
        """Perplexity should be between 1 and codebook_size."""
        K = 64
        vq = VectorQuantizer(codebook_size=K, d_model=32)
        z = torch.randn(4, 32, 32)
        out = vq(z)
        assert out["perplexity"].item() >= 1.0
        assert out["perplexity"].item() <= K

    def test_codebook_usage_range(self):
        vq = VectorQuantizer(codebook_size=64, d_model=32)
        z = torch.randn(4, 32, 32)
        out = vq(z)
        assert 0.0 <= out["codebook_usage"].item() <= 1.0

    def test_eval_mode_no_ema(self):
        """In eval mode, codebook should not change."""
        vq = VectorQuantizer(codebook_size=16, d_model=8)
        vq.eval()
        initial_weights = vq.embedding.weight.clone()

        z = torch.randn(4, 32, 8)
        _ = vq(z)

        assert torch.allclose(vq.embedding.weight, initial_weights)
