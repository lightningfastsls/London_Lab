"""Tests for the full VQVAE model."""

from __future__ import annotations

import torch
import pytest

from usv_language.src.model.vqvae import VQVAE, VQVAEConfig


@pytest.fixture
def model():
    config = VQVAEConfig(n_freq=170, d_model=32, n_heads=4, n_layers=2, d_ff=64, codebook_size=64)
    return VQVAE(config)


class TestVQVAEConfig:
    def test_defaults(self):
        cfg = VQVAEConfig()
        assert cfg.n_freq == 170
        assert cfg.d_model == 64

    def test_validation(self):
        with pytest.raises(ValueError, match="n_freq"):
            VQVAEConfig(n_freq=0)
        with pytest.raises(ValueError, match="d_model.*n_heads"):
            VQVAEConfig(d_model=65, n_heads=4)  # Not divisible


class TestVQVAEForward:
    def test_output_keys(self, model):
        x = torch.randn(2, 16, 170)
        out = model(x)
        assert "x_recon" in out
        assert "vq_loss" in out
        assert "indices" in out
        assert "next_code_logits" in out
        assert "perplexity" in out
        assert "codebook_usage" in out

    def test_output_shapes(self, model):
        x = torch.randn(2, 16, 170)
        out = model(x)
        assert out["x_recon"].shape == (2, 16, 170)
        assert out["indices"].shape == (2, 16)
        assert out["next_code_logits"].shape == (2, 16, 64)

    def test_recon_range(self, model):
        """Reconstruction should be in [0, 1] due to sigmoid decoder."""
        x = torch.randn(2, 16, 170)
        out = model(x)
        assert out["x_recon"].min() >= 0.0
        assert out["x_recon"].max() <= 1.0

    def test_backward_pass(self, model):
        x = torch.randn(2, 16, 170)
        out = model(x)
        loss = out["x_recon"].mean() + out["vq_loss"]
        loss.backward()
        # Encoder should have gradients
        for p in model.encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestVQVAEEncode:
    def test_encode_to_codes(self, model):
        x = torch.randn(2, 16, 170)
        codes = model.encode_to_codes(x)
        assert codes.shape == (2, 16)
        assert codes.dtype == torch.long
        assert codes.min() >= 0
        assert codes.max() < model.config.codebook_size


class TestVQVAEDecode:
    def test_decode_codes(self, model):
        codes = torch.randint(0, 64, (2, 16))
        recon = model.decode_codes_to_spectrogram(codes)
        assert recon.shape == (2, 16, 170)


class TestVQVAEGenerate:
    def test_generate(self, model):
        prompt = torch.randint(0, 64, (1, 5))
        generated = model.generate(prompt, max_new=10, temperature=1.0)
        assert generated.shape == (1, 15)  # 5 prompt + 10 new
        assert generated[:, :5].equal(prompt)
