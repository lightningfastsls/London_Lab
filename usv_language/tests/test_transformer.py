"""Tests for positional encoding and causal transformer."""

from __future__ import annotations

import torch

from usv_language.src.model.positional import SinusoidalPositionalEncoding
from usv_language.src.model.transformer import CausalTransformerBlock, CausalTransformer


class TestSinusoidalPositionalEncoding:
    def test_output_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=256, dropout=0.0)
        x = torch.zeros(2, 100, 64)
        out = pe(x)
        assert out.shape == (2, 100, 64)

    def test_adds_positional_signal(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=256, dropout=0.0)
        x = torch.zeros(1, 50, 64)
        out = pe(x)
        # Should not be all zeros after adding positional encoding
        assert out.abs().sum() > 0

    def test_deterministic(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=256, dropout=0.0)
        x = torch.ones(1, 30, 64)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)


class TestCausalTransformerBlock:
    def test_output_shape(self):
        block = CausalTransformerBlock(d_model=64, n_heads=4, d_ff=256)
        x = torch.randn(2, 32, 64)
        mask = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)
        out = block(x, mask)
        assert out.shape == (2, 32, 64)


class TestCausalTransformer:
    def test_output_shape(self):
        transformer = CausalTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128)
        x = torch.randn(2, 32, 64)
        out = transformer(x)
        assert out.shape == (2, 32, 64)

    def test_causal_masking(self):
        """Output at position t should not depend on inputs at position t+1."""
        transformer = CausalTransformer(d_model=32, n_heads=2, n_layers=1, d_ff=64)
        transformer.eval()

        x = torch.randn(1, 10, 32)
        out_full = transformer(x)

        # Modify position 9, check that positions 0-8 are unchanged
        x_modified = x.clone()
        x_modified[0, 9] = torch.randn(32)
        out_modified = transformer(x_modified)

        # Positions 0-8 should be identical (causal: can't see position 9)
        assert torch.allclose(out_full[0, :9], out_modified[0, :9], atol=1e-5)

    def test_gradient_flow(self):
        transformer = CausalTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = transformer(x)
        out.sum().backward()
        assert x.grad is not None
