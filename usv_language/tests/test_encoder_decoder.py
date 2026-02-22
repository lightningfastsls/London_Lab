"""Tests for ColumnEncoder and ColumnDecoder."""

from __future__ import annotations

import torch

from usv_language.src.model.encoder import ColumnEncoder
from usv_language.src.model.decoder import ColumnDecoder


class TestColumnEncoder:
    def test_output_shape(self):
        enc = ColumnEncoder(n_freq=170, d_model=64)
        x = torch.randn(2, 32, 170)
        out = enc(x)
        assert out.shape == (2, 32, 64)

    def test_custom_hidden_dims(self):
        enc = ColumnEncoder(n_freq=170, d_model=32, hidden_dims=[128, 64])
        x = torch.randn(1, 16, 170)
        out = enc(x)
        assert out.shape == (1, 16, 32)

    def test_gradient_flow(self):
        enc = ColumnEncoder(n_freq=170, d_model=64)
        x = torch.randn(2, 16, 170, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestColumnDecoder:
    def test_output_shape(self):
        dec = ColumnDecoder(d_model=64, n_freq=170)
        x = torch.randn(2, 32, 64)
        out = dec(x)
        assert out.shape == (2, 32, 170)

    def test_output_range(self):
        """Decoder uses sigmoid, output should be in [0, 1]."""
        dec = ColumnDecoder(d_model=64, n_freq=170)
        x = torch.randn(2, 16, 64)
        out = dec(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_gradient_flow(self):
        dec = ColumnDecoder(d_model=64, n_freq=170)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = dec(x)
        out.sum().backward()
        assert x.grad is not None


class TestEncoderDecoder:
    def test_round_trip_shape(self):
        """Encoder→Decoder should preserve batch and sequence dimensions."""
        enc = ColumnEncoder(n_freq=170, d_model=64)
        dec = ColumnDecoder(d_model=64, n_freq=170)
        x = torch.randn(4, 32, 170)
        z = enc(x)
        x_hat = dec(z)
        assert x_hat.shape == x.shape

    def test_round_trip_gradient(self):
        """Gradients should flow through encoder→decoder."""
        enc = ColumnEncoder(n_freq=170, d_model=64)
        dec = ColumnDecoder(d_model=64, n_freq=170)
        x = torch.randn(2, 16, 170, requires_grad=True)
        z = enc(x)
        x_hat = dec(z)
        loss = torch.nn.functional.mse_loss(x_hat, torch.zeros_like(x_hat))
        loss.backward()
        assert x.grad is not None
