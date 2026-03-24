"""Tests for VQVAELoss."""

from __future__ import annotations

import torch

from usv_language.src.training.losses import VQVAELoss


class TestVQVAELoss:
    def test_output_keys(self):
        criterion = VQVAELoss()
        model_output = {
            "x_recon": torch.randn(2, 16, 170),
            "vq_loss": torch.tensor(0.1),
            "indices": torch.randint(0, 64, (2, 16)),
            "next_code_logits": torch.randn(2, 16, 64),
        }
        target = torch.randn(2, 16, 170)
        losses = criterion(model_output, target)

        assert "total_loss" in losses
        assert "recon_loss" in losses
        assert "vq_loss" in losses
        assert "next_code_loss" in losses

    def test_loss_positive(self):
        criterion = VQVAELoss()
        model_output = {
            "x_recon": torch.randn(2, 16, 170),
            "vq_loss": torch.tensor(0.1),
            "indices": torch.randint(0, 64, (2, 16)),
            "next_code_logits": torch.randn(2, 16, 64),
        }
        target = torch.randn(2, 16, 170)
        losses = criterion(model_output, target)

        assert losses["total_loss"].item() > 0
        assert losses["recon_loss"].item() > 0
        assert losses["next_code_loss"].item() > 0

    def test_weights_affect_total(self):
        """Changing weights should change the total loss."""
        model_output = {
            "x_recon": torch.randn(2, 16, 170),
            "vq_loss": torch.tensor(0.5),
            "indices": torch.randint(0, 64, (2, 16)),
            "next_code_logits": torch.randn(2, 16, 64),
        }
        target = torch.randn(2, 16, 170)

        loss_a = VQVAELoss(recon_weight=1.0, next_code_weight=0.0)(model_output, target)
        loss_b = VQVAELoss(recon_weight=1.0, next_code_weight=1.0)(model_output, target)
        assert loss_b["total_loss"].item() > loss_a["total_loss"].item()

    def test_backward(self):
        criterion = VQVAELoss()
        x_recon = torch.randn(2, 16, 170, requires_grad=True)
        model_output = {
            "x_recon": x_recon,
            "vq_loss": torch.tensor(0.1, requires_grad=True),
            "indices": torch.randint(0, 64, (2, 16)),
            "next_code_logits": torch.randn(2, 16, 64, requires_grad=True),
        }
        target = torch.randn(2, 16, 170)
        losses = criterion(model_output, target)
        losses["total_loss"].backward()
        assert x_recon.grad is not None
