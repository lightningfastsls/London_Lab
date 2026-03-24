"""Loss functions for VQ-VAE training.

Combines three loss terms:
1. Reconstruction loss (MSE between input and reconstructed spectrogram)
2. VQ commitment loss (from the quantizer, encourages encoder → codebook alignment)
3. Next-code prediction loss (cross-entropy for autoregressive code prediction)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAELoss(nn.Module):
    """Weighted combination of reconstruction, VQ, and next-code losses.

    Parameters
    ----------
    recon_weight:
        Weight for MSE reconstruction loss.
    vq_weight:
        Weight for VQ commitment loss (typically 1.0 since the quantizer
        already scales by commitment_weight).
    next_code_weight:
        Weight for cross-entropy next-code prediction loss.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        vq_weight: float = 1.0,
        next_code_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight
        self.vq_weight = vq_weight
        self.next_code_weight = next_code_weight

    def forward(self, model_output: dict, target: torch.Tensor) -> dict:
        """Compute all loss terms.

        Parameters
        ----------
        model_output:
            Dict from VQVAE.forward() with keys: x_recon, vq_loss,
            indices, next_code_logits.
        target:
            Original input spectrogram (B, seq_len, n_freq).

        Returns
        -------
        Dict with keys: total_loss, recon_loss, vq_loss, next_code_loss.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(model_output["x_recon"], target)

        # VQ loss (already computed by quantizer)
        vq_loss = model_output["vq_loss"]

        # Next-code prediction loss
        # Predict code at position t+1 from representation at position t
        logits = model_output["next_code_logits"][:, :-1]  # (B, S-1, K)
        targets = model_output["indices"][:, 1:]            # (B, S-1)

        next_code_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        total_loss = (
            self.recon_weight * recon_loss
            + self.vq_weight * vq_loss
            + self.next_code_weight * next_code_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "next_code_loss": next_code_loss,
        }
