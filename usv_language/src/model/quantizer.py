"""Vector Quantizer with EMA codebook updates and dead code reset.

Implements the VQ-VAE discrete bottleneck from scratch:
- Nearest-neighbor codebook lookup
- Straight-through gradient estimator (gradients bypass quantization)
- Exponential moving average (EMA) codebook updates (more stable than gradient-based)
- Dead code reset: re-initializes unused codes from random encoder outputs

Reference: van den Oord et al., "Neural Discrete Representation Learning" (2017)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates.

    Parameters
    ----------
    codebook_size:
        Number of discrete codes (K). Default 512.
    d_model:
        Dimension of each code vector. Must match encoder output dim.
    ema_decay:
        Decay rate for EMA codebook updates. Higher = slower updates.
    commitment_weight:
        Beta: how strongly encoder output is pulled toward codebook entry.
    dead_code_threshold:
        Reset a code if unused for this many forward passes.
    epsilon:
        Small constant for Laplace smoothing in EMA updates.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        d_model: int = 64,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        dead_code_threshold: int = 2,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight
        self.dead_code_threshold = dead_code_threshold
        self.epsilon = epsilon

        # Codebook: (K, d_model)
        self.embedding = nn.Embedding(codebook_size, d_model)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking buffers (not model parameters)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_sum", self.embedding.weight.clone())
        self.register_buffer("usage_count", torch.zeros(codebook_size, dtype=torch.long))

    def forward(self, z: torch.Tensor) -> dict:
        """Quantize encoder output to nearest codebook entries.

        Parameters
        ----------
        z: (B, seq_len, d_model) — continuous encoder output

        Returns
        -------
        Dict with keys:
            z_q: (B, seq_len, d_model) — quantized vectors (with straight-through grad)
            indices: (B, seq_len) — codebook indices
            vq_loss: scalar — commitment loss
            perplexity: scalar — codebook usage diversity measure
            codebook_usage: scalar — fraction of codes used in this batch
        """
        B, S, D = z.shape
        z_flat = z.reshape(-1, D)  # (B*S, D)

        # Compute distances: ||z - e||^2 = ||z||^2 - 2*z·e + ||e||^2
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).T
        )

        # Nearest neighbor lookup
        indices = distances.argmin(dim=1)  # (B*S,)
        z_q = self.embedding(indices)  # (B*S, D)

        # EMA codebook update (training only)
        if self.training:
            self._ema_update(z_flat, indices)
            self._reset_dead_codes(z_flat)

        # Commitment loss: pull encoder output toward codebook entry
        commitment_loss = F.mse_loss(z_flat, z_q.detach())
        vq_loss = self.commitment_weight * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q.reshape(B, S, D) - z).detach()

        # Perplexity: measures effective codebook usage
        encodings = F.one_hot(indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        codebook_usage = (avg_probs > 0).float().mean()

        indices = indices.reshape(B, S)

        return {
            "z_q": z_q,
            "indices": indices,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "codebook_usage": codebook_usage,
        }

    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook embeddings via exponential moving average."""
        encodings = F.one_hot(indices, self.codebook_size).float()  # (N, K)

        # Update cluster sizes
        cluster_size = encodings.sum(dim=0)  # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)

        # Update embedding sums
        embedding_sum = encodings.T @ z_flat  # (K, D)
        self.ema_embedding_sum.mul_(self.ema_decay).add_(embedding_sum, alpha=1 - self.ema_decay)

        # Laplace smoothing to avoid division by zero
        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )

        # Update codebook
        self.embedding.weight.data.copy_(
            self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1)
        )

    def _reset_dead_codes(self, z_flat: torch.Tensor) -> None:
        """Re-initialize codes that haven't been used recently."""
        # Track usage
        with torch.no_grad():
            distances = (
                z_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_flat @ self.embedding.weight.T
                + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).T
            )
            used_indices = distances.argmin(dim=1).unique()
            mask = torch.zeros(self.codebook_size, dtype=torch.bool, device=z_flat.device)
            mask[used_indices] = True

            # Increment unused counter, reset used counter
            self.usage_count += 1
            self.usage_count[mask] = 0

            # Reset dead codes
            dead_mask = self.usage_count >= self.dead_code_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                # Sample random encoder outputs as replacements
                rand_indices = torch.randint(0, z_flat.shape[0], (n_dead,), device=z_flat.device)
                self.embedding.weight.data[dead_mask] = z_flat[rand_indices].detach()
                self.ema_cluster_size[dead_mask] = 0
                self.ema_embedding_sum[dead_mask] = z_flat[rand_indices].detach()
                self.usage_count[dead_mask] = 0

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors by index.

        Parameters
        ----------
        indices: LongTensor of any shape

        Returns
        -------
        Codebook vectors, shape (*indices.shape, d_model)
        """
        return self.embedding(indices)
