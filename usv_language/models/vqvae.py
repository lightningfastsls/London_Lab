"""Hidden-state VQ-VAE for transformer interpretability.

Compresses the autoregressive transformer's hidden states (512-dim vectors)
into K=64 discrete codebook entries, discovering latent "concepts" the
transformer uses internally.  The transformer is frozen; this VQ-VAE is a
post-hoc analysis tool (Phase 2 of ADR-007 two-phase architecture).

Key differences from v1 VectorQuantizer (src/model/quantizer.py):
  - L2-normalized codebook and encoder outputs (unit hypersphere)
  - Commitment loss returned raw (caller applies beta scaling)
  - K-means codebook initialization from encoder outputs
  - Re-normalize codebook after each EMA update

Reference: van den Oord et al., "Neural Discrete Representation Learning" (2017)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VQVAEConfig:
    """Configuration for the hidden-state VQ-VAE.

    Parameters
    ----------
    d_model:
        Dimension of transformer hidden states (input to encoder).
    codebook_size:
        Number of discrete codes K.
    codebook_dim:
        Dimension of each codebook vector (encoder projects to this).
    commitment_weight:
        Beta: scaling factor for commitment loss (applied in forward, not
        inside the quantizer).
    ema_decay:
        Decay rate for exponential moving average codebook updates.
    dead_code_threshold:
        Reset a code if its EMA cluster size falls below this value.
    use_conv_encoder:
        If True, use Conv1d encoder (captures local temporal context).
        If False, use pointwise Linear encoder.
    conv_kernel_size:
        Kernel size for Conv1d encoder. Must be odd (for symmetric padding).
    """

    d_model: int = 512
    codebook_size: int = 64
    codebook_dim: int = 64
    commitment_weight: float = 0.25
    ema_decay: float = 0.99
    dead_code_threshold: float = 2.0
    use_conv_encoder: bool = True
    conv_kernel_size: int = 5

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {self.codebook_size}")
        if self.codebook_dim <= 0:
            raise ValueError(f"codebook_dim must be positive, got {self.codebook_dim}")
        if not (0 < self.ema_decay < 1):
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")
        if self.commitment_weight < 0:
            raise ValueError(
                f"commitment_weight must be >= 0, got {self.commitment_weight}"
            )
        if self.dead_code_threshold < 0:
            raise ValueError(
                f"dead_code_threshold must be >= 0, got {self.dead_code_threshold}"
            )
        if self.conv_kernel_size % 2 == 0:
            raise ValueError(
                f"conv_kernel_size must be odd for symmetric padding, "
                f"got {self.conv_kernel_size}"
            )


# ---------------------------------------------------------------------------
# Helper: dimension transposition for Conv1d in nn.Sequential
# ---------------------------------------------------------------------------


class _Transpose(nn.Module):
    """Swap last two dimensions — enables Conv1d inside nn.Sequential.

    Conv1d expects (B, C, S) but our data is (B, S, C).
    Insert _Transpose before Conv1d and after Conv1d to swap back.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-2, -1).contiguous()


# ---------------------------------------------------------------------------
# Vector Quantizer v2
# ---------------------------------------------------------------------------


class VectorQuantizerV2(nn.Module):
    """Vector Quantization with L2-normalized codebook and EMA updates.

    Differences from v1 (src/model/quantizer.py):
      - Codebook vectors are L2-normalized (unit hypersphere)
      - Returns raw commitment loss (NOT multiplied by beta)
      - Supports k-means initialization via ``initialize_from_data``
      - Re-normalizes codebook after every EMA update

    Parameters
    ----------
    codebook_size:
        Number of discrete codes K.
    codebook_dim:
        Dimension of each code vector.
    ema_decay:
        Decay for EMA codebook updates.
    dead_code_threshold:
        Reset code if EMA cluster size < this.
    epsilon:
        Laplace smoothing constant.
    """

    def __init__(
        self,
        codebook_size: int = 64,
        codebook_dim: int = 64,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 2.0,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.epsilon = epsilon

        # Codebook: (K, codebook_dim) — will be L2-normalized
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        # Normalize initial codebook to unit sphere
        with torch.no_grad():
            self.embedding.weight.copy_(
                F.normalize(self.embedding.weight, dim=1)
            )

        # EMA tracking buffers
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_sum", self.embedding.weight.clone())
        self.register_buffer("usage_count", torch.zeros(codebook_size, dtype=torch.long))

    def forward(
        self, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Quantize input to nearest (L2-normalized) codebook entries.

        Parameters
        ----------
        z:
            (B, S, codebook_dim) — L2-normalized encoder output.

        Returns
        -------
        z_q:
            (B, S, codebook_dim) — quantized vectors (straight-through grad).
        indices:
            (B, S) — codebook indices.
        commit_loss:
            Scalar raw MSE commitment loss (NOT multiplied by beta).
        metrics:
            Dict with ``perplexity``, ``codebook_usage``, ``avg_code_norm``.
        """
        B, S, D = z.shape
        z_flat = z.reshape(-1, D)  # (N, D)

        # L2 distances on unit sphere: ||a-b||^2 = 2 - 2*a·b
        # Equivalent to: z_flat.pow(2).sum(1,keepdim=True) - 2*z@e.T + e.pow(2).sum(1)
        # But since both are unit-norm, simplifies to 2 - 2*cosine_sim
        codebook = self.embedding.weight  # (K, D)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ codebook.T
            + codebook.pow(2).sum(dim=1, keepdim=True).T
        )

        # Nearest neighbor
        indices = distances.argmin(dim=1)  # (N,)
        z_q = self.embedding(indices)  # (N, D)

        # EMA codebook update (training only)
        if self.training:
            self._ema_update(z_flat, indices)
            self._reset_dead_codes(z_flat, indices)

        # Raw commitment loss (caller applies beta scaling)
        commit_loss = F.mse_loss(z_flat, z_q.detach())

        # Straight-through estimator
        z_q_st = z + (z_q.reshape(B, S, D) - z).detach()

        # Metrics
        encodings = F.one_hot(indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        codebook_usage = (avg_probs > 0).float().mean()
        avg_code_norm = codebook.norm(dim=1).mean()

        indices = indices.reshape(B, S)

        metrics = {
            "perplexity": perplexity,
            "codebook_usage": codebook_usage,
            "avg_code_norm": avg_code_norm,
        }

        return z_q_st, indices, commit_loss, metrics

    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook via EMA, then re-normalize to unit sphere."""
        encodings = F.one_hot(indices, self.codebook_size).float()

        # EMA cluster sizes
        cluster_size = encodings.sum(dim=0)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay,
        )

        # EMA embedding sums
        embedding_sum = encodings.T @ z_flat  # (K, D)
        self.ema_embedding_sum.mul_(self.ema_decay).add_(
            embedding_sum, alpha=1 - self.ema_decay,
        )

        # Laplace smoothing
        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )

        # Update codebook weights
        updated = self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1)

        # Re-normalize to unit sphere (weighted average of unit vectors != unit vector)
        self.embedding.weight.data.copy_(F.normalize(updated, dim=1))

    def _reset_dead_codes(
        self, z_flat: torch.Tensor, indices: torch.Tensor,
    ) -> None:
        """Re-initialize codes whose EMA cluster size is below threshold."""
        with torch.no_grad():
            # Check which codes are "dead" based on EMA cluster size
            dead_mask = self.ema_cluster_size < self.dead_code_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                # Replace with random encoder outputs (normalized)
                rand_idx = torch.randint(
                    0, z_flat.shape[0], (n_dead,), device=z_flat.device,
                )
                replacements = F.normalize(z_flat[rand_idx].detach(), dim=1)
                self.embedding.weight.data[dead_mask] = replacements
                self.ema_cluster_size[dead_mask] = self.dead_code_threshold
                self.ema_embedding_sum[dead_mask] = replacements
                self.usage_count[dead_mask] = 0

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors by index.

        Parameters
        ----------
        indices:
            LongTensor of any shape.

        Returns
        -------
        Codebook vectors, shape ``(*indices.shape, codebook_dim)``.
        """
        return self.embedding(indices)

    def initialize_from_data(self, data: torch.Tensor) -> None:
        """Initialize codebook via k-means on encoder outputs.

        Runs a simple k-means (20 iterations) to find good initial
        codebook vectors, then L2-normalizes them.

        Parameters
        ----------
        data:
            (N, codebook_dim) — L2-normalized encoder outputs.
        """
        K = self.codebook_size
        N = data.shape[0]

        # k-means++ initialization: pick first center randomly
        idx = torch.randint(0, N, (1,), device=data.device)
        centers = data[idx]  # (1, D)

        for _ in range(K - 1):
            dists = torch.cdist(data, centers).min(dim=1).values  # (N,)
            # D² weighting per Arthur & Vassilvitskii 2007 (true k-means++)
            dists_sq = dists.pow(2)
            probs = dists_sq / (dists_sq.sum() + 1e-10)
            next_idx = torch.multinomial(probs, 1)
            centers = torch.cat([centers, data[next_idx]], dim=0)

        # Lloyd's iterations
        for _ in range(20):
            dists = torch.cdist(data, centers)  # (N, K)
            assignments = dists.argmin(dim=1)
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    centers[k] = data[mask].mean(dim=0)

        # Normalize and set
        with torch.no_grad():
            centers = F.normalize(centers, dim=1)
            self.embedding.weight.copy_(centers)
            self.ema_embedding_sum.copy_(centers)
            self.ema_cluster_size.fill_(1.0)


# ---------------------------------------------------------------------------
# Hidden State VQ-VAE
# ---------------------------------------------------------------------------


class HiddenStateVQVAE(nn.Module):
    """VQ-VAE that compresses transformer hidden states into discrete codes.

    Architecture:
        Encoder: hidden_state (d_model) -> bottleneck (codebook_dim) -> L2-norm
        Quantizer: bottleneck -> nearest codebook entry (straight-through)
        Decoder: codebook entry (codebook_dim) -> reconstructed (d_model)

    The encoder can use either Conv1d (captures local temporal context via
    kernel) or pointwise Linear layers.

    Parameters
    ----------
    config:
        VQVAEConfig with architecture hyperparameters.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self.config = config

        # --- Encoder ---
        if config.use_conv_encoder:
            pad = config.conv_kernel_size // 2
            self.encoder = nn.Sequential(
                _Transpose(),  # (B, S, d_model) -> (B, d_model, S)
                nn.Conv1d(config.d_model, 256, config.conv_kernel_size, padding=pad),
                nn.GELU(),
                _Transpose(),  # (B, 256, S) -> (B, S, 256)
                nn.Linear(256, config.codebook_dim),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(config.d_model, 256),
                nn.GELU(),
                nn.Linear(256, config.codebook_dim),
            )

        # --- Quantizer ---
        self.quantizer = VectorQuantizerV2(
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            ema_decay=config.ema_decay,
            dead_code_threshold=config.dead_code_threshold,
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(config.codebook_dim, 256),
            nn.GELU(),
            nn.Linear(256, config.d_model),
        )

    def forward(
        self, hidden_states: torch.Tensor,
    ) -> dict:
        """Encode, quantize, and reconstruct hidden states.

        Parameters
        ----------
        hidden_states:
            (B, S, d_model) — transformer hidden states.

        Returns
        -------
        Dict with:
            recon: (B, S, d_model) — reconstructed hidden states
            indices: (B, S) — codebook indices
            recon_loss: scalar — MSE reconstruction loss
            commit_loss: scalar — raw commitment loss (before beta)
            total_loss: scalar — recon_loss + beta * commit_loss
            perplexity: scalar — codebook usage diversity
            codebook_usage: scalar — fraction of codes used
        """
        # Encode
        z_e = self.encoder(hidden_states)  # (B, S, codebook_dim)

        # L2 normalize encoder output
        z_e = F.normalize(z_e, dim=-1)

        # Quantize
        z_q, indices, commit_loss, metrics = self.quantizer(z_e)

        # Decode
        recon = self.decoder(z_q)  # (B, S, d_model)

        # Losses
        recon_loss = F.mse_loss(recon, hidden_states)
        total_loss = recon_loss + self.config.commitment_weight * commit_loss

        return {
            "recon": recon,
            "indices": indices,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "total_loss": total_loss,
            "perplexity": metrics["perplexity"],
            "codebook_usage": metrics["codebook_usage"],
        }

    def initialize_codebook_from_data(
        self, hidden_states: torch.Tensor,
    ) -> None:
        """Initialize codebook via k-means on encoder outputs.

        Runs the encoder on the provided hidden states, then uses k-means
        to set the codebook to good initial positions.

        Parameters
        ----------
        hidden_states:
            (N, d_model) or (B, S, d_model) — sample hidden states.
        """
        self.eval()
        with torch.no_grad():
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(0)
            z_e = self.encoder(hidden_states)
            z_e = F.normalize(z_e, dim=-1)
            z_e_flat = z_e.reshape(-1, self.config.codebook_dim)
            self.quantizer.initialize_from_data(z_e_flat)

    def encode_to_codes(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to discrete code indices (inference helper).

        Parameters
        ----------
        hidden_states:
            (B, S, d_model) — transformer hidden states.

        Returns
        -------
        indices:
            (B, S) — codebook indices.
        """
        self.eval()
        with torch.no_grad():
            z_e = self.encoder(hidden_states)
            z_e = F.normalize(z_e, dim=-1)
            _, indices, _, _ = self.quantizer(z_e)
            return indices

    def count_parameters(self) -> int:
        """Count trainable parameters (excludes EMA buffers)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
