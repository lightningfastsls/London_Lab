"""Global-code VQ-VAE for USV shape clustering (Mickey hypothesis test, 2026-07-03).

WHY THIS EXISTS
---------------
The prior contour-VAE and 6 image-objective variants failed to separate USV
*shape* families (best shape eta^2 = 0.105, gate 0.12; production plain image-VAE
= 0.099). Collaborator hypothesis: the missing ingredient is *vector
quantization* -- a discrete codebook bottleneck that forces the encoder to
commit each call to one of K prototypes.

This module implements the fair, apples-to-apples test of that hypothesis: a
VQ autoencoder with a SINGLE GLOBAL code per input, so the code index IS the
cluster assignment (directly comparable to the K=20 KMeans shape alphabets).

Two encoders/decoders are provided:
  * 1-D  : operates on the registered 50-point ridge (the substrate that carries
           shape after pitch/duration are removed). CPU-trainable on the box.
  * 2-D  : operates on contour-masked spectrogram patches (the *original*
           substrate the plain VAE failed on -- the literal Mickey test).
           Intended for the GPU rig.

DESIGN NOTES
  * Global single code: the encoder produces ONE latent vector z_e in R^D per
    input; VQ snaps it to the nearest of K codebook vectors. This is a
    clustering, not a spatial code grid. If we used a grid of codes (standard
    image VQ-VAE) we'd get great reconstructions but no per-call "class".
  * EMA codebook updates (van den Oord 2017 Appendix A.1) -- more stable than
    codebook-gradient; the known VQ failure mode is codebook collapse, so we
    also (a) report perplexity and (b) reset dead codes to random encoder
    outputs each epoch.
  * Straight-through estimator copies decoder gradients past the argmin.

No corpus constants are touched here -- this operates on already-registered
ridges / already-rendered patches, downstream of the STFT stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """Global vector quantizer with EMA codebook updates + dead-code reset.

    Expects a plain (B, D) batch of latent vectors -- one code per input.
    Returns the quantized vectors (with straight-through gradient), the code
    indices, the commitment loss, and the batch perplexity (effective # of
    codes in use; low perplexity == collapse).
    """

    def __init__(self, num_codes: int, dim: int, commitment: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.commitment = commitment
        self.decay = decay
        self.eps = eps
        # Codebook is a buffer (updated by EMA, not by autograd).
        embed = torch.randn(num_codes, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z_e: torch.Tensor):
        # z_e: (B, D). Distances to each codebook vector.
        d = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )  # (B, K)
        idx = d.argmin(1)                                   # (B,)
        onehot = F.one_hot(idx, self.num_codes).type_as(z_e)  # (B, K)
        z_q = onehot @ self.embed                            # (B, D)

        if self.training:
            # EMA update of codebook toward assigned encoder outputs.
            n = onehot.sum(0)                               # (K,) counts this batch
            embed_sum = onehot.t() @ z_e                    # (K, D)
            self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            total = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (total + self.num_codes * self.eps) * total
            )
            self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))

        # Commitment loss keeps the encoder near its chosen code.
        commit_loss = self.commitment * F.mse_loss(z_q.detach(), z_e)
        # Straight-through: gradients flow to encoder as if quantization were identity.
        z_q_st = z_e + (z_q - z_e).detach()

        # Perplexity: exp(entropy of codebook usage this batch).
        avg = onehot.mean(0)
        perplexity = torch.exp(-(avg * (avg + 1e-10).log()).sum())
        return z_q_st, idx, commit_loss, perplexity

    @torch.no_grad()
    def reset_dead_codes(self, z_e_pool: torch.Tensor, min_count: float = 1.0):
        """Reset codes that no input used to random samples from the encoder pool.

        Returns the number of codes reset (0 == healthy codebook)."""
        dead = self.cluster_size < min_count
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            pick = torch.randint(0, z_e_pool.shape[0], (n_dead,), device=z_e_pool.device)
            self.embed[dead] = z_e_pool[pick]
            self.embed_avg[dead] = z_e_pool[pick]
            self.cluster_size[dead] = 1.0
        return n_dead


# --------------------------------------------------------------------------- #
# 1-D encoder/decoder over the registered 50-point ridge
# --------------------------------------------------------------------------- #
class Encoder1D(nn.Module):
    def __init__(self, dim: int, length: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        )
        with torch.no_grad():
            flat = self.net(torch.zeros(1, 1, length)).flatten(1).shape[1]
        self.flat = flat
        self.fc = nn.Linear(flat, dim)

    def forward(self, x):  # x: (B, length)
        h = self.net(x.unsqueeze(1))
        return self.fc(h.flatten(1))


class Decoder1D(nn.Module):
    def __init__(self, dim: int, flat: int, length: int = 50):
        super().__init__()
        self.length = length
        self.fc = nn.Linear(dim, flat)
        # flat corresponds to (128, ceil(length/8)) -> reshape then upsample.
        self.c = 128
        self.t = flat // 128
        self.up = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.c, self.t)
        out = self.up(h).squeeze(1)                 # (B, ~length)
        # Match exact length (conv arithmetic can be off by a couple samples).
        if out.shape[1] != self.length:
            out = F.interpolate(out.unsqueeze(1), size=self.length,
                                mode="linear", align_corners=False).squeeze(1)
        return out


class VQVAE1D(nn.Module):
    """Global-code VQ-VAE over the registered ridge. Reconstruction = MSE."""

    def __init__(self, num_codes: int = 20, dim: int = 64, length: int = 50,
                 commitment: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.enc = Encoder1D(dim, length)
        self.vq = VectorQuantizerEMA(num_codes, dim, commitment, decay)
        self.dec = Decoder1D(dim, self.enc.flat, length)

    def forward(self, x):
        z_e = self.enc(x)
        z_q, idx, commit, ppl = self.vq(z_e)
        recon = self.dec(z_q)
        recon_loss = F.mse_loss(recon, x)
        return {
            "recon": recon, "recon_loss": recon_loss, "commit_loss": commit,
            "loss": recon_loss + commit, "z_e": z_e, "idx": idx, "perplexity": ppl,
        }

    @torch.no_grad()
    def encode(self, x):
        """Return (code_index, continuous_latent) for clustering/eval."""
        z_e = self.enc(x)
        d = (z_e.pow(2).sum(1, keepdim=True) - 2 * z_e @ self.vq.embed.t()
             + self.vq.embed.pow(2).sum(1))
        return d.argmin(1), z_e
