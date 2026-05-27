"""Pathway B+A — hybrid contrastive-encoder + VAE-decoder + ridge-derivative
shape model (rig, GPU).

Canonical spec: ``PLAN_geometric_shape_clustering_vae.md`` §3 Option B+A and
``docs/handoffs/2026-05-27_shape-vae-BA-hybrid.md``.

The bet (riskiest of the three sibling pathways): re-introducing a
reconstruction objective — which the 2026-05-26 denoised retrain showed spends
latent capacity on pitch/duration pixel-variance (shape eta2 0.081) — is worth
it IF the contrastive + shift-augmentation + latent-consistency terms dominate
it and pull the latent toward *shape*. Mitigations baked in:
  * weights heavily contrastive-dominant; recon/KL/derivative ANNEALED in late
    (see :func:`annealed_weights`), so reconstruction cannot hijack early
    training;
  * KL weight ``beta`` kept LOW by default (the dead-end used beta=1.0; KL
    over-smoothing is the prime suspect);
  * pitch/time-shift augmentation realizes the dF/dt invariance idea as a
    contrastive positive pair (shift a call in pitch/time -> same shape).

Design decisions locked with the user (2026-05-27):
  * loss-weight schedule = contrastive-dominant staged anneal;
  * differentiable ridge = soft-argmax expected-frequency proxy (NOT the
    non-differentiable Viterbi ``track_ridge``; that supplies the *target*
    only, cached offline by ``extract_ridge_targets_v3.py``);
  * tuning budget = small sweep (3-4 runs on 5970).

Base architecture is the FROZEN ``ImageVAE`` from ``scripts/train_contour_vae_v2.py``
(256^2, latent_dim=32). This script imports it — it never redefines or
overwrites it.

NOT a unit-tested path: the dataset/train loop run on the rig with the denoised
patch corpus and the cached ridge targets (gated launch). The pure functions
(config, soft_argmax_ridge, nt_xent, latent_consistency, derivative_loss,
augment_pitch_time_shift, annealed_weights, hybrid_loss) are covered by
``tests/test_shape_vae_v3_hybrid.py`` (CPU, synthetic).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path bootstrap — make the repo root and src/ importable regardless of cwd, so
# this runs standalone on the rig AND imports cleanly under pytest.
# scripts/experiments/<this>.py -> parents[2] is the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from usv_spectrogram import corpus  # noqa: E402

# Reuse the frozen base — do NOT redefine these.
from scripts.train_contour_vae_v2 import (  # noqa: E402
    ImageVAE,
    ImageVAEConfig,
    PaddingSpec,
    _compute_band_slice,
    image_vae_loss,
)


# ===========================================================================
# Config
# ===========================================================================


@dataclass(frozen=True)
class ShapeVAEv3Config:
    """Hyperparameters for the B+A hybrid shape model.

    Loss weights are the *full* (post-anneal) targets; :func:`annealed_weights`
    ramps ``lambda_recon``/``beta``/``lambda_deriv`` from 0 to these values over
    the anneal window while ``lambda_nt``/``lambda_lc`` stay constant.

    Defaults are intentionally contrastive-dominant with a LOW ``beta`` — see
    the module docstring for why.
    """

    latent_dim: int = 32
    image_size: int = 256

    # Loss weights (full / post-anneal targets).
    lambda_nt: float = 1.0       # NT-Xent contrastive (the clustering driver)
    lambda_recon: float = 0.05   # reconstruction (BCE) — kept modest
    beta: float = 0.05           # KL weight — LOW by design
    lambda_lc: float = 1.0       # latent-consistency ||z(x) - z(shift(x))||^2
    lambda_deriv: float = 0.1    # ridge-derivative MSE(dF/dt_decoded, dF/dt_true)

    nt_temperature: float = 0.2

    # Augmentation (the dF/dt invariance, realized as the positive pair).
    max_df_hz: float = 15_000.0
    max_dt_frames: int = 20
    # RESERVED — declared per the spec ("optional" time-warp) but NOT yet
    # applied by augment_pitch_time_shift. Wiring it in would have to leave the
    # zero-shift identity case intact (see test_augment_zero_shift_unchanged);
    # deferred to a future sweep knob (see successor handoff). Stored only so a
    # later implementation has a stable config slot.
    time_warp_range: Tuple[float, float] = (0.9, 1.1)

    # Contrastive-dominant staged anneal: recon/KL/deriv ramp 0 -> full over
    # [recon_anneal_start, recon_anneal_start + recon_anneal_epochs].
    recon_anneal_start: int = 10
    recon_anneal_epochs: int = 20

    # Soft-argmax temperature for the differentiable in-loss ridge proxy.
    softargmax_temp: float = 1.0

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        for name in ("lambda_nt", "lambda_recon", "lambda_lc", "lambda_deriv"):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if self.max_df_hz < 0:
            raise ValueError(f"max_df_hz must be >= 0, got {self.max_df_hz}")
        if self.max_dt_frames < 0:
            raise ValueError(f"max_dt_frames must be >= 0, got {self.max_dt_frames}")
        if self.recon_anneal_epochs < 1:
            raise ValueError(
                f"recon_anneal_epochs must be >= 1, got {self.recon_anneal_epochs}"
            )
        # image_size must be a power of 2 >= 16 (4 stride-2 convs in ImageVAE).
        if self.image_size <= 0 or (self.image_size & (self.image_size - 1)) != 0:
            raise ValueError(
                f"image_size must be a positive power of 2, got {self.image_size}"
            )
        if self.image_size < 16:
            raise ValueError(
                f"image_size must be >= 16 for the 4-stride-2 encoder, got {self.image_size}"
            )


# ===========================================================================
# Pure loss / augmentation components (unit-tested)
# ===========================================================================


def soft_argmax_ridge(
    magnitude: torch.Tensor,
    freqs_hz: torch.Tensor,
    temp: float = 1.0,
) -> torch.Tensor:
    """Differentiable expected-frequency ridge per time column.

    For each time column, softmax over the frequency rows (scaled by ``temp``)
    and take the frequency expectation. This is the differentiable stand-in for
    the Viterbi ``track_ridge`` so the derivative term can backprop through the
    decoded magnitude — the real Viterbi ridge is used only for the *target*.

    Parameters
    ----------
    magnitude:
        ``(B, 1, H, W)`` or ``(B, H, W)`` non-negative magnitude. H = frequency
        rows, W = time columns.
    freqs_hz:
        ``(H,)`` frequency (Hz) of each row.
    temp:
        Softmax temperature. Lower -> sharper (closer to a hard argmax).

    Returns
    -------
    ``(B, W)`` expected frequency (Hz) per time column. As a convex combination
    of ``freqs_hz`` it always lies within ``[freqs_hz.min(), freqs_hz.max()]``.
    """
    if magnitude.dim() == 4:
        mag = magnitude.squeeze(1)  # (B, H, W)
    elif magnitude.dim() == 3:
        mag = magnitude
    else:
        raise ValueError(
            f"magnitude must be (B,1,H,W) or (B,H,W), got shape {tuple(magnitude.shape)}"
        )
    # Softmax over the frequency (row) axis; torch's softmax subtracts the max
    # internally so large ``mag/temp`` values are numerically stable.
    weights = F.softmax(mag / temp, dim=1)  # (B, H, W)
    f = freqs_hz.to(dtype=mag.dtype, device=mag.device).view(1, -1, 1)  # (1, H, 1)
    return (weights * f).sum(dim=1)  # (B, W)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    """NT-Xent contrastive loss over the positive pair ``(z1[i], z2[i])``.

    Port of the M9 reference (``scripts/experiments/rig_M9_contrastive.py``):
    L2-normalize the 2N embeddings, build the cosine-similarity matrix scaled by
    ``temp``, mask the self-similarity diagonal, and cross-entropy each row
    against its positive partner. Returns the mean (scalar).
    """
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    n = z1.shape[0]
    sim = z @ z.T / temp
    sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def latent_consistency(z: torch.Tensor, z_aug: torch.Tensor) -> torch.Tensor:
    """Mean squared L2 distance between a latent and its shifted-view latent.

    ``mean_b ||z_b - z_aug_b||^2`` — an explicit invariance pressure: shifting a
    call in pitch/time should not move its embedding. Exactly 0 when
    ``z_aug == z``.
    """
    return ((z - z_aug) ** 2).sum(dim=-1).mean()


def derivative_loss(
    ridge_decoded: torch.Tensor,
    ridge_true: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between the finite-difference dF/dt of the decoded and true ridges.

    Parameters
    ----------
    ridge_decoded, ridge_true:
        ``(B, W)`` frequency-per-column trajectories.
    valid_mask:
        Optional ``(B, W-1)`` boolean mask aligned to the ``torch.diff`` output
        (one shorter than W). When given, only masked-True columns contribute;
        if every column is masked out the loss is a NaN-safe 0.

    dF/dt is invariant to absolute pitch (a constant offset differentiates away)
    and to time position, which is the whole point of operating on the
    derivative rather than the raw ridge.
    """
    d_dec = torch.diff(ridge_decoded, dim=-1)  # (B, W-1)
    d_true = torch.diff(ridge_true, dim=-1)
    sq = (d_dec - d_true) ** 2
    if valid_mask is None:
        return sq.mean()
    m = valid_mask.to(dtype=sq.dtype)
    denom = m.sum()
    return (sq * m).sum() / (denom + 1e-8)


def _shift_image(
    x: torch.Tensor,
    df_bins: torch.Tensor,
    dt_frames: torch.Tensor,
) -> torch.Tensor:
    """Non-wrapping per-sample integer shift on a (B,C,H,W) batch.

    Positive ``df_bins`` shifts content toward HIGHER row indices (= higher
    frequency, given freqs ascend with row). Positive ``dt_frames`` shifts
    content toward later time columns. Energy that leaves the canvas is dropped
    (zero-filled), never wrapped — implemented as a sub-rectangle copy into a
    zeroed output.
    """
    b, c, h, w = x.shape
    out = torch.zeros_like(x)
    for i in range(b):
        df = int(df_bins[i].item())
        dt = int(dt_frames[i].item())
        if df >= 0:
            sf0, sf1, dfa, dfb = 0, h - df, df, h
        else:
            sf0, sf1, dfa, dfb = -df, h, 0, h + df
        if dt >= 0:
            st0, st1, dta, dtb = 0, w - dt, dt, w
        else:
            st0, st1, dta, dtb = -dt, w, 0, w + dt
        out[i, :, dfa:dfb, dta:dtb] = x[i, :, sf0:sf1, st0:st1]
    return out


def augment_pitch_time_shift(
    x: torch.Tensor,
    ridge_lo_hz: torch.Tensor,
    ridge_hi_hz: torch.Tensor,
    cfg: ShapeVAEv3Config,
    freq_per_bin_hz: float,
    band_lo_hz: float,
    band_hi_hz: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pitch + time shift augmentation with an in-band guarantee.

    The positive view for the contrastive loss: shift the whole call in pitch
    (vertical) and time (horizontal). The pitch shift is clamped per-sample so
    the call's ridge — spanning ``[ridge_lo_hz, ridge_hi_hz]`` — never crosses
    the corpus USV band ``[band_lo_hz, band_hi_hz]``.

    Parameters
    ----------
    x:
        ``(B, 1, H, W)`` image batch.
    ridge_lo_hz, ridge_hi_hz:
        ``(B,)`` min/max frequency of each sample's true ridge (Hz).
    cfg:
        Provides ``max_df_hz`` / ``max_dt_frames``.
    freq_per_bin_hz:
        Hz per image row (= STFT bin width).
    band_lo_hz, band_hi_hz:
        Corpus USV band edges (import from ``corpus`` at the call site).

    Returns
    -------
    ``(x_aug, df_bins, dt_frames)`` — augmented batch and the integer shifts
    actually applied per sample (``df_bins``/``dt_frames`` shape ``(B,)``).
    """
    b = x.shape[0]
    df_bins = torch.zeros(b, dtype=torch.long)
    dt_frames = torch.zeros(b, dtype=torch.long)

    max_df_bins = (
        int(round(cfg.max_df_hz / freq_per_bin_hz)) if freq_per_bin_hz > 0 else 0
    )
    max_dt = int(cfg.max_dt_frames)

    for i in range(b):
        # Pitch shift, clamped to the in-band feasible integer interval.
        # ceil on the lower bound / floor on the upper bound guarantee the
        # INTEGER shift never rounds the ridge across a band edge.
        if max_df_bins > 0 and freq_per_bin_hz > 0:
            lo_allowed = math.ceil(
                (band_lo_hz - float(ridge_lo_hz[i])) / freq_per_bin_hz
            )
            hi_allowed = math.floor(
                (band_hi_hz - float(ridge_hi_hz[i])) / freq_per_bin_hz
            )
            lo = max(-max_df_bins, lo_allowed)
            hi = min(max_df_bins, hi_allowed)
            if hi >= lo:
                df_bins[i] = torch.randint(
                    lo, hi + 1, (1,), generator=generator
                ).item()
        # Time shift, no band constraint.
        if max_dt > 0:
            dt_frames[i] = torch.randint(
                -max_dt, max_dt + 1, (1,), generator=generator
            ).item()

    x_aug = _shift_image(x, df_bins, dt_frames)
    return x_aug, df_bins, dt_frames


def annealed_weights(cfg: ShapeVAEv3Config, epoch: int) -> Dict[str, float]:
    """Contrastive-dominant staged anneal of the five loss weights.

    ``lambda_nt`` and ``lambda_lc`` are constant for all epochs (the encoder
    learns invariant structure from the start). ``lambda_recon``, ``beta`` and
    ``lambda_deriv`` ramp linearly from 0 (at ``recon_anneal_start``) to their
    full cfg values (at ``recon_anneal_start + recon_anneal_epochs``), so the
    reconstruction objective cannot hijack the latent before clusters form.
    """
    frac = (epoch - cfg.recon_anneal_start) / cfg.recon_anneal_epochs
    frac = min(1.0, max(0.0, frac))
    return {
        "lambda_nt": cfg.lambda_nt,
        "lambda_lc": cfg.lambda_lc,
        "lambda_recon": frac * cfg.lambda_recon,
        "beta": frac * cfg.beta,
        "lambda_deriv": frac * cfg.lambda_deriv,
    }


def hybrid_loss(
    model_out: Dict[str, torch.Tensor],
    x: torch.Tensor,
    x_aug: torch.Tensor,
    ridge_true: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    freqs_hz: torch.Tensor,
    weights: Dict[str, float],
    cfg: ShapeVAEv3Config,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Assemble the hybrid loss from a model_out dict — the ASSEMBLY REFERENCE.

    ``model_out`` carries ``x_recon, mu, logvar, z, z_aug`` (z / z_aug are the
    contrastive embeddings of the clean and shifted views). Returns
    ``(total, components)`` where ``components`` holds the UNWEIGHTED value of
    each term keyed by its weight name, and ``total = sum_k weights[k] *
    components[k]``.

    SCOPE / why the rig train loop does NOT call this:
      * This function's derivative term takes a FULL ridge target
        (``ridge_true``, shape (B, W)) and diffs it internally. The rig train
        loop instead consumes the SHARED Track-0 cache, which stores the
        *pre-differenced* dF/dt target (Hz/frame) — so it computes the masked
        dF/dt MSE inline (against ``dfdt_true``) rather than through here.
        ``hybrid_loss`` is therefore the reference assembler used by the unit
        tests and by any caller that holds a full ridge target (e.g. eval).
      * BAND-INPUT CONTRACT: pass a BAND-RESTRICTED ``x_recon`` and a matching
        ``freqs_hz`` of the same row count. Do NOT pass the zero-padded 256-row
        canvas — the soft-argmax would put softmax mass on the padded rows and
        bias the expected frequency toward the image centre.

    ``x_aug`` is accepted for API symmetry; the augmented view's embedding is
    already encoded into ``model_out['z_aug']`` upstream.
    """
    x_recon = model_out["x_recon"]
    mu = model_out["mu"]
    logvar = model_out["logvar"]
    z = model_out["z"]
    z_aug = model_out["z_aug"]

    # recon (BCE, summed-per-pixel / batch) and raw KL from the frozen base.
    _total_ignored, recon, kl = image_vae_loss(x_recon, x, mu, logvar, beta=1.0)

    nt = nt_xent(z, z_aug, cfg.nt_temperature)
    lc = latent_consistency(z, z_aug)
    ridge_dec = soft_argmax_ridge(x_recon, freqs_hz, cfg.softargmax_temp)
    deriv = derivative_loss(ridge_dec, ridge_true, valid_mask)

    components: Dict[str, torch.Tensor] = {
        "lambda_nt": nt,
        "lambda_recon": recon,
        "beta": kl,
        "lambda_lc": lc,
        "lambda_deriv": deriv,
    }
    total = (
        weights["lambda_nt"] * nt
        + weights["lambda_recon"] * recon
        + weights["beta"] * kl
        + weights["lambda_lc"] * lc
        + weights["lambda_deriv"] * deriv
    )
    return total, components


# ===========================================================================
# Model wrapper (built on the frozen ImageVAE)
# ===========================================================================


class ShapeVAEv3Hybrid(nn.Module):
    """ImageVAE wrapped to expose the contrastive + generative latents.

    The decoder reconstructs from a sampled ``z`` (the generative / navigable
    path). The contrastive (NT-Xent) and latent-consistency terms operate on the
    posterior MEANS (``mu``, ``mu_aug``) — deterministic embeddings are the
    standard, lower-variance choice for clustering.
    """

    def __init__(self, cfg: ShapeVAEv3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.vae = ImageVAE(
            ImageVAEConfig(image_size=cfg.image_size, latent_dim=cfg.latent_dim)
        )

    def forward(
        self, x: torch.Tensor, x_aug: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.vae.encode(x)
        z_sample = self.vae.reparameterize(mu, logvar)
        x_recon = self.vae.decode(z_sample)
        mu_aug, _ = self.vae.encode(x_aug)
        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": mu,        # contrastive embedding (posterior mean)
            "z_aug": mu_aug,
        }


# ===========================================================================
# Dataset — denoised patches + cached true ridge F(t)
# ===========================================================================


class DenoisedPatchRidgeDataset(torch.utils.data.Dataset):
    """Denoised power patches + the SHARED Track-0 dF/dt target cache.

    Reuses the cache produced by ``extract_ridge_targets_v3.py`` (the shared
    Track-0 artifact, also consumed by Pathway A): keys ``dFdt_true (N, T-1)``
    and ``valid_mask (N, T-1)``, in **kHz/frame**. We convert dF/dt to Hz/frame
    so it matches the soft-argmax ridge proxy (which returns Hz). The raw
    absolute ridge is NOT in that cache, so the augmentation's in-band span
    ``[ridge_lo_hz, ridge_hi_hz]`` is derived cheaply from each patch's own
    energy extent (a conservative over-estimate — exactly what a safety clamp
    wants). This keeps Track-0 a single shared computation: no duplicate
    24-min ridge pass, no edit to the sibling pathway's extractor.

    Each item returns ``(image, dfdt_true_hz, dfdt_valid, ridge_lo_hz,
    ridge_hi_hz)``:
      * ``image``: band-cropped, log1p, per-patch min/max, zero-padded to
        ``image_size`` (same preprocessing as the frozen ``MaskedPatchDataset``),
        shape ``(1, image_size, image_size)``.
      * ``dfdt_true_hz``: cached true dF/dt over real time columns (Hz/frame),
        length ``T-1``.
      * ``dfdt_valid``: bool mask (length ``T-1``) — True where both ridge
        endpoints were active; masks the derivative loss.
      * ``ridge_lo_hz`` / ``ridge_hi_hz``: energy-extent frequency span (Hz),
        for the in-band augmentation clamp.

    Never full-scans ``patches.npz`` (the box OOM-crashes on the lab corpus);
    slices per ``__getitem__`` off a memmap.
    """

    # Energy floor for the band-row span used by the augmentation clamp.
    _ENERGY_FRACTION = 0.02

    def __init__(
        self,
        patches: np.ndarray,          # (N, F, T) power, may be a memmap
        dfdt_true_hz: np.ndarray,     # (N, T-1) true dF/dt target, Hz/frame
        dfdt_valid: np.ndarray,       # (N, T-1) bool validity mask
        band_freqs_hz: np.ndarray,    # (F_band,) Hz of each band row
        band_slice: slice,
        padding: PaddingSpec,
    ) -> None:
        self._patches = patches
        self._dfdt = dfdt_true_hz
        self._dfdt_valid = dfdt_valid
        self._band_freqs_hz = np.asarray(band_freqs_hz, dtype=np.float32)
        self._band_slice = band_slice
        self._padding = padding

    def __len__(self) -> int:
        return self._patches.shape[0]

    def __getitem__(self, idx: int):
        raw = np.asarray(self._patches[idx, self._band_slice, :])  # (F_band, T_in) power
        x = np.log1p(raw).astype(np.float32)
        p_min, p_max = float(x.min()), float(x.max())
        x_n = (x - p_min) / max(p_max - p_min, 1e-6)
        x_p = self._padding.pad(x_n).astype(np.float32)
        image = torch.from_numpy(x_p[np.newaxis, :, :].copy())

        # In-band augmentation span from energy extent (conservative): band rows
        # whose per-row time-max clears a fraction of the patch max.
        row_energy = raw.max(axis=1)
        rmax = float(row_energy.max()) if row_energy.size else 0.0
        active_rows = np.where(row_energy > self._ENERGY_FRACTION * rmax)[0] if rmax > 0 else np.array([], dtype=int)
        if active_rows.size:
            lo = float(self._band_freqs_hz[active_rows].min())
            hi = float(self._band_freqs_hz[active_rows].max())
        else:
            lo = hi = float(corpus.USV_FREQ_MIN_HZ)

        dfdt = np.asarray(self._dfdt[idx], dtype=np.float32)
        dvalid = np.asarray(self._dfdt_valid[idx], dtype=bool)

        return (
            image,
            torch.from_numpy(dfdt.copy()),
            torch.from_numpy(dvalid.copy()),
            torch.tensor(lo, dtype=torch.float32),
            torch.tensor(hi, dtype=torch.float32),
        )


# ===========================================================================
# Training
# ===========================================================================


def _load_freqs_hz(npz: Any, n_freq_bins: int) -> np.ndarray:
    """Frequency (Hz) of each patch row — from the npz if present, else derive
    from the corpus STFT grid (rfftfreq for n_fft=512 @ 300 kHz -> 257 bins)."""
    if "freqs_kHz" in npz:
        return np.asarray(npz["freqs_kHz"], dtype=float) * 1000.0
    if "freqs_hz" in npz:
        return np.asarray(npz["freqs_hz"], dtype=float)
    freqs = np.fft.rfftfreq(corpus.STFT_N_FFT, d=1.0 / corpus.SAMPLE_RATE_HZ)
    if len(freqs) != n_freq_bins:
        raise RuntimeError(
            f"derived {len(freqs)} STFT freq bins but patches have {n_freq_bins}; "
            "store freqs_kHz in the npz to disambiguate."
        )
    return freqs.astype(float)


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    cfg = ShapeVAEv3Config(
        latent_dim=args.latent_dim,
        lambda_nt=args.lambda_nt,
        lambda_recon=args.lambda_recon,
        beta=args.beta,
        lambda_lc=args.lambda_lc,
        lambda_deriv=args.lambda_deriv,
        nt_temperature=args.nt_temperature,
        max_df_hz=args.max_df_hz,
        max_dt_frames=args.max_dt_frames,
        recon_anneal_start=args.recon_anneal_start,
        recon_anneal_epochs=args.recon_anneal_epochs,
        softargmax_temp=args.softargmax_temp,
    )

    dev = args.device if torch.cuda.is_available() else "cpu"
    patches_npz = np.load(args.patches_npz, mmap_mode="r", allow_pickle=False)
    patches = patches_npz["patches"]
    n, f_bins, t_in = patches.shape
    freqs_hz = _load_freqs_hz(patches_npz, f_bins)
    freqs_khz = freqs_hz / 1000.0
    band_slice, b0, b1 = _compute_band_slice(freqs_khz)
    band_freqs_hz = torch.from_numpy(freqs_hz[band_slice].astype(np.float32))
    f_band = b1 - b0
    padding = PaddingSpec.for_shape(f_band, t_in, cfg.image_size)
    freq_per_bin_hz = float(corpus.SAMPLE_RATE_HZ) / float(corpus.STFT_N_FFT)
    band_lo_hz = float(corpus.USV_FREQ_MIN_HZ)
    band_hi_hz = float(corpus.USV_FREQ_MAX_HZ)
    # Padded band occupies rows [pad_f_top : pad_f_top + f_band] in the image.
    row0, row1 = padding.pad_f_top, padding.pad_f_top + f_band

    # SHARED Track-0 cache (also consumed by Pathway A): dF/dt target in
    # kHz/frame -> convert to Hz/frame to match the soft-argmax ridge (Hz).
    ridge_cache = np.load(args.ridge_npz, mmap_mode="r")
    dfdt_true_hz = np.asarray(ridge_cache["dFdt_true"]) * 1000.0
    dfdt_valid = np.asarray(ridge_cache["valid_mask"])
    if dfdt_true_hz.shape[0] != n:
        raise RuntimeError(
            f"ridge cache rows ({dfdt_true_hz.shape[0]}) != patches ({n}); rerun "
            "extract_ridge_targets_v3.py on the same patches.npz."
        )
    if dfdt_true_hz.shape[1] != t_in - 1:
        raise RuntimeError(
            f"ridge cache dF/dt width ({dfdt_true_hz.shape[1]}) != T-1 ({t_in - 1})."
        )

    # ----- PARAM PRINT (lab discipline: params/thresholds/row counts) -----
    print(
        "[PARAM] shape_vae_v3_hybrid "
        f"N={n} f_bins={f_bins} band=[{b0}:{b1}]({f_band}) t_in={t_in} "
        f"image_size={cfg.image_size} latent_dim={cfg.latent_dim} dev={dev}",
        flush=True,
    )
    print(
        "[PARAM] weights(full) "
        f"nt={cfg.lambda_nt} recon={cfg.lambda_recon} beta={cfg.beta} "
        f"lc={cfg.lambda_lc} deriv={cfg.lambda_deriv} nt_temp={cfg.nt_temperature} "
        f"softargmax_temp={cfg.softargmax_temp}",
        flush=True,
    )
    print(
        "[PARAM] anneal start={} epochs={} | aug max_df_hz={} ({} bins) "
        "max_dt_frames={} | band=[{:.0f},{:.0f}]Hz fpb={:.2f}Hz".format(
            cfg.recon_anneal_start, cfg.recon_anneal_epochs, cfg.max_df_hz,
            int(round(cfg.max_df_hz / freq_per_bin_hz)), cfg.max_dt_frames,
            band_lo_hz, band_hi_hz, freq_per_bin_hz,
        ),
        flush=True,
    )

    ds = DenoisedPatchRidgeDataset(
        patches, dfdt_true_hz, dfdt_valid, freqs_hz[band_slice], band_slice, padding
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
    )

    model = ShapeVAEv3Hybrid(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params} epochs={args.epochs} batch={args.batch_size}", flush=True)

    out_model = Path(args.output_model_dir)
    out_results = Path(args.output_results_dir)
    out_model.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)
    (out_model / "hyperparams.json").write_text(
        json.dumps(
            {
                "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "patches_npz": str(args.patches_npz),
                "ridge_npz": str(args.ridge_npz),
                "n_patches": int(n),
                "band_rows": [int(row0), int(row1)],
                "freq_per_bin_hz": freq_per_bin_hz,
                "lr": args.lr,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    log_path = out_results / "training_log.csv"
    log_lines = ["epoch,total,nt,recon,kl,lc,deriv,w_recon,w_beta,w_deriv"]
    best_total = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        w = annealed_weights(cfg, epoch)
        model.train()
        sums = {"total": 0.0, "nt": 0.0, "recon": 0.0, "kl": 0.0, "lc": 0.0, "deriv": 0.0}
        nb = 0
        for image, dfdt_true, dfdt_valid, ridge_lo, ridge_hi in loader:
            image = image.to(dev, non_blocking=True)
            dfdt_true = dfdt_true.to(dev, non_blocking=True)
            dfdt_valid = dfdt_valid.to(dev, non_blocking=True)
            # Positive contrastive view: pitch/time shift, clamped in-band.
            x_aug, _, _ = augment_pitch_time_shift(
                image, ridge_lo, ridge_hi, cfg,
                freq_per_bin_hz=freq_per_bin_hz,
                band_lo_hz=band_lo_hz, band_hi_hz=band_hi_hz,
                generator=gen,
            )
            out = model(image, x_aug)
            # Decoded dF/dt: soft-argmax over the BAND rows / real time columns,
            # finite-differenced, compared (masked) to the cached true target.
            ridge_dec = soft_argmax_ridge(
                out["x_recon"][:, :, row0:row1, :t_in], band_freqs_hz, cfg.softargmax_temp
            )
            dfdt_dec = torch.diff(ridge_dec, dim=-1)             # (B, t_in-1)
            m = dfdt_valid.to(dfdt_dec.dtype)
            deriv = ((dfdt_dec - dfdt_true) ** 2 * m).sum() / (m.sum() + 1e-8)
            nt = nt_xent(out["z"], out["z_aug"], cfg.nt_temperature)
            lc = latent_consistency(out["z"], out["z_aug"])
            _ignored, recon, kl = image_vae_loss(
                out["x_recon"], image, out["mu"], out["logvar"], beta=1.0
            )
            total = (
                w["lambda_nt"] * nt
                + w["lambda_recon"] * recon
                + w["beta"] * kl
                + w["lambda_lc"] * lc
                + w["lambda_deriv"] * deriv
            )
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            sums["total"] += float(total.item())
            sums["nt"] += float(nt.item())
            sums["recon"] += float(recon.item())
            sums["kl"] += float(kl.item())
            sums["lc"] += float(lc.item())
            sums["deriv"] += float(deriv.item())
            nb += 1
        nb = max(nb, 1)
        avg = {k: v / nb for k, v in sums.items()}
        log_lines.append(
            f"{epoch},{avg['total']:.5f},{avg['nt']:.5f},{avg['recon']:.5f},"
            f"{avg['kl']:.5f},{avg['lc']:.5f},{avg['deriv']:.5f},"
            f"{w['lambda_recon']:.4f},{w['beta']:.4f},{w['lambda_deriv']:.4f}"
        )
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"  ep{epoch:3d} total={avg['total']:.4f} nt={avg['nt']:.4f} "
                f"recon={avg['recon']:.2f} kl={avg['kl']:.3f} lc={avg['lc']:.4f} "
                f"deriv={avg['deriv']:.4f} | w_recon={w['lambda_recon']:.3f}",
                flush=True,
            )
        if avg["total"] < best_total:
            best_total = avg["total"]
            torch.save(model.state_dict(), out_model / "best.pt")
    torch.save(model.state_dict(), out_model / "last.pt")
    log_path.write_text("\n".join(log_lines))
    print(
        f"[DONE] shape_vae_v3_hybrid {(time.time()-t0)/60:.1f} min "
        f"best_total={best_total:.4f} -> {out_model}",
        flush=True,
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="B+A hybrid shape VAE (rig, GPU).")
    ap.add_argument("--patches-npz", required=True,
                    help="denoised patches.npz (do NOT rebuild; rig path)")
    ap.add_argument("--ridge-npz", required=True,
                    help="per-patch true ridge cache from extract_ridge_targets_v3.py")
    ap.add_argument("--output-model-dir", required=True)
    ap.add_argument("--output-results-dir", required=True)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--lambda-nt", type=float, default=1.0)
    ap.add_argument("--lambda-recon", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--lambda-lc", type=float, default=1.0)
    ap.add_argument("--lambda-deriv", type=float, default=0.1)
    ap.add_argument("--nt-temperature", type=float, default=0.2)
    ap.add_argument("--max-df-hz", type=float, default=15_000.0)
    ap.add_argument("--max-dt-frames", type=int, default=20)
    ap.add_argument("--recon-anneal-start", type=int, default=10)
    ap.add_argument("--recon-anneal-epochs", type=int, default=20)
    ap.add_argument("--softargmax-temp", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> int:
    return train(_build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
