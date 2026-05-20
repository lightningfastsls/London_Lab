"""Port of ../vae-pytorch-pivot/usv_language/models/image_vae.py adapted for
contour-masked patches, image_size=256.

Phase 4 v2 — Train a 32-dim continuous-latent VAE on the 291 contour-masked
USV patches at ``results/masked_patches/5970/``. This is a port of the
"ours" architecture from the prior worktree (see
``../vae-pytorch-pivot/usv_language/models/image_vae.py``) — the same
4-stride-2 channel-doubling encoder + mirrored decoder + sigmoid output +
BCE-reconstruction ELBO that converged successfully on lab/wild VAE
comparison. We do NOT reinvent the architecture; we port it.

Why v2 (not editing v1):
    v1 (``scripts/train_contour_vae.py``) used a smaller architecture (latent
    8, channel ladder 1→16→32→64→128, MSE in z-scored space) and per-patch
    z-score normalization. It produced visibly noisy reconstructions on this
    sparse contour-masked input. v2 fixes both pieces:

      * Architecture — port of image_vae.py (32D latent, 1→32→64→128→256,
        LeakyReLU + BatchNorm, sigmoid + BCE). This is the architecture
        that already worked on 16K spectrogram images for the cross-cohort
        VAE comparison.
      * Preprocessing — refinement-D compatible amplitude stripper:
        per-patch min/max rescale to [0, 1] after log1p. This is the
        normalization documented in the orchestrator handoff (refinement D
        is implemented as min/max [0,1], not z-score, because BCE requires
        a [0,1] target).

Key contract with the corpus:
    - Imports ``USV_FREQ_MIN_HZ`` / ``USV_FREQ_MAX_HZ`` from
      ``src/usv_spectrogram/corpus.py``. Never redeclares 20000 or 120000.
    - The band-crop is computed from the ``freqs_kHz`` array shipped inside
      ``patches.npz`` against those corpus constants.

The architecture's encoder is 4 stride-2 convs, so ``image_size`` must be a
power of 2 ≥ 16. We zero-pad the band-cropped (170 freq × 234 time) input
out to (256, 256) so the original power-of-2 encoder works unchanged.
Padding strategy is symmetric on the frequency axis (43 bins above, 43
below) and right-pad-only on the time axis (22 columns on the right). This
is documented in ``hyperparams.json`` and reversed when rendering
reconstruction PNGs (so visualizations show the real 170×234 region, not
the padded 256×256).

Self-contained: the ImageVAE + ImageVAEConfig classes are inlined below so
this script depends on no file from the prior worktree. The only repo-local
import is ``src/usv_spectrogram/corpus``.

Outputs (under the dirs passed via --output-model-dir / --output-results-dir):
    <model-dir>/best.pt              — state_dict at lowest val_recon
    <model-dir>/last.pt              — state_dict at final epoch
    <model-dir>/hyperparams.json     — full config dump
    <results-dir>/training_log.csv   — one row per epoch
    <results-dir>/latents.parquet    — 291 × (patch_idx, z_0..z_31, manifest joins)
    <results-dir>/reconstructions/   — 20 input/output PNG pairs (val-set, seed=42)

CLI:
    .venv/bin/python scripts/train_contour_vae_v2.py \\
        --patches-npz results/masked_patches/5970/patches.npz \\
        --manifest-parquet results/masked_patches/5970/patches_manifest.parquet \\
        --output-model-dir models/contour_vae_v2_5970 \\
        --output-results-dir results/contour_vae_v2/5970

Defaults (matching the orchestrator handoff):
    --latent-dim 32  --batch-size 32  --lr 2.5e-4
    --max-epochs 500 --patience 50 --seed 42

The script is CUDA-aware: if ``torch.cuda.is_available()`` it uses the GPU,
prints the device name, and runs on the rig. Otherwise it falls back to CPU
(useful for smoke-testing).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend; this script may run on a remote rig

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

# Make src/ importable regardless of cwd. The script must run standalone on a
# remote rig, so we bootstrap the path the same way the v1 script does.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from usv_spectrogram import corpus  # noqa: E402

# ---------------------------------------------------------------------------
# Image size — 256x256 so the ported 4-stride-2 encoder works unchanged.
# ---------------------------------------------------------------------------

IMAGE_SIZE = 256


# ---------------------------------------------------------------------------
# Model — port of image_vae.py (do NOT rename; preserves lineage for readers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageVAEConfig:
    """Configuration for the image VAE. PORTED from
    ``../vae-pytorch-pivot/usv_language/models/image_vae.py``.

    Parameters
    ----------
    image_size:
        Side length of the square input image. Must be a power of 2 >= 16
        because the encoder applies 4 stride-2 convs.
    in_channels:
        Channels per image (1 for grayscale spectrogram patches).
    latent_dim:
        Latent space dimensionality (default 32, matches DS / Goffinet 2021
        and the prior worktree's "ours" run).
    base_channels:
        Channel count of the first encoder conv. Subsequent layers double.
        With base=32 and image_size=256 the encoder bottleneck is
        (256, 16, 16).
    beta:
        KL weight in the ELBO (1.0 = standard VAE).
    """

    image_size: int = IMAGE_SIZE
    in_channels: int = 1
    latent_dim: int = 32
    base_channels: int = 32
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.image_size <= 0 or (self.image_size & (self.image_size - 1)) != 0:
            raise ValueError(
                f"image_size must be a positive power of 2, got {self.image_size}"
            )
        if self.image_size < 16:
            raise ValueError(
                f"image_size must be >= 16 for the 4-stride-2 encoder, got {self.image_size}"
            )
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")

    @property
    def bottleneck_spatial(self) -> int:
        return self.image_size // 16

    @property
    def bottleneck_channels(self) -> int:
        return self.base_channels * 8


class ImageVAE(nn.Module):
    """Continuous-latent VAE on grayscale spectrogram images. PORTED from
    ``../vae-pytorch-pivot/usv_language/models/image_vae.py``.

    ``forward(x)`` returns ``(x_recon, mu, logvar)``. Use ``encode_mean(x)``
    for a deterministic embedding (posterior mean).
    """

    def __init__(self, cfg: ImageVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ImageVAEConfig()
        c = self.cfg.base_channels
        s = self.cfg.bottleneck_spatial
        b = self.cfg.bottleneck_channels

        # ----- Encoder: 4 stride-2 convs, channel doubling. -----
        self.encoder = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # ----- Latent heads (mean + logvar from a single linear layer). -----
        self.fc_latent = nn.Linear(b * s * s, 2 * self.cfg.latent_dim)

        # ----- Decoder: Linear -> 4 stride-2 tconvs + 2 refinement tconvs. -----
        self.fc_decode = nn.Linear(self.cfg.latent_dim, b * s * s)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(b, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            # Refinement: two stride-1 layers (matches the prior worktree).
            nn.ConvTranspose2d(c // 2, c // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c // 2, self.cfg.in_channels, kernel_size=3, stride=1, padding=1),
        )

    # ---------- core API ----------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        z = self.fc_latent(h)
        mu, logvar = z.chunk(2, dim=1)
        # Numerical stability: clamp logvar so exp(logvar) in the KL term
        # cannot overflow to Inf (which produces nan in the KL sum).
        # ±10 → variance ∈ [e^-10, e^10] ≈ [4.5e-5, 2.2e4], well beyond any
        # reasonable latent dim's needed dynamic range.
        logvar = logvar.clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = self.cfg.bottleneck_channels
        s = self.cfg.bottleneck_spatial
        h = self.fc_decode(z)
        h = h.view(-1, b, s, s)
        return torch.sigmoid(self.decoder(h))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        mu, _ = self.encode(x)
        return mu


# ---------------------------------------------------------------------------
# Loss — BCE + KL (ported from image_vae.py)
# ---------------------------------------------------------------------------


def image_vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO = BCE(recon, target) + beta * KL(q(z|x) || N(0, I)).

    Sum over pixels, mean over batch (Kingma & Welling 2014 convention).
    """
    # Defensive: BCE's CUDA kernel asserts x_recon ∈ [0, 1] elementwise.
    # In rare cases the sigmoid output picks up NaN (typically from NaN/Inf
    # leaking back from earlier training instability — though theoretically
    # the sigmoid output should already be bounded). Sanitize to keep
    # training stable, in line with the standard "clamp + nan_to_num" defense.
    x_recon = torch.nan_to_num(x_recon, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    loss = recon + beta * kl
    return loss, recon, kl


# ---------------------------------------------------------------------------
# Preprocessing — band crop + log1p + per-patch min/max + zero pad to 256x256
# ---------------------------------------------------------------------------


def _compute_band_slice(freqs_kHz: np.ndarray) -> Tuple[slice, int, int]:
    """Return slice covering the corpus USV band, inclusive."""
    fmin_kHz = corpus.USV_FREQ_MIN_HZ / 1000.0
    fmax_kHz = corpus.USV_FREQ_MAX_HZ / 1000.0
    mask = (freqs_kHz >= fmin_kHz) & (freqs_kHz <= fmax_kHz)
    if not mask.any():
        raise RuntimeError("USV band slice is empty — check freqs_kHz.")
    start = int(np.argmax(mask))
    end = int(len(mask) - np.argmax(mask[::-1]))  # exclusive
    return slice(start, end), start, end


@dataclass(frozen=True)
class PaddingSpec:
    """Documents how (F_in, T_in) → (image_size, image_size).

    Symmetric pad on the freq axis (centers the USV band vertically inside
    the 256-tall image). Right-pad-only on the time axis (preserves the
    onset alignment of the 234-wide patch at column 0). The reverse op is
    applied when we render reconstruction PNGs.
    """

    f_in: int
    t_in: int
    image_size: int
    pad_f_top: int
    pad_f_bot: int
    pad_t_left: int
    pad_t_right: int

    @classmethod
    def for_shape(cls, f_in: int, t_in: int, image_size: int) -> "PaddingSpec":
        if f_in > image_size or t_in > image_size:
            raise ValueError(
                f"input ({f_in}, {t_in}) larger than image_size {image_size}"
            )
        pad_f = image_size - f_in
        pad_f_top = pad_f // 2
        pad_f_bot = pad_f - pad_f_top
        pad_t = image_size - t_in
        # Right-pad-only on time: preserve onset alignment at column 0.
        pad_t_left = 0
        pad_t_right = pad_t
        return cls(
            f_in=f_in, t_in=t_in, image_size=image_size,
            pad_f_top=pad_f_top, pad_f_bot=pad_f_bot,
            pad_t_left=pad_t_left, pad_t_right=pad_t_right,
        )

    def pad(self, arr: np.ndarray) -> np.ndarray:
        """Pad a (F, T) array to (image_size, image_size) with zeros."""
        return np.pad(
            arr,
            (
                (self.pad_f_top, self.pad_f_bot),
                (self.pad_t_left, self.pad_t_right),
            ),
            mode="constant",
            constant_values=0.0,
        )

    def crop(self, padded: np.ndarray) -> np.ndarray:
        """Reverse op for visualization: padded (image_size, image_size) →
        (F_in, T_in). Works on np arrays only (not tensors)."""
        f0 = self.pad_f_top
        f1 = f0 + self.f_in
        t0 = self.pad_t_left
        t1 = t0 + self.t_in
        return padded[f0:f1, t0:t1]


class MaskedPatchDataset(Dataset):
    """Loads contour-masked patches and preprocesses them in [0, 1].

    Pipeline per refinement D:
        1. Crop to USV band (170 freq bins from corpus constants).
        2. log1p(power).
        3. Per-patch min/max rescale to [0, 1].
        4. Zero-pad to (image_size, image_size).
        5. Add channel dim → (1, image_size, image_size).

    All preprocessing happens per ``__getitem__`` so the dataset's resident
    memory is O(1) regardless of N. The original implementation cached the
    log1p output array in __init__ (~11 GB for the combined 4-cohort run),
    which OOM-kills on hosts with < ~30 GB free. Moving log1p into
    __getitem__ trades a small CPU cost per access (microseconds per patch)
    for a bounded memory footprint. DataLoader workers absorb the cost.
    """

    def __init__(
        self,
        patches: np.ndarray,
        band_slice: slice,
        padding: PaddingSpec,
    ) -> None:
        # Keep ``patches`` as-is (may be a memmap view). Slicing into it
        # per-getitem reads only the relevant rows; band_slice + log1p
        # happen at access time, not here.
        self._patches = patches
        self._band_slice = band_slice
        self._padding = padding

    def __len__(self) -> int:
        return self._patches.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw = self._patches[idx, self._band_slice, :]  # (F_in, T_in) power
        x = np.log1p(raw).astype(np.float32)
        p_min = float(x.min())
        p_max = float(x.max())
        denom = max(p_max - p_min, 1e-6)
        x_n = (x - p_min) / denom  # [0, 1]
        # Pad to (image_size, image_size). Zeros sit at the [0,1] floor and
        # match the masked-out areas in the contour patches.
        x_p = self._padding.pad(x_n).astype(np.float32)
        # Add channel dim.
        return torch.from_numpy(x_p[np.newaxis, :, :].copy())


# ---------------------------------------------------------------------------
# Train / val loop
# ---------------------------------------------------------------------------


def _epoch(
    model: ImageVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    beta: float,
) -> Tuple[float, float, float]:
    """Run one epoch. If optimizer is None, runs in eval (no_grad)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            # Sanity: ensure inputs sit in [0, 1] for BCE.
            if torch.isnan(x).any():
                raise RuntimeError("input contains NaN — preprocessing bug")
            x_recon, mu, logvar = model(x)
            loss, recon, kl = image_vae_loss(x_recon, x, mu, logvar, beta=beta)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            total_recon += float(recon.item())
            total_kl += float(kl.item())
            n_batches += 1
    return (
        total_loss / max(n_batches, 1),
        total_recon / max(n_batches, 1),
        total_kl / max(n_batches, 1),
    )


# ---------------------------------------------------------------------------
# Latent extraction + reconstruction PNGs
# ---------------------------------------------------------------------------


def _encode_all(
    model: ImageVAE, dataset: MaskedPatchDataset, device: torch.device, batch_size: int
) -> np.ndarray:
    """Return (N, latent_dim) mu vectors in dataset order."""
    model.eval()
    out: List[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            mu, _ = model.encode(x)
            out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def _build_latents_df(
    z_all: np.ndarray, manifest: pd.DataFrame, latent_dim: int
) -> pd.DataFrame:
    """Assemble the per-patch latents DataFrame from encoded z and manifest.

    Uses positional alignment (``z_all[i]`` ↔ ``manifest.iloc[i]``). Does NOT
    merge on ``patch_idx``, because in a combined-cohort manifest that column
    is not globally unique (each per-cohort manifest restarts patch_idx at 0)
    and a merge would cross-join those duplicates — silently inflating row
    counts and scrambling the (z ↔ wav_stem) mapping.
    """
    if len(z_all) != len(manifest):
        raise ValueError(
            f"z_all length {len(z_all)} != manifest length {len(manifest)}"
        )
    latent_cols = {
        f"z_{k}": z_all[:, k].astype(np.float32) for k in range(latent_dim)
    }
    out = pd.DataFrame({
        "patch_idx": np.arange(len(manifest), dtype=np.int64),
        **latent_cols,
        "wav_stem": manifest["wav_stem"].reset_index(drop=True).values,
        "call_id": manifest["call_id"].reset_index(drop=True).values,
        "window_idx": manifest["window_idx"].reset_index(drop=True).values,
    })
    if "cohort" in manifest.columns:
        out["cohort"] = manifest["cohort"].reset_index(drop=True).values
    return out


def _save_reconstruction_pngs(
    model: ImageVAE,
    dataset: MaskedPatchDataset,
    val_indices: np.ndarray,
    manifest: pd.DataFrame,
    padding: PaddingSpec,
    out_dir: Path,
    device: torch.device,
    n_examples: int,
    seed: int,
) -> List[Path]:
    """Save N input/recon side-by-side PNGs from the val set.

    Crops the 256×256 padded output back to the (F_in, T_in) real region
    before plotting — otherwise the borders of zeros dominate the figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    n = min(n_examples, len(val_indices))
    chosen = rng.choice(val_indices, size=n, replace=False)
    model.eval()
    saved_paths: List[Path] = []
    with torch.no_grad():
        for ci, ds_idx in enumerate(chosen):
            x = dataset[int(ds_idx)].unsqueeze(0).to(device)  # (1, 1, 256, 256)
            x_recon, _, _ = model(x)
            x_np_full = x.squeeze().cpu().numpy()  # (256, 256)
            x_hat_full = x_recon.squeeze().cpu().numpy()  # (256, 256)
            # Crop the padding so the figure shows the real (F_in, T_in)
            # region — visualizing the padded image would mislead the
            # reader into thinking the recon is mostly zeros (it is, but
            # those are the padded borders, not the model's mistake).
            x_np = padding.crop(x_np_full)
            x_hat_np = padding.crop(x_hat_full)
            row = manifest.iloc[int(ds_idx)]
            patch_idx = int(row["patch_idx"])
            wav_stem = str(row["wav_stem"])
            call_id = int(row["call_id"])
            wav_suffix = wav_stem[-15:] if len(wav_stem) > 15 else wav_stem
            vmin = 0.0
            vmax = 1.0
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(
                x_np, origin="lower", aspect="auto",
                cmap="magma", vmin=vmin, vmax=vmax,
            )
            axes[0].set_title("input (per-patch min/max log power) — band region")
            axes[0].set_xlabel("time bin")
            axes[0].set_ylabel("freq bin (USV band)")
            axes[1].imshow(
                x_hat_np, origin="lower", aspect="auto",
                cmap="magma", vmin=vmin, vmax=vmax,
            )
            axes[1].set_title("reconstruction (sigmoid output, band region)")
            axes[1].set_xlabel("time bin")
            fig.suptitle(
                f"patch_idx={patch_idx}  call_id={call_id}  wav=...{wav_suffix}"
            )
            fig.tight_layout()
            out_path = out_dir / f"recon_{ci:02d}_patch{patch_idx:04d}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            saved_paths.append(out_path)
    return saved_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Port of image_vae.py to contour-masked patches with min/max preprocessing"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patches-npz", type=Path, required=True,
                   help="Path to patches.npz with keys 'patches' (N,F,T) and 'freqs_kHz' (F,)")
    p.add_argument("--manifest-parquet", type=Path, required=True,
                   help="Path to patches_manifest.parquet — one row per patch, "
                        "must contain patch_idx, wav_stem, call_id, window_idx")
    p.add_argument("--output-model-dir", type=Path, required=True,
                   help="Directory for best.pt / last.pt / hyperparams.json")
    p.add_argument("--output-results-dir", type=Path, required=True,
                   help="Directory for training_log.csv / latents.parquet / reconstructions/")
    p.add_argument("--latent-dim", type=int, default=32,
                   help="Latent dimensionality (matches prior 'ours' run)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Mini-batch size (32 is sensible for N=291)")
    p.add_argument("--lr", type=float, default=2.5e-4,
                   help="Adam learning rate (matches prior 'ours' run)")
    p.add_argument("--max-epochs", type=int, default=500,
                   help="Maximum training epochs (early stopping may cut short)")
    p.add_argument("--patience", type=int, default=50,
                   help="Early-stopping patience on val_recon")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for torch / numpy / train-val split")
    p.add_argument("--beta", type=float, default=1.0,
                   help="KL weight (1.0 = standard ELBO; handoff: 'no beta-VAE "
                        "before standard ELBO is working')")
    p.add_argument("--n-recon-pngs", type=int, default=20,
                   help="Number of input/recon PNG pairs to render (val-set)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ----- Determinism -----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Note: cudnn.benchmark is left at default (False); enabling it would
    # auto-tune kernels and trade reproducibility for speed.

    # ----- Device -----
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else None
    print(f"[device] torch={torch.__version__} device={device} "
          f"cuda_available={use_cuda} gpu_name={gpu_name}")

    # ----- Load patches -----
    # mmap_mode='r' so we don't materialize the full 16+ GB into RAM at
    # startup — the downstream band-crop + log1p still extracts a smaller
    # contiguous array, but the source array remains memory-mapped.
    # Requires patches.npz to be uncompressed (ZIP_STORED), which our
    # assemble_combined_patches.py guarantees.
    print(f"[load] patches: {args.patches_npz}")
    data = np.load(args.patches_npz, mmap_mode="r")
    patches = data["patches"]
    freqs_kHz = data["freqs_kHz"]
    print(f"[load] patches shape={patches.shape} dtype={patches.dtype}")
    print(f"[load] freqs_kHz: {freqs_kHz[0]:.3f}..{freqs_kHz[-1]:.3f} kHz, "
          f"{len(freqs_kHz)} bins")
    print(f"[load] patches raw range: min={float(patches.min()):.3f} "
          f"max={float(patches.max()):.3f}")

    # ----- Band crop -----
    band_slice, slice_start, slice_end = _compute_band_slice(freqs_kHz)
    F_in = slice_end - slice_start
    T_in = patches.shape[2]
    print(
        f"[band] USV band [{corpus.USV_FREQ_MIN_HZ/1000:.1f}, "
        f"{corpus.USV_FREQ_MAX_HZ/1000:.1f}] kHz → slice "
        f"[{slice_start}:{slice_end}] = {F_in} freq bins"
    )

    # ----- Padding -----
    padding = PaddingSpec.for_shape(F_in, T_in, IMAGE_SIZE)
    print(
        f"[pad] {F_in}x{T_in} → {IMAGE_SIZE}x{IMAGE_SIZE}: "
        f"freq pad top={padding.pad_f_top} bot={padding.pad_f_bot}; "
        f"time pad left={padding.pad_t_left} right={padding.pad_t_right}"
    )

    # ----- Manifest -----
    manifest = pd.read_parquet(args.manifest_parquet)
    if len(manifest) != patches.shape[0]:
        raise RuntimeError(
            f"manifest length {len(manifest)} != patches.shape[0] {patches.shape[0]}"
        )
    manifest = manifest.reset_index(drop=True)

    # ----- Dataset -----
    dataset = MaskedPatchDataset(patches, band_slice=band_slice, padding=padding)
    print(f"[dataset] N={len(dataset)} input_shape=(1,{IMAGE_SIZE},{IMAGE_SIZE})")

    # Quick post-norm range check on a sample (covers refinement D verification).
    sample = dataset[0].numpy()
    print(f"[norm] post-min/max+pad sample range: min={float(sample.min()):.6f} "
          f"max={float(sample.max()):.6f} mean={float(sample.mean()):.6f}")

    # ----- Train/val split (80/20, seed=42) -----
    n_total = len(dataset)
    indices = np.arange(n_total)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed, shuffle=True,
    )
    print(f"[split] n_train={len(train_idx)} n_val={len(val_idx)}")

    class _Subset(Dataset):
        def __init__(self, base: MaskedPatchDataset, idx: np.ndarray) -> None:
            self.base = base
            self.idx = idx

        def __len__(self) -> int:
            return len(self.idx)

        def __getitem__(self, i: int) -> torch.Tensor:
            return self.base[int(self.idx[i])]

    train_ds = _Subset(dataset, train_idx)
    val_ds = _Subset(dataset, val_idx)
    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=use_pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=use_pin, drop_last=False,
    )

    # ----- Model -----
    vae_cfg = ImageVAEConfig(
        image_size=IMAGE_SIZE,
        in_channels=1,
        latent_dim=int(args.latent_dim),
        base_channels=32,
        beta=float(args.beta),
    )
    model = ImageVAE(vae_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params} latent_dim={vae_cfg.latent_dim} "
          f"bottleneck=({vae_cfg.bottleneck_channels}, "
          f"{vae_cfg.bottleneck_spatial}, {vae_cfg.bottleneck_spatial})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    # ----- Output dirs -----
    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    args.output_results_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = args.output_results_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = args.output_results_dir / "training_log.csv"

    # ----- Hyperparams dump (write BEFORE training so we have a record even
    #       if the run crashes mid-flight) -----
    hyperparams = {
        # Architecture
        "architecture": (
            "ImageVAE (port of ../vae-pytorch-pivot/usv_language/models/image_vae.py): "
            "encoder 4x stride-2 conv (1->32->64->128->256) BN+LeakyReLU; "
            "fc_latent -> (mu, logvar); decoder fc + 4x stride-2 tconv "
            "(256->128->64->32->16) + 2x stride-1 refinement tconv (16->16->1); "
            "sigmoid output. BCE reconstruction + KL ELBO."
        ),
        "image_vae_config": asdict(vae_cfg),
        "image_size": int(IMAGE_SIZE),
        "in_channels": 1,
        "latent_dim": int(args.latent_dim),
        "base_channels": 32,
        "n_params": int(n_params),
        # Training hyperparameters
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "beta": float(args.beta),
        "optimizer": "Adam (weight_decay=0)",
        "loss": (
            "ELBO = BCE(recon, target) sum-over-pixels / batch + "
            "beta * KL sum-over-dim / batch"
        ),
        "early_stopping_metric": "val_recon",
        # Reproducibility
        "seed": int(args.seed),
        "cudnn_deterministic": True,
        "torch_version": str(torch.__version__),
        "device": str(device),
        "cuda_available": bool(use_cuda),
        "gpu_name": gpu_name,
        # Data
        "patches_npz": str(args.patches_npz),
        "manifest_parquet": str(args.manifest_parquet),
        "n_total": int(n_total),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        # Preprocessing
        "preprocessing": (
            "1) band-crop to corpus USV band; 2) log1p; "
            "3) per-patch min/max rescale to [0, 1] (refinement D, "
            "amplitude stripper compatible with BCE); "
            "4) zero-pad to (image_size, image_size); 5) add channel dim."
        ),
        "band_slice": [int(slice_start), int(slice_end)],
        "F_in": int(F_in),
        "T_in": int(T_in),
        "padding": {
            "pad_f_top": padding.pad_f_top,
            "pad_f_bot": padding.pad_f_bot,
            "pad_t_left": padding.pad_t_left,
            "pad_t_right": padding.pad_t_right,
            "strategy": (
                "symmetric on freq axis (centers USV band); "
                "right-pad-only on time axis (preserves onset at column 0)"
            ),
        },
        # Corpus constants (recorded for traceability)
        "corpus_USV_FREQ_MIN_HZ": int(corpus.USV_FREQ_MIN_HZ),
        "corpus_USV_FREQ_MAX_HZ": int(corpus.USV_FREQ_MAX_HZ),
        "corpus_SAMPLE_RATE_HZ": int(corpus.SAMPLE_RATE_HZ),
        "corpus_STFT_N_FFT": int(corpus.STFT_N_FFT),
        "corpus_STFT_HOP": int(corpus.STFT_HOP),
    }
    with open(args.output_model_dir / "hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    # ----- Training loop -----
    print(f"[train] max_epochs={args.max_epochs} patience={args.patience} "
          f"beta={args.beta}")
    best_val_recon = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    log_rows: List[dict] = []
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        tr_total, tr_recon, tr_kl = _epoch(model, train_loader, optimizer, device, args.beta)
        va_total, va_recon, va_kl = _epoch(model, val_loader, None, device, args.beta)
        log_rows.append({
            "epoch": epoch,
            "train_total": tr_total,
            "train_recon": tr_recon,
            "train_kl": tr_kl,
            "val_total": va_total,
            "val_recon": va_recon,
            "val_kl": va_kl,
            "lr": args.lr,
        })
        if va_recon < best_val_recon:
            best_val_recon = va_recon
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(model.state_dict(), args.output_model_dir / "best.pt")
        else:
            epochs_since_best += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.max_epochs:
            print(
                f"  epoch {epoch:3d}  train_recon={tr_recon:9.3f} kl={tr_kl:8.3f}  "
                f"val_recon={va_recon:9.3f} kl={va_kl:8.3f}  "
                f"best_val={best_val_recon:9.3f}@ep{best_epoch}  "
                f"no_improve={epochs_since_best}"
            )
        if epochs_since_best >= args.patience:
            print(f"[early-stop] no improvement for {args.patience} epochs at epoch {epoch}")
            break

    wall = time.time() - t0
    final_epoch = log_rows[-1]["epoch"]
    print(f"[train] done. final_epoch={final_epoch} best_epoch={best_epoch} "
          f"best_val_recon={best_val_recon:.4f}  wall={wall:.1f}s")

    # ----- Save last.pt and training_log.csv -----
    torch.save(model.state_dict(), args.output_model_dir / "last.pt")
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_csv_path, index=False)

    # ----- Load best.pt for downstream encoding + QA -----
    model.load_state_dict(
        torch.load(args.output_model_dir / "best.pt", map_location=device)
    )
    model.eval()

    # ----- Encode all patches → latents.parquet -----
    print("[encode] computing latents for all patches...")
    z_all = _encode_all(model, dataset, device, batch_size=max(args.batch_size, 32))
    assert z_all.shape == (len(dataset), args.latent_dim), (
        f"z_all shape {z_all.shape} != ({len(dataset)}, {args.latent_dim})"
    )
    latents_df = _build_latents_df(z_all, manifest, args.latent_dim)
    latents_path = args.output_results_dir / "latents.parquet"
    latents_df.to_parquet(latents_path, index=False)
    print(f"[encode] wrote {latents_path} shape={latents_df.shape}")

    # ----- Reconstruction PNGs -----
    print(f"[qa] writing {args.n_recon_pngs} reconstruction PNGs...")
    val_idx_arr = np.array(val_idx)
    saved = _save_reconstruction_pngs(
        model=model,
        dataset=dataset,
        val_indices=val_idx_arr,
        manifest=manifest,
        padding=padding,
        out_dir=recon_dir,
        device=device,
        n_examples=args.n_recon_pngs,
        seed=args.seed,
    )
    print(f"[qa] wrote {len(saved)} PNGs to {recon_dir}")

    # ----- Final summary -----
    print()
    print("=" * 72)
    print("PHASE 4 V2 TRAINING SUMMARY")
    print("=" * 72)
    print(f"  device                  : {device} ({gpu_name or 'CPU'})")
    print(f"  wall clock              : {wall:.1f} s")
    print(f"  final epoch             : {final_epoch}")
    print(f"  best epoch              : {best_epoch}")
    print(f"  best val_recon          : {best_val_recon:.4f}")
    print(f"  final train_recon       : {log_rows[-1]['train_recon']:.4f}")
    print(f"  final val_recon         : {log_rows[-1]['val_recon']:.4f}")
    print(f"  final train_kl          : {log_rows[-1]['train_kl']:.4f}")
    print(f"  final val_kl            : {log_rows[-1]['val_kl']:.4f}")
    print(f"  outputs:")
    print(f"    {args.output_model_dir / 'best.pt'}")
    print(f"    {args.output_model_dir / 'last.pt'}")
    print(f"    {args.output_model_dir / 'hyperparams.json'}")
    print(f"    {log_csv_path}")
    print(f"    {latents_path}")
    print(f"    {recon_dir}/ ({len(saved)} PNGs)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
