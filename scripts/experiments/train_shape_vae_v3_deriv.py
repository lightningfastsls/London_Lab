"""Pathway A — derivative-loss contour-VAE (the literal dF/dt hypothesis).

Spec: docs/handoffs/2026-05-27_shape-vae-A-derivative-loss.md
Plan: PLAN_geometric_shape_clustering_vae.md §3 Option A.

THE IDEA
--------
Add a *ridge-derivative* term to the production ImageVAE loss so the model is
penalised for getting the **slope dF/dt** (and optionally curvature d²F/dt²) of
the call's frequency contour wrong. dF/dt is invariant to absolute pitch (a
constant shift differentiates away) and to time position, so in principle it
pushes the latent toward *shape* and away from pitch / duration.

WHY THE PRIOR TESTS DON'T COUNT (read before changing anything)
---------------------------------------------------------------
- **M8** put the derivative term on a 1-D ridge that was *already registered*
  (pitch removed by preprocessing) → the term had nothing left to do.
- **M10** used a raw **pixel gradient** ``MSE(∂I/∂t)+MSE(∂I/∂f)`` as "the
  derivative". A pixel gradient is **not** dF/dt and is **not** pitch-invariant:
  shift a call up 5 kHz and every lit pixel moves rows, so the term is large for
  an identical shape. → shape η² 0.009.

THE FAITHFUL TEST (this module)
-------------------------------
Keep the **2-D denoised image** as input (so jumps / sub-harmonics survive) and
compute the derivative on a **kHz-aware soft-argmax ridge** extracted from the
decoded recon, matched to the **true** ridge dF/dt. Because the ridge is mapped
to real kHz, a pure pitch shift moves the expected frequency but leaves dF/dt
unchanged — the property M10's pixel gradient lacked.

DIFFERENTIABILITY CHOICE (user decision, 2026-05-27): **soft-argmax proxy**.
``track_ridge`` (Viterbi) is non-differentiable, so dF/dt is taken on the
differentiable *expected-frequency-per-column* of the decoded magnitude
(softmax over the frequency axis, temperature ``tau``). Gradient therefore flows
end-to-end into the encoder/decoder — the mechanism that can re-weight the
latent toward the low-variance ridge curve.
  KNOWN RISK: expected-frequency collapses to a weighted mean between bands on
  frequency jumps / sub-harmonic stacks, smoothing exactly the multi-component
  structure the 2-D input was kept to preserve. Mitigation: keep ``tau`` low
  (sharper softmax) and report jump-capture (eval gate #4) explicitly.

HONEST PRIOR — the bar this must clear
--------------------------------------
The 2026-05-26 denoised retrain (no derivative term) was a dead end: shape η²
0.081 vs 0.75 registration ceiling; the latent still sorted by pitch/duration.
This derivative term is the one untested re-weighting mechanism — but it may not
be enough. Kill if shape η² < 0.12 after tuning λ_d / β. See the handoff's kill
criteria; treat this as a hypothesis test, not a sure win.

Corpus constants (sample rate / USV band / STFT) are imported from
``usv_spectrogram.corpus`` via the frozen baseline — never redeclared here.

Do NOT overwrite ``scripts/train_contour_vae_v2.py`` (the frozen baseline whose
``ImageVAE`` this module reuses).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path bootstrap — make the frozen baseline (scripts/) and src/ importable
# whether run under pytest or as a CLI on the rig.
# ---------------------------------------------------------------------------
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]          # repo/scripts
_REPO_ROOT = _SCRIPTS_ROOT.parent                            # repo
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_SCRIPTS_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Frozen baseline — reused, never modified.
from train_contour_vae_v2 import (  # noqa: E402
    ImageVAE,
    ImageVAEConfig,
    PaddingSpec,
    MaskedPatchDataset,
    _compute_band_slice,
    image_vae_loss,
)

# Allowed reconstruction objectives for the config validator.
_ALLOWED_RECON = ("bce", "mse")


# ===========================================================================
# Config
# ===========================================================================


@dataclass(frozen=True)
class ShapeVAEDerivConfig:
    """Configuration for the derivative-loss shape VAE.

    Extends the frozen ImageVAE knobs with the derivative-term weights. ``beta``
    defaults LOW (0.1) — the denoised dead-end used 1.0 and KL over-smoothing is
    a prime suspect for the latent collapsing onto pitch/duration.

    Parameters
    ----------
    image_size, latent_dim, base_channels:
        Forwarded to ``ImageVAEConfig`` (the encoder/decoder shape).
    beta:
        KL weight. LOW by design (0.1, not the dead-end's 1.0).
    lambda_d:
        Weight on the first-derivative term ``MSE(dF/dt_decoded, dF/dt_true)``.
    lambda_c:
        Weight on the optional curvature term ``MSE(d²F/dt²)``. 0 disables it.
    tau:
        Soft-argmax temperature (lower = sharper ≈ hard argmax, but weaker
        gradient). Keep low to limit jump smoothing.
    recon:
        Reconstruction objective: "bce" (matches the frozen baseline) or "mse".
    mask_recon:
        If True, weight the recon loss by a per-pixel mask (the prefilter call
        region) so the objective is not dominated by background. Flag-gated;
        report with/without per the handoff.
    """

    image_size: int = 256
    latent_dim: int = 32
    base_channels: int = 32
    beta: float = 0.1
    lambda_d: float = 1.0
    lambda_c: float = 0.0
    tau: float = 0.05
    recon: str = "bce"
    mask_recon: bool = False

    def __post_init__(self) -> None:
        if self.recon not in _ALLOWED_RECON:
            raise ValueError(
                f"recon must be one of {_ALLOWED_RECON}, got {self.recon!r}"
            )
        if self.lambda_d < 0:
            raise ValueError(f"lambda_d must be >= 0, got {self.lambda_d}")
        if self.lambda_c < 0:
            raise ValueError(f"lambda_c must be >= 0, got {self.lambda_c}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")

    def image_vae_config(self) -> ImageVAEConfig:
        """Build the frozen-baseline config from these knobs."""
        return ImageVAEConfig(
            image_size=self.image_size,
            in_channels=1,
            latent_dim=self.latent_dim,
            base_channels=self.base_channels,
            beta=self.beta,
        )


# ===========================================================================
# Soft-argmax ridge + derivatives (the differentiable, pitch-invariant core)
# ===========================================================================


def soft_argmax_frequency(
    img: torch.Tensor,
    freq_khz: torch.Tensor,
    tau: float = 0.05,
) -> torch.Tensor:
    """Differentiable expected frequency per time column (the soft ridge).

    For each time column the decoded magnitude over frequency rows is treated as
    a distribution ``softmax(img / tau)``; the expected frequency is the
    softmax-weighted sum of the per-row kHz values. This is the differentiable
    stand-in for ``argmax`` over the frequency axis.

    Because the weights multiply the *kHz axis* (not pixel rows), a pure pitch
    shift moves the output by the corresponding kHz but leaves its time
    derivative unchanged — the pitch-invariance M10's pixel gradient lacked.

    Parameters
    ----------
    img:
        ``(B, 1, H, W)`` or ``(B, H, W)`` decoded magnitude (rows = frequency).
    freq_khz:
        ``(H,)`` frequency in kHz of each row.
    tau:
        Softmax temperature. Lower → sharper (closer to hard argmax).

    Returns
    -------
    Tensor ``(B, W)`` — expected frequency (kHz) per time column.
    """
    if img.dim() == 4:
        img = img.squeeze(1)  # (B, 1, H, W) -> (B, H, W); channel is size 1
    if img.dim() != 3:
        raise ValueError(f"img must be (B,1,H,W) or (B,H,W), got {tuple(img.shape)}")
    # softmax over the frequency axis (rows = dim 1)
    weights = torch.softmax(img / tau, dim=1)  # (B, H, W)
    f = freq_khz.to(dtype=img.dtype, device=img.device).view(1, -1, 1)  # (1, H, 1)
    return (weights * f).sum(dim=1)  # (B, W)


def ridge_first_derivative(track: torch.Tensor) -> torch.Tensor:
    """First finite difference along time: ``(B, W) -> (B, W-1)``."""
    return track[:, 1:] - track[:, :-1]


def ridge_second_derivative(track: torch.Tensor) -> torch.Tensor:
    """Second finite difference along time: ``(B, W) -> (B, W-2)``."""
    return track[:, 2:] - 2.0 * track[:, 1:-1] + track[:, :-2]


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean squared error, optionally restricted to ``mask`` columns.

    With a mask, columns where ``mask == 0`` contribute nothing — both to the
    value and (since they are multiplied by 0) to the gradient. The denominator
    is the number of valid entries, clamped to >= 1 to avoid div-by-zero.
    """
    diff2 = (pred - target) ** 2
    if mask is None:
        return diff2.mean()
    mask = mask.to(dtype=diff2.dtype, device=diff2.device)
    denom = mask.sum().clamp_min(1.0)
    return (diff2 * mask).sum() / denom


def derivative_loss(
    img: torch.Tensor,
    freq_khz: torch.Tensor,
    dFdt_true: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    tau: float = 0.05,
    lambda_d: float = 1.0,
    lambda_c: float = 0.0,
    d2_true: torch.Tensor | None = None,
    valid_mask2: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Ridge-derivative penalty on the decoded image.

    Returns a dict with raw (unweighted) ``deriv`` and ``curv`` MSE components
    and the weighted ``total = lambda_d * deriv + lambda_c * curv``.

    The curvature term is computed only when ``lambda_c > 0`` and ``d2_true`` is
    provided; otherwise ``curv`` is exactly 0.
    """
    track = soft_argmax_frequency(img, freq_khz, tau=tau)  # (B, W)
    dFdt = ridge_first_derivative(track)                   # (B, W-1)
    dFdt_true = dFdt_true.to(dtype=dFdt.dtype, device=dFdt.device)
    deriv = _masked_mse(dFdt, dFdt_true, valid_mask)

    if lambda_c > 0 and d2_true is not None:
        d2 = ridge_second_derivative(track)  # (B, W-2)
        d2_true = d2_true.to(dtype=d2.dtype, device=d2.device)
        curv = _masked_mse(d2, d2_true, valid_mask2)
    else:
        curv = torch.zeros((), dtype=deriv.dtype, device=deriv.device)

    total = lambda_d * deriv + lambda_c * curv
    return {"deriv": deriv, "curv": curv, "total": total}


def total_shape_vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    freq_khz: torch.Tensor,
    dFdt_true: torch.Tensor,
    valid_mask: torch.Tensor | None,
    cfg: ShapeVAEDerivConfig,
    recon_weight_mask: torch.Tensor | None = None,
    d2_true: torch.Tensor | None = None,
    valid_mask2: torch.Tensor | None = None,
    deriv_img: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Full objective: ``recon + beta*KL + lambda_d*deriv + lambda_c*curv``.

    The (recon, kl) terms are computed identically to the frozen baseline
    ``image_vae_loss`` so that with ``lambda_d == lambda_c == 0`` and
    ``recon == "bce"`` the objective reduces *exactly* to the baseline ELBO.
    This keeps the derivative term a pure addition — any clustering change is
    attributable to it alone.

    The **reconstruction** term runs on ``x_recon`` vs ``x`` (in training these
    are the FULL padded 256² images, so the recon objective is byte-identical
    to the frozen baseline → clean attribution). The **derivative** term runs on
    ``deriv_img`` — the band-cropped recon (so the ridge maps to real kHz);
    ``freq_khz`` must match ``deriv_img``'s frequency-axis length. When
    ``deriv_img is None`` the derivative runs on ``x_recon`` (used by the unit
    tests, where ``x_recon`` and ``freq_khz`` already share a frequency axis).
    """
    batch = x.shape[0]
    # KL — identical to image_vae_loss.
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / batch

    # Reconstruction — same sanitisation as the baseline (BCE asserts [0,1]).
    xr = torch.nan_to_num(x_recon, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if cfg.recon == "bce":
        per_px = F.binary_cross_entropy(xr, x, reduction="none")
    elif cfg.recon == "mse":
        per_px = (xr - x) ** 2
    else:  # defensive — config validates, but never trust a silent fallthrough
        raise ValueError(f"unsupported recon objective {cfg.recon!r}")
    if cfg.mask_recon and recon_weight_mask is not None:
        per_px = per_px * recon_weight_mask
    recon = per_px.sum() / batch

    img_for_deriv = x_recon if deriv_img is None else deriv_img
    dloss = derivative_loss(
        img_for_deriv,
        freq_khz,
        dFdt_true,
        valid_mask=valid_mask,
        tau=cfg.tau,
        lambda_d=cfg.lambda_d,
        lambda_c=cfg.lambda_c,
        d2_true=d2_true,
        valid_mask2=valid_mask2,
    )

    total = recon + cfg.beta * kl + dloss["total"]
    return {
        "total": total,
        "recon": recon,
        "kl": kl,
        "deriv": dloss["deriv"],
        "curv": dloss["curv"],
    }


# ===========================================================================
# Ridge target cache (the precomputed dF/dt_true the loss matches against)
# ===========================================================================


def load_ridge_cache(npz_path: Any) -> dict[str, np.ndarray]:
    """Load the per-patch true-ridge derivative cache produced by Track 0.

    Required arrays: ``dFdt_true`` (N, W-1) and ``valid_mask`` (N, W-1).
    Optional arrays: ``d2_true`` (N, W-2) and ``valid_mask2`` (N, W-2).

    All arrays must share the same first dimension N (one row per patch);
    a mismatch raises ``ValueError`` (it would silently misalign targets).
    """
    with np.load(str(npz_path)) as z:
        keys = list(z.keys())
        if "dFdt_true" not in keys or "valid_mask" not in keys:
            raise ValueError(
                "ridge cache must contain 'dFdt_true' and 'valid_mask'; "
                f"got keys {keys}"
            )
        out = {k: z[k] for k in keys}
    first_dims = {k: int(v.shape[0]) for k, v in out.items()}
    if len(set(first_dims.values())) != 1:
        raise ValueError(
            f"ridge cache arrays are misaligned on the first dim: {first_dims}"
        )
    return out


# ===========================================================================
# Band-region extraction from the padded 256x256 recon (differentiable)
# ===========================================================================


def extract_band_region(img256: torch.Tensor, padding: PaddingSpec) -> torch.Tensor:
    """Crop the padded ``(B, 1, S, S)`` recon back to the real ``(B, 1, F_in, T_in)``
    USV-band region using the same geometry the baseline pads with.

    Torch slicing preserves the autograd graph, so the derivative term flows
    gradients only through the real call region (not the zero-padding).
    """
    f0 = padding.pad_f_top
    f1 = f0 + padding.f_in
    t0 = padding.pad_t_left
    t1 = t0 + padding.t_in
    return img256[..., f0:f1, t0:t1]


def band_freq_khz(freqs_khz_full: np.ndarray) -> torch.Tensor:
    """The USV-band frequency axis (kHz per row) used by the soft-argmax ridge."""
    band_slice, _, _ = _compute_band_slice(freqs_khz_full)
    return torch.from_numpy(freqs_khz_full[band_slice].astype(np.float32))


# ===========================================================================
# Dataset — base image patches + aligned ridge-derivative targets
# ===========================================================================


class RidgePatchDataset(torch.utils.data.Dataset):
    """Wraps the frozen ``MaskedPatchDataset`` and attaches the per-patch
    derivative targets, returning a dict per item.

    The base dataset performs the exact baseline preprocessing (band crop →
    log1p → per-patch min/max → zero-pad → channel). The ridge targets come
    from the Track-0 cache, aligned by patch index.
    """

    def __init__(
        self,
        base: MaskedPatchDataset,
        ridge_cache: dict[str, np.ndarray],
    ) -> None:
        if len(base) != ridge_cache["dFdt_true"].shape[0]:
            raise ValueError(
                f"base dataset ({len(base)}) and ridge cache "
                f"({ridge_cache['dFdt_true'].shape[0]}) length mismatch"
            )
        self._base = base
        self._dFdt = ridge_cache["dFdt_true"]
        self._vmask = ridge_cache["valid_mask"]
        self._d2 = ridge_cache.get("d2_true")
        self._vmask2 = ridge_cache.get("valid_mask2")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "x": self._base[idx],  # (1, S, S)
            "dFdt_true": torch.from_numpy(self._dFdt[idx].astype(np.float32)),
            "valid_mask": torch.from_numpy(self._vmask[idx].astype(np.float32)),
        }
        if self._d2 is not None and self._vmask2 is not None:
            item["d2_true"] = torch.from_numpy(self._d2[idx].astype(np.float32))
            item["valid_mask2"] = torch.from_numpy(self._vmask2[idx].astype(np.float32))
        return item


# ===========================================================================
# Training loop (runs on the rig — heavy I/O / GPU; not unit-tested)
# ===========================================================================


def _run_epoch(
    model: ImageVAE,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: ShapeVAEDerivConfig,
    freq_khz: torch.Tensor,
    padding: PaddingSpec,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "deriv": 0.0, "curv": 0.0}
    n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    freq_khz = freq_khz.to(device)
    with ctx:
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            if torch.isnan(x).any():
                raise RuntimeError("input contains NaN — preprocessing bug")
            dFdt_true = batch["dFdt_true"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            d2_true = batch.get("d2_true")
            valid_mask2 = batch.get("valid_mask2")
            if d2_true is not None:
                d2_true = d2_true.to(device, non_blocking=True)
                valid_mask2 = valid_mask2.to(device, non_blocking=True)

            x_recon, mu, logvar = model(x)
            # Recon (BCE) on the FULL padded image — byte-identical to the frozen
            # baseline objective, so lambda_d=0 reduces to the dead-end exactly
            # and any shape-eta2 gain is attributable to the derivative term alone.
            # The derivative term runs on the band crop (ridge mapped to real kHz).
            recon_band = extract_band_region(x_recon, padding)
            losses = total_shape_vae_loss(
                x_recon,
                x,
                mu,
                logvar,
                freq_khz,
                dFdt_true,
                valid_mask,
                cfg,
                d2_true=d2_true,
                valid_mask2=valid_mask2,
                deriv_img=recon_band,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
            for k in totals:
                totals[k] += float(losses[k].item())
            n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Pathway-A derivative-loss shape VAE.")
    ap.add_argument("--patches", required=True, help="patches.npz (denoised corpus)")
    ap.add_argument("--ridge-cache", required=True, help="Track-0 dFdt_true npz")
    ap.add_argument("--out", required=True, help="output dir for model + latents")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.1, help="KL weight (LOW by design)")
    ap.add_argument("--lambda-d", type=float, default=1.0, help="dF/dt term weight")
    ap.add_argument("--lambda-c", type=float, default=0.0, help="curvature term weight")
    ap.add_argument("--tau", type=float, default=0.05, help="soft-argmax temperature")
    ap.add_argument("--recon", choices=_ALLOWED_RECON, default="bce")
    ap.add_argument("--mask-recon", action="store_true", help="weight recon by call region")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = ShapeVAEDerivConfig(
        latent_dim=args.latent_dim,
        beta=args.beta,
        lambda_d=args.lambda_d,
        lambda_c=args.lambda_c,
        tau=args.tau,
        recon=args.recon,
        mask_recon=args.mask_recon,
    )
    # PRINT DISCIPLINE (lab convention): params, thresholds, sort keys, counts.
    print(
        "[params] " + json.dumps({
            "patches": args.patches,
            "ridge_cache": args.ridge_cache,
            "latent_dim": cfg.latent_dim,
            "beta": cfg.beta,
            "lambda_d": cfg.lambda_d,
            "lambda_c": cfg.lambda_c,
            "tau": cfg.tau,
            "recon": cfg.recon,
            "mask_recon": cfg.mask_recon,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "device": args.device,
        }),
        flush=True,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.patches, mmap_mode="r")
    patches = z["patches"]
    freqs_full = np.asarray(z["freqs_kHz"])
    band_slice, f_start, f_end = _compute_band_slice(freqs_full)
    f_in = f_end - f_start
    t_in = int(patches.shape[2])
    padding = PaddingSpec.for_shape(f_in, t_in, cfg.image_size)
    freq_khz = band_freq_khz(freqs_full)
    print(f"[data] patches={patches.shape} band={f_in}x{t_in} -> {cfg.image_size}^2 "
          f"freq_khz[{float(freq_khz[0]):.1f},{float(freq_khz[-1]):.1f}]", flush=True)

    ridge_cache = load_ridge_cache(args.ridge_cache)
    base_ds = MaskedPatchDataset(patches, band_slice, padding)
    full_ds = RidgePatchDataset(base_ds, ridge_cache)

    n_total = len(full_ds)
    n_val = max(1, int(round(args.val_frac * n_total)))
    perm = np.random.permutation(n_total)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = torch.utils.data.Subset(full_ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(full_ds, val_idx.tolist())
    print(f"[split] train={len(train_ds)} val={len(val_ds)} (val_frac={args.val_frac})",
          flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = ImageVAE(cfg.image_vae_config()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params} latent_dim={cfg.latent_dim}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist the split so eval_shape_vae_v3 can score on the HELD-OUT rows
    # (not training data). Reconstructing from --seed alone is error-prone.
    np.save(out_dir / "val_idx.npy", val_idx)
    np.save(out_dir / "train_idx.npy", train_idx)
    print(f"[split] saved val_idx ({len(val_idx)}) + train_idx ({len(train_idx)}) "
          f"-> {out_dir}", flush=True)
    best_val = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        tr = _run_epoch(model, train_loader, optimizer, device, cfg, freq_khz, padding)
        va = _run_epoch(model, val_loader, None, device, cfg, freq_khz, padding)
        print(
            f"[ep {ep:03d}] "
            f"train total={tr['total']:.3f} recon={tr['recon']:.3f} kl={tr['kl']:.3f} "
            f"deriv={tr['deriv']:.4f} curv={tr['curv']:.4f} | "
            f"val total={va['total']:.3f} deriv={va['deriv']:.4f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
        if va["total"] < best_val:
            best_val = va["total"]
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg.__dict__,
                 "epoch": ep, "val_total": best_val},
                out_dir / "best.pt",
            )

    # Encode posterior-mean latents for the full corpus (eval input).
    model.eval()
    mus = []
    enc_loader = torch.utils.data.DataLoader(
        full_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
    )
    with torch.no_grad():
        for batch in enc_loader:
            mus.append(model.encode_mean(batch["x"].to(device)).cpu().numpy())
    latents = np.concatenate(mus, axis=0)
    np.save(out_dir / "latents.npy", latents)
    print(f"[done] best_val={best_val:.3f} latents={latents.shape} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
