"""Track 0 (step 2) — extract per-patch TRUE ridge dF/dt targets for Pathway A.

Spec: docs/handoffs/2026-05-27_shape-vae-A-derivative-loss.md  (Prerequisites §2)

The denoised patch corpus already exists on the rig (do NOT rebuild). What is
MISSING — and what this script builds once — is the per-patch *true ridge* dF/dt
array used as the derivative loss TARGET in train_shape_vae_v3_deriv.py.

For each patch we:
  1. crop to the USV band (corpus constants, via the frozen baseline helper),
  2. run the SAME Viterbi tracker that feeds registration
     (``ridge_tracker.track_ridge``) → true F(t) per time column (kHz),
  3. finite-difference F(t) → dF/dt_true (length T-1), and the second
     difference → d²F/dt²_true (length T-2),
  4. record a validity mask marking columns where the true ridge is active
     (a derivative bin is valid only when both endpoints are finite).

``track_ridge`` is ~50× slower on denoised (rich) patches than on masked ridges
(≈24 min/corpus, per project memory). Compute ONCE, cache, never recompute in
the train loop. Output is a parallel .npz aligned by patch index:

    dFdt_true   (N, T-1) float32   — derivative target (kHz / frame)
    valid_mask  (N, T-1) bool      — True where both endpoints active
    d2_true     (N, T-2) float32   — curvature target
    valid_mask2 (N, T-2) bool      — True where the 3-point window is active

Corpus constants are imported via the frozen baseline + ridge_tracker; never
redeclared here. This script is designed to run on the rig.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SCRIPTS_ROOT.parent
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_SCRIPTS_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_contour_vae_v2 import _compute_band_slice  # noqa: E402
from usv_spectrogram.features.ridge_tracker import track_ridge, RidgeConfig  # noqa: E402

# Active-column floor — mirrors rig_R2_shape_alphabet.register_one so the target
# ridge is defined exactly as the registration baseline defines it.
MIN_ACTIVE_COLS = 6


def true_ridge_khz(crop: np.ndarray, freqs_khz: np.ndarray) -> np.ndarray:
    """Per-column true frequency (kHz), NaN where the column is inactive.

    Uses the same RidgeConfig as the registration baseline (relative silence
    threshold at 2% of the crop max) so the target matches the production ridge.
    """
    thr = max(1e-9, 0.02 * float(crop.max()))
    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10, silence_threshold=thr)
    fm, _ = track_ridge(crop, freqs_khz.astype(float), cfg)
    return fm  # (T,), NaN where inactive


def derivative_targets(fm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference the true ridge into (dFdt, vmask, d2, vmask2).

    A first-difference bin is valid only when BOTH endpoint columns are active
    (finite); inactive columns contribute 0 derivative and a False mask so the
    loss ignores them. NaNs are zeroed AFTER the validity mask is taken.
    """
    active = np.isfinite(fm)
    f = np.nan_to_num(fm, nan=0.0).astype(np.float32)

    dFdt = f[1:] - f[:-1]
    vmask = active[1:] & active[:-1]
    dFdt = np.where(vmask, dFdt, 0.0).astype(np.float32)

    d2 = (f[2:] - 2.0 * f[1:-1] + f[:-2]).astype(np.float32)
    vmask2 = active[2:] & active[1:-1] & active[:-2]
    d2 = np.where(vmask2, d2, 0.0).astype(np.float32)

    return dFdt, vmask, d2, vmask2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract true-ridge dF/dt targets (Track 0).")
    ap.add_argument("--patches", required=True, help="patches.npz (denoised corpus)")
    ap.add_argument("--out", required=True, help="output ridge-cache .npz")
    ap.add_argument("--log-every", type=int, default=5000)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    z = np.load(args.patches, mmap_mode="r")
    patches = z["patches"]
    freqs_full = np.asarray(z["freqs_kHz"])
    band_slice, f0, f1 = _compute_band_slice(freqs_full)
    freqs_band = freqs_full[band_slice]
    n, t_in = patches.shape[0], patches.shape[2]
    print(f"[params] patches={patches.shape} band=[{f0}:{f1}] ({f1-f0} bins) "
          f"freq_khz[{freqs_band[0]:.1f},{freqs_band[-1]:.1f}] T={t_in} "
          f"MIN_ACTIVE_COLS={MIN_ACTIVE_COLS}", flush=True)

    dFdt = np.zeros((n, t_in - 1), np.float32)
    vmask = np.zeros((n, t_in - 1), bool)
    d2 = np.zeros((n, t_in - 2), np.float32)
    vmask2 = np.zeros((n, t_in - 2), bool)

    t0 = time.time()
    n_active = 0
    for i in range(n):
        crop = np.asarray(patches[i, band_slice, :])
        fm = true_ridge_khz(crop, freqs_band)
        if np.isfinite(fm).sum() >= MIN_ACTIVE_COLS:
            n_active += 1
        dFdt[i], vmask[i], d2[i], vmask2[i] = derivative_targets(fm)
        if args.log_every and i and i % args.log_every == 0:
            rate = i / (time.time() - t0)
            print(f"  {i}/{n} active={n_active} ({rate:.0f}/s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, dFdt_true=dFdt, valid_mask=vmask, d2_true=d2, valid_mask2=vmask2)
    frac = n_active / max(n, 1)
    print(f"[done] N={n} active_ridges={n_active} ({frac:.1%}) "
          f"dFdt={dFdt.shape} -> {out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
