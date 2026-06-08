"""Minimal KILL-GATE eval for the B+A hybrid (run1) — shape η² only.

Spec: `PLAN_geometric_shape_clustering_vae.md` §5 (kill criteria) — shape η² < 0.12
after tuning ⇒ KILL the pathway. This script answers ONE question fast:

  Does run1's latent partition the registered-ridge shape axis better than the
  0.12 production-VAE baseline?

It is NOT the full shared `eval_shape_vae_v3.py` (which scores all gates +
navigable-map figure + UMAP + HDBSCAN — deferred to coordinated authoring
with the parallel Pathway-A chat). This script intentionally costs ~10 min so
we don't burn rig time on runs 2-4 of the sweep before knowing if run1
clears the kill gate.

Method (matches the rig_M10 / rig_R2 baselines for direct comparability):
  1. Encode all 5970 denoised patches via best.pt → posterior means (12440, 32).
  2. KMeans(20) on the latents (random_state=42, n_init=10).
  3. Per patch: band-crop, track_ridge (Viterbi) → fm_hz; M10-style register_one
     (center = subtract mean, resample active span to 50 pts → shape vector).
  4. eta2(shapes, labels) = the kill-gate metric. Also score pitch and duration
     η² for diagnostic context.
  5. Print PARAMS / THRESHOLDS / ROW COUNTS (lab convention) and VERDICT.

Run on rig:
  cd /data/mickey_london_lab && PYTHONPATH=src .venv/bin/python \
    scripts/experiments/eval_shape_kill_gate_v3.py \
    --patches-npz /data/shachar/contour_vae/results/denoised_patches/5970/patches.npz \
    --model-pt   /data/shachar/contour_vae/models/shape_vae_v3_hybrid/run1/best.pt \
    --out-json   /data/shachar/contour_vae/results/shape_vae_v3_hybrid/run1/killgate.json \
    --device cuda:3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from usv_spectrogram import corpus  # noqa: E402
from usv_spectrogram.features.ridge_tracker import RidgeConfig, track_ridge  # noqa: E402
from scripts.train_contour_vae_v2 import PaddingSpec, _compute_band_slice  # noqa: E402
from scripts.experiments.train_shape_vae_v3_hybrid import (  # noqa: E402
    ShapeVAEv3Config, ShapeVAEv3Hybrid,
)

N_RESAMPLE = 50
KILL_THRESHOLD = 0.12  # production-VAE baseline; below this ⇒ KILL pathway (spec §5)
PRODUCTION_BAR = 0.50  # target
REGISTRATION_CEILING = 0.58  # stretch (matches handoff §4)


def eta2(v: np.ndarray, lab: np.ndarray) -> float:
    """Between-cluster variance fraction (matches M9/M10/R1/R2 definition)."""
    v = v if v.ndim == 2 else v[:, None]
    keep = lab >= 0
    v, lab = v[keep], lab[keep]
    g = v.mean(0)
    tot = float(((v - g) ** 2).sum())
    if tot <= 0:
        return 0.0
    w = sum(float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum()) for l in np.unique(lab))
    return 1.0 - w / tot


def register_one(crop: np.ndarray, freqs_khz: np.ndarray):
    """M10/R1 register_one: (crop, freqs_khz) → (pitch, shape(50,), duration) or None."""
    cmax = float(crop.max())
    if cmax <= 0:
        return None
    cfg = RidgeConfig(
        transition_penalty=0.1, max_jump_bins=10,
        silence_threshold=max(1e-9, 0.02 * cmax),
    )
    fm, _ = track_ridge(crop, freqs_khz.astype(float), cfg)
    finite = np.isfinite(fm)
    idx = np.where(finite)[0]
    if len(idx) < 6:
        return None
    lo, hi = idx[0], idx[-1]
    span = fm[lo:hi + 1].copy()
    nanm = ~np.isfinite(span)
    if nanm.any():
        g = np.where(~nanm)[0]
        span[nanm] = np.interp(np.where(nanm)[0], g, span[g])
    pitch = float(span.mean())
    sc = span - pitch
    shape = np.interp(
        np.linspace(0, 1, N_RESAMPLE),
        np.linspace(0, 1, len(sc)),
        sc,
    ).astype(np.float32)
    duration = float(hi - lo + 1)
    return pitch, shape, duration


def encode_all(model, patches, band_slice, padding, device, batch: int = 64) -> np.ndarray:
    """Encode every patch → posterior mean (N, latent_dim)."""
    n = patches.shape[0]
    out = np.empty((n, model.cfg.latent_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            imgs = []
            for i in range(s, e):
                raw = np.asarray(patches[i, band_slice, :])
                x = np.log1p(raw).astype(np.float32)
                p_min, p_max = float(x.min()), float(x.max())
                x_n = (x - p_min) / max(p_max - p_min, 1e-6)
                x_p = padding.pad(x_n).astype(np.float32)
                imgs.append(x_p)
            xt = torch.from_numpy(np.stack(imgs)[:, None, :, :]).to(device, non_blocking=True)
            mu, _ = model.vae.encode(xt)
            out[s:e] = mu.cpu().numpy()
            if s % (batch * 32) == 0:
                print(f"  encoded {e}/{n}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-npz", required=True)
    ap.add_argument("--model-pt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = args.device if torch.cuda.is_available() else "cpu"
    z = np.load(args.patches_npz, mmap_mode="r", allow_pickle=False)
    patches = z["patches"]
    freqs_khz = np.asarray(z["freqs_kHz"], dtype=float)
    n, f_bins, t_in = patches.shape
    band_slice, b0, b1 = _compute_band_slice(freqs_khz)
    freqs_band_khz = freqs_khz[band_slice]
    padding = PaddingSpec.for_shape(b1 - b0, t_in, 256)

    print(
        f"[PARAM] killgate N={n} band=[{b0}:{b1}]({b1-b0}) freq_khz[{freqs_band_khz[0]:.1f},{freqs_band_khz[-1]:.1f}] "
        f"image_size=256 dev={dev} batch={args.batch} K={args.n_clusters} seed={args.seed}",
        flush=True,
    )
    print(
        f"[PARAM] thresholds kill={KILL_THRESHOLD} target={PRODUCTION_BAR} stretch={REGISTRATION_CEILING}",
        flush=True,
    )

    # ----- 1. Load model -----
    cfg = ShapeVAEv3Config()  # image_size=256, latent_dim=32 (matches run1 hyperparams)
    model = ShapeVAEv3Hybrid(cfg).to(dev)
    sd = torch.load(args.model_pt, map_location=dev, weights_only=True)
    model.load_state_dict(sd)
    print(f"[INFO] loaded model: {args.model_pt} (params={sum(p.numel() for p in model.parameters())})", flush=True)

    # ----- 2. Encode all patches -----
    t0 = time.time()
    Z = encode_all(model, patches, band_slice, padding, dev, batch=args.batch)
    print(f"[INFO] encoded {Z.shape} in {(time.time()-t0)/60:.1f} min", flush=True)

    # ----- 3. Register-shape per patch (run track_ridge again, M10-style) -----
    pitches, shapes, durations, idx_kept = [], [], [], []
    t1 = time.time()
    for i in range(n):
        crop = np.asarray(patches[i, band_slice, :])
        r = register_one(crop, freqs_band_khz)
        if r is None:
            continue
        p, sh, d = r
        pitches.append(p); shapes.append(sh); durations.append(d); idx_kept.append(i)
        if (i + 1) % 2000 == 0:
            print(f"  registered {i+1}/{n} (kept {len(idx_kept)}, {(time.time()-t1):.0f}s)", flush=True)
    pitches = np.array(pitches, dtype=np.float32)
    shapes = np.stack(shapes).astype(np.float32)
    durations = np.array(durations, dtype=np.float32)
    idx_kept = np.array(idx_kept, dtype=np.int64)
    Zk = Z[idx_kept]
    print(
        f"[INFO] registered {len(idx_kept)}/{n} patches "
        f"(skipped {n-len(idx_kept)} with <6 active cols) in {(time.time()-t1)/60:.1f} min",
        flush=True,
    )

    # ----- 4. KMeans on latents + scorecard -----
    print(f"[INFO] KMeans(K={args.n_clusters}) on {Zk.shape} latents ...", flush=True)
    km = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=args.seed)
    labels = km.fit_predict(Zk)

    score = {
        "shape": eta2(shapes, labels),
        "pitch": eta2(pitches, labels),
        "duration": eta2(durations, labels),
        "n_total": int(n),
        "n_valid": int(len(idx_kept)),
        "k_clusters": int(args.n_clusters),
        "kill_threshold": KILL_THRESHOLD,
        "production_baseline": 0.12,
        "registration_ceiling": REGISTRATION_CEILING,
    }
    verdict = "PASS_KILL_GATE" if score["shape"] >= KILL_THRESHOLD else "KILL_PATHWAY"
    score["verdict"] = verdict

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(score, indent=2))
    print("\n===== B+A HYBRID KILL-GATE SCORECARD =====")
    print(
        f"  shape η² = {score['shape']:.4f}   (kill<{KILL_THRESHOLD}, "
        f"prod=0.12, target {PRODUCTION_BAR}, stretch {REGISTRATION_CEILING})"
    )
    print(f"  pitch η² = {score['pitch']:.4f}   (LOWER is better — invariance)")
    print(f"  dur   η² = {score['duration']:.4f}   (LOWER is better — invariance)")
    print(f"  N total={n}, valid={len(idx_kept)}, K={args.n_clusters}")
    print(f"  VERDICT: {verdict}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
