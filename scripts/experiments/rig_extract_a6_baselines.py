"""α₃-C A6 — extract the two MANDATORY eval baselines on the rig.

The eval-validity rule (handoff 2026-05-30): a learned shape representation
must beat BOTH a random-init-encoder AND a column-mean-identity baseline by
≥0.10 NMI, and the baselines MUST be computed on the SAME substrate the
production VAE saw. That substrate is the contour-masked patches in
``results/masked_patches/combined_all_cohorts/patches.npz``, preprocessed by
``train_contour_vae_v2.py`` (band-crop to corpus USV band → log1p → per-patch
min/max [0,1] → zero-pad to 256×256). The production latents
(``results/contour_vae_combined/latents.parquet``, z_0..z_31) were produced by
that script's ``ImageVAE`` (z=32).

To guarantee byte-identical preprocessing + architecture we IMPORT the
producer's own classes rather than re-implement them:
  - ``MaskedPatchDataset`` / ``PaddingSpec`` / ``_compute_band_slice`` — the
    exact patch→tensor pipeline.
  - ``ImageVAE`` / ``ImageVAEConfig`` — the exact encoder; instantiated
    UNTRAINED with a fixed seed → the random-init-encoder baseline.

Per labeled call_id (matched_call_id namespace, joined via the bridge keymap)
we emit, mean-pooled over that call_id's windows (matching the eval's
``--agg mean``):
  - ``z_random_*`` (32 dims) = untrained ImageVAE.encode_mean(x).
  - ``identity_*`` (256 dims) = column_mean of the (1,256,256) VAE input
    (mean over the freq axis → per-time-column vector), i.e. the
    ``column_mean_features`` recipe applied to the production substrate.

Output: ``--out`` parquet (call_id, z_random_0..31, identity_0..255) → rsync to
the box → ``eval_a6_existing_latents.py --baseline-features``.

Read-only on all production artifacts; writes only ``--out``. Compute launch is
gated per-session (feedback_rig_claude_mediation) — run only with the user OK.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import the producer's exact preprocessing + model (byte-identical substrate).
PRODUCER_DIR = "/data/shachar/contour_vae/scripts"
if PRODUCER_DIR not in sys.path:
    sys.path.insert(0, PRODUCER_DIR)
from train_contour_vae_v2 import (  # noqa: E402
    ImageVAE,
    ImageVAEConfig,
    MaskedPatchDataset,
    PaddingSpec,
    _compute_band_slice,
    IMAGE_SIZE,
)

NPZ = "/data/shachar/contour_vae/results/masked_patches/combined_all_cohorts/patches.npz"
MANIFEST = "/data/shachar/contour_vae/results/masked_patches/combined_all_cohorts/patches_manifest.parquet"


def column_mean_features(x_bchw: np.ndarray) -> np.ndarray:
    """Identity baseline = per-column mean of the [B,1,H,W] VAE input
    (mean over freq rows H) → [B,W]. Matches train_shape_vae_alpha3's
    column_mean_features exactly (X[:,0].mean(axis=1))."""
    return x_bchw[:, 0, :, :].mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keymap", required=True, type=Path,
                    help="CSV: wav_stem,call_id,window_idx,matched_call_id,overlap_frac "
                         "(exported from the box bridge).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cohort", default="lab_131204")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--agg", choices=["mean", "max_overlap"], default="mean",
                    help="per-call_id aggregation; must match the eval (default mean).")
    a = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = a.device if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("α₃-C A6 baselines — random-init ImageVAE + column-mean identity")
    print("=" * 72)
    print(f"  npz       : {NPZ}")
    print(f"  manifest  : {MANIFEST}")
    print(f"  keymap    : {a.keymap}")
    print(f"  cohort    : {a.cohort}")
    print(f"  seed      : {a.seed}   device: {dev}   agg: {a.agg}")
    print(f"  IMAGE_SIZE: {IMAGE_SIZE}")

    # ---- manifest is positionally aligned with patches.npz rows ----
    man = pd.read_parquet(MANIFEST).reset_index(drop=True)
    man["npz_row"] = np.arange(len(man), dtype=np.int64)
    man = man[man["cohort"] == a.cohort].copy()
    man["call_id"] = man["call_id"].astype(int)
    man["window_idx"] = man["window_idx"].astype(int)
    print(f"\n  manifest rows (cohort={a.cohort}): {len(man)}")

    km = pd.read_csv(a.keymap)
    km["call_id"] = km["call_id"].astype(int)
    km["window_idx"] = km["window_idx"].astype(int)
    rows = man.merge(km, on=["wav_stem", "call_id", "window_idx"], how="inner")
    print(f"  joined to keymap (matched_call_id): {len(rows)} "
          f"({rows['matched_call_id'].nunique()} distinct call_ids)")
    if len(rows) == 0:
        raise SystemExit("0 rows after keymap join — check key dtypes/namespaces.")

    # ---- substrate: mmap patches, build the producer's dataset (identical preproc) ----
    z = np.load(NPZ, mmap_mode="r")
    patches = z["patches"]            # (N, F, T) raw power, memmap
    freqs = z["freqs_kHz"]
    band_slice, b0, b1 = _compute_band_slice(np.asarray(freqs))
    f_in = b1 - b0
    t_in = patches.shape[2]
    padding = PaddingSpec.for_shape(f_in, t_in, IMAGE_SIZE)
    print(f"  band_slice: [{b0}:{b1}] (f_in={f_in})  t_in={t_in}  pad→{IMAGE_SIZE}²")
    ds = MaskedPatchDataset(patches, band_slice, padding)

    # ---- untrained encoder (random-init baseline) ----
    model = ImageVAE(ImageVAEConfig()).to(dev).eval()

    npz_rows = rows["npz_row"].to_numpy()
    n = len(npz_rows)
    z_rand = np.zeros((n, 32), dtype=np.float32)
    ident = np.zeros((n, IMAGE_SIZE), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, a.batch):
            sl = npz_rows[s:s + a.batch]
            batch = torch.stack([ds[int(i)] for i in sl]).to(dev)  # (B,1,256,256)
            mu = model.encode_mean(batch).cpu().numpy()
            z_rand[s:s + len(sl)] = mu
            ident[s:s + len(sl)] = column_mean_features(batch.cpu().numpy())
            if (s // a.batch) % 10 == 0:
                print(f"    {s + len(sl)}/{n}", flush=True)

    # ---- assemble per-window, aggregate per matched_call_id ----
    zr = pd.DataFrame(z_rand, columns=[f"z_random_{i}" for i in range(32)])
    idn = pd.DataFrame(ident, columns=[f"identity_{i}" for i in range(IMAGE_SIZE)])
    feat = pd.concat(
        [rows[["matched_call_id", "overlap_frac"]].reset_index(drop=True), zr, idn], axis=1
    )
    feat = feat.rename(columns={"matched_call_id": "call_id"})

    if a.agg == "max_overlap":
        idx = feat.groupby("call_id")["overlap_frac"].idxmax()
        out = feat.loc[idx].drop(columns=["overlap_frac"]).reset_index(drop=True)
    else:
        out = feat.drop(columns=["overlap_frac"]).groupby("call_id").mean().reset_index()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"\n  per-call_id baseline rows: {len(out)}")
    print(f"  cols: call_id + {sum(c.startswith('z_random_') for c in out.columns)} z_random_* "
          f"+ {sum(c.startswith('identity_') for c in out.columns)} identity_*")
    print(f"  wrote {a.out}  ({(time.time()-t0)/60:.1f} min)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
