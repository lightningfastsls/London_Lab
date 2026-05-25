"""R2 (rig) — productionize the registered-shape alphabet.

Re-runs the EXACT R1 ridge extraction (rig_R1_true_ridges.py) but joins each
surviving ridge to its call identity from patches_manifest.parquet, fits the
K=20 shape alphabet, and aggregates to ONE shape-letter per call.

Why re-extract instead of reusing true_registered_ridges.npz?
  R1 saved only `shapes` + `cohort` per ridge; it dropped <6-active-column
  patches WITHOUT recording which patch_idx survived. The extraction order is
  deterministic (cohort order x patch-row order, drop-if-<6-active), so a faithful
  re-run reproduces the same 67,337 ridges AND lets us attach (wav_stem, call_id,
  abs_time_start_s) from the manifest. We assert np.allclose vs the cached R1
  ridges as a hard guarantee that the re-run is faithful.

Outputs (parallel dirs — never overwrites latent or R1 artifacts):
  /data/shachar/contour_vae/results/latent_transitions/shape_alphabet/
      true_registered_ridges_meta.npz   (shapes, patch_label, cohort/wav_stem/call_id/abs_time)
      shape_call_letters.parquet        (cohort, wav_stem, call_id, begin_time_s, shape_letter, n_patches)
  /data/mickey_london_lab/models/shape_kmeans/k20.joblib

Memory: processed PER COHORT (peak ~13 GB for lab) — proven safe on the 31 GB rig.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, "/data/mickey_london_lab/src")
from usv_spectrogram.features.ridge_tracker import track_ridge, RidgeConfig

R = Path("/data/shachar/contour_vae")
MP = R / "results/masked_patches"
OUT = R / "results/latent_transitions/shape_alphabet"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = Path("/data/mickey_london_lab/models/shape_kmeans")
MODELS.mkdir(parents=True, exist_ok=True)
CACHED = R / "results/latent_transitions/shape_registered_TRUE/true_registered_ridges.npz"

COHORTS = ["5970", "3452", "9252", "lab_131204"]   # R1 order — MUST match
BAND0, BAND1 = 35, 205
N_RESAMPLE = 50
MIN_ACTIVE_COLS = 6


def register_one(crop, freqs_khz):
    """Verbatim from rig_R1_true_ridges.py. -> shape50 or None."""
    thr = max(1e-9, 0.02 * float(crop.max()))
    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10, silence_threshold=thr)
    fm, am = track_ridge(crop, freqs_khz.astype(float), cfg)
    active = np.isfinite(fm)
    if active.sum() < MIN_ACTIVE_COLS:
        return None
    idx = np.where(active)[0]; lo, hi = idx[0], idx[-1]
    span = fm[lo:hi + 1].copy()
    nanm = ~np.isfinite(span)
    if nanm.any():
        good = np.where(~nanm)[0]
        span[nanm] = np.interp(np.where(nanm)[0], good, span[good])
    pitch = float(span.mean())
    sc = span - pitch
    shape = np.interp(np.linspace(0, 1, N_RESAMPLE), np.linspace(0, 1, len(sc)), sc)
    return shape.astype(np.float32)


def main():
    t0 = time.time()
    shapes = []
    meta = []  # (cohort, wav_stem, call_id, abs_time_start_s)
    for c in COHORTS:
        p = MP / f"{c}_focus/patches.npz"
        man = pd.read_parquet(MP / f"{c}_focus/patches_manifest.parquet").reset_index(drop=True)
        z = np.load(p)
        arr = z["patches"]
        freqs = z["freqs_kHz"][BAND0:BAND1]
        assert len(man) == arr.shape[0], f"{c}: manifest {len(man)} != patches {arr.shape[0]}"
        print(f"[INFO] {c}: {arr.shape[0]} patches", flush=True)
        for i in range(arr.shape[0]):
            r = register_one(arr[i, BAND0:BAND1, :], freqs)
            if r is not None:
                shapes.append(r)
                row = man.iloc[i]
                meta.append((c, str(row["wav_stem"]), int(row["call_id"]),
                             float(row["abs_time_start_s"])))
            if i % 10000 == 0 and i:
                print(f"  {c} {i}/{arr.shape[0]}  ridges={len(shapes)}  "
                      f"({len(shapes)/(time.time()-t0):.0f}/s)", flush=True)
        del arr, z
    Sh = np.array(shapes, np.float32)
    meta = pd.DataFrame(meta, columns=["cohort", "wav_stem", "call_id", "abs_time_start_s"])
    print(f"[INFO] total ridges: {len(Sh)} in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- Hard guarantee: faithful re-extraction matches cached R1 ----
    cached = np.load(CACHED, allow_pickle=True)
    assert len(Sh) == len(cached["shapes"]), \
        f"count mismatch: re-extract {len(Sh)} != cached {len(cached['shapes'])}"
    if not np.allclose(Sh, cached["shapes"], atol=1e-5):
        mad = float(np.abs(Sh - cached["shapes"]).max())
        raise SystemExit(f"[FATAL] ridges differ from cached R1 (max abs diff {mad}); "
                         f"extraction is NOT faithful — aborting.")
    print("[OK] re-extraction is byte-faithful to cached R1 ridges (allclose).", flush=True)

    # ---- Fit K=20 shape alphabet (same recipe as R1's lab_shape) ----
    km = KMeans(20, n_init=10, random_state=42).fit(Sh)
    patch_label = km.labels_.astype(np.int32)
    cached_lab = cached["lab_shape"]
    agree = float((patch_label == cached_lab).mean())
    ari = float(adjusted_rand_score(cached_lab, patch_label))
    print(f"[CHECK] refit vs cached lab_shape: raw_agreement={agree:.4f}  ARI={ari:.4f}", flush=True)
    joblib.dump(km, MODELS / "k20.joblib")
    np.savez_compressed(
        OUT / "true_registered_ridges_meta.npz",
        shapes=Sh, patch_label=patch_label,
        cohort=meta["cohort"].to_numpy(),
        wav_stem=meta["wav_stem"].to_numpy(),
        call_id=meta["call_id"].to_numpy(),
        abs_time_start_s=meta["abs_time_start_s"].to_numpy(),
    )

    # ---- Aggregate to one shape-letter per call ----
    # Per call: average the 50-pt registered ridges (all aligned), then predict.
    meta["row"] = np.arange(len(meta))
    recs = []
    for (co, ws, cid), g in meta.groupby(["cohort", "wav_stem", "call_id"], sort=False):
        rows = g["row"].to_numpy()
        mean_ridge = Sh[rows].mean(0)
        letter = int(km.predict(mean_ridge[None, :])[0])
        recs.append((co, ws, cid, float(g["abs_time_start_s"].min()), letter, int(len(rows))))
    call_df = pd.DataFrame(
        recs, columns=["cohort", "wav_stem", "call_id", "begin_time_s",
                       "shape_letter", "n_patches"])
    call_df.to_parquet(OUT / "shape_call_letters.parquet")
    print(f"[INFO] calls with a shape-letter: {len(call_df)}", flush=True)
    print(f"[INFO] per-cohort: {call_df['cohort'].value_counts().to_dict()}", flush=True)
    print(f"[INFO] letter histogram: {pd.Series(call_df['shape_letter']).value_counts().sort_index().to_dict()}",
          flush=True)
    print(f"[DONE] {OUT}  +  {MODELS/'k20.joblib'}  total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
