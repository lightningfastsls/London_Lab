"""α₃-C A6 (staged path): evaluate the EXISTING production contour-VAE latents
as a shape representation — no retraining.

The user chose "existing latents first" (the cheap, informative move; modal
prior P(2-D adds nothing) ≥ 0.5). This driver feeds the bridged production
latents into the SAME metric harness `train_shape_vae_alpha3.run_metrics`
(reused, not reinvented — NMI / chevron-kNN purity / linear probe), against
either the oracle taxonomy or the γ hand-labels.

Inputs
------
  --bridge   : lab_131204_latent_bridge.parquet from build_a6_latent_bridge.py
               (z_* + matched_call_id, possibly many latents per call_id).
  --labels   : either the oracle CSV (call_id, top1_class, top1_prob,
               high_confidence) OR the γ CSV (call_id, shape_label).
  --label-kind {oracle,gamma}
  --baseline-features (optional): parquet with per-call_id baseline reps
               (z_random_*, identity_*) from rig_extract_a6_baselines.py. If
               absent, a PRELIMINARY random-Gaussian placeholder is used and
               the verdict is marked NON-BINDING (the eval-validity rule
               requires the real random-init-encoder + column-mean-identity
               baselines).

Many-to-one handling: a call_id can map to >1 latent window (mean 1.39). We
mean-pool z_* per call_id by default (`--agg mean`) or take the max-overlap
representative (`--agg max_overlap`).

Decision gates (from the roadmap; do NOT re-derive here):
  SHIP : NMI ≥ 0.25  ∧  chevron kNN ≥ 0.50  ∧  beats BOTH baselines by ≥ 0.10
  KILL : NMI < 0.15  OR  fails the baseline margin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

# Reuse the canonical metric harness + label taxonomy (do not duplicate).
from train_shape_vae_alpha3 import (  # noqa: E402
    GEOM_FAMILY, run_metrics,
)
from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402

Z_COLS = [f"z_{i}" for i in range(32)]

# 2026-05-30: γ now uses the FULL 12-class VocalMat taxonomy (same label space as
# the oracle), so γ→coarse-shape folding reuses the SAME GEOM_FAMILY map as the
# oracle side — identical folding on both axes (Chevron+Reverse Chevron→chevron,
# Step*/Multi-steps→jump, Complex→complex, Flat→flat; the rest → 'others').


def aggregate_latents(bridge: pd.DataFrame, agg: str) -> pd.DataFrame:
    """Collapse many latents/call_id to one row per matched_call_id."""
    b = bridge[bridge["matched_call_id"].notna()].copy()
    zc = [c for c in Z_COLS if c in b.columns]
    if agg == "max_overlap":
        idx = b.groupby("matched_call_id")["overlap_frac"].idxmax()
        out = b.loc[idx, ["matched_call_id"] + zc].reset_index(drop=True)
    else:  # mean-pool
        out = b.groupby("matched_call_id")[zc].mean().reset_index()
    return out.rename(columns={"matched_call_id": "call_id"})


def load_oracle_labels(path: Path, min_prob: float, use_high_conf: bool):
    lab = pd.read_csv(path)
    if use_high_conf and "high_confidence" in lab.columns:
        hc = lab["high_confidence"]
        if hc.dtype == object:
            hc = hc.astype(str).str.lower().isin(["true", "1", "yes"])
        lab = lab[hc.astype(bool)]
        gate = "high_confidence==True"
    elif "top1_prob" in lab.columns:
        lab = lab[lab["top1_prob"].astype(float) >= min_prob]
        gate = f"top1_prob>={min_prob}"
    else:
        gate = "none"
    lab = lab[["call_id", "top1_class"]].rename(columns={"top1_class": "label_name"})
    name_to_idx = {c: i for i, c in enumerate(GRIMSLEY_12_CLASSES)}
    lab["class_idx"] = lab["label_name"].map(lambda c: name_to_idx.get(c, -1)).astype(int)
    lab["geom"] = lab["label_name"].map(lambda c: GEOM_FAMILY.get(c, "others"))
    return lab, gate


def load_gamma_labels(path: Path):
    lab = pd.read_csv(path)
    lab = lab[lab["shape_label"].astype(str).str.lower() != "unclear"].copy()
    lab = lab.rename(columns={"shape_label": "label_name"})
    codes, _ = pd.factorize(lab["label_name"])
    lab["class_idx"] = codes.astype(int)
    lab["geom"] = lab["label_name"].map(lambda c: GEOM_FAMILY.get(str(c), "others"))
    return lab[["call_id", "label_name", "class_idx", "geom"]], "gamma-12class (unclear dropped)"


def make_val_mask(n: int, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    n_val = max(1, int(round(n * frac)))
    mask[rng.choice(n, size=min(n_val, n), replace=False)] = True
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bridge", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--label-kind", choices=["oracle", "gamma"], required=True)
    ap.add_argument("--baseline-features", type=Path, default=None)
    ap.add_argument("--agg", choices=["mean", "max_overlap"], default="mean")
    ap.add_argument("--min-prob", type=float, default=0.85)
    ap.add_argument("--no-high-conf", action="store_true",
                    help="oracle: use min-prob gate instead of high_confidence flag")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON scorecard out")
    a = ap.parse_args()

    bridge = pd.read_parquet(a.bridge)
    lat = aggregate_latents(bridge, a.agg)

    if a.label_kind == "oracle":
        lab, gate = load_oracle_labels(a.labels, a.min_prob, not a.no_high_conf)
    else:
        lab, gate = load_gamma_labels(a.labels)

    df = lat.merge(lab, on="call_id", how="inner").reset_index(drop=True)
    n = len(df)
    print(f"=== A6 existing-latents eval — anchor={a.label_kind} ===")
    print(f"label gate                : {gate}")
    print(f"latents (per call_id, {a.agg}-agg): {len(lat)}")
    print(f"labeled rows joined       : {n}")
    if n < 30:
        print(f"\n⚠ only {n} labeled rows — metrics will be high-variance.")
    if n == 0:
        raise SystemExit("0 joined rows — check call_id namespaces match.")

    Z = df[[c for c in Z_COLS if c in df.columns]].to_numpy(dtype=np.float64)
    class_idx = df["class_idx"].to_numpy()
    geom = df["geom"].to_numpy().astype(object)
    val_mask = make_val_mask(n, a.val_frac, a.seed)

    print("\nlabel distribution:")
    print(df["label_name"].value_counts().to_string())
    print("\ngeom family distribution:")
    print(pd.Series(geom).value_counts().to_string())

    # ---- learned (existing production latents) ----
    m_learned = run_metrics(Z, class_idx, geom, val_mask, a.knn_k)

    # ---- baselines ----
    binding = a.baseline_features is not None
    if binding:
        bf = pd.read_parquet(a.baseline_features)
        bf = bf.merge(df[["call_id"]], on="call_id", how="right")
        rnd_cols = [c for c in bf.columns if c.startswith("z_random_")]
        idn_cols = [c for c in bf.columns if c.startswith("identity_")]
        Zr = bf[rnd_cols].to_numpy(dtype=np.float64)
        Zi = bf[idn_cols].to_numpy(dtype=np.float64)
        m_rand = run_metrics(Zr, class_idx, geom, val_mask, a.knn_k)
        m_idn = run_metrics(Zi, class_idx, geom, val_mask, a.knn_k)
        base_note = "BINDING (real random-init-encoder + column-mean-identity baselines)"
    else:
        rng = np.random.default_rng(a.seed)
        Zr = rng.standard_normal((n, Z.shape[1]))
        m_rand = run_metrics(Zr, class_idx, geom, val_mask, a.knn_k)
        m_idn = {k: float("nan") for k in m_learned}
        base_note = ("PRELIMINARY / NON-BINDING — random-Gaussian placeholder; "
                     "real baselines require rig_extract_a6_baselines.py")

    # ---- scorecard ----
    rows = [
        ("nmi_kmeans20", "NMI (KMeans20 vs anchor)"),
        ("knn_purity_chevron", "kNN purity chevron"),
        ("knn_purity_jump", "kNN purity jump"),
        ("knn_purity_complex", "kNN purity complex"),
        ("knn_purity_flat", "kNN purity flat"),
        ("linear_probe_val_acc", "linear probe val acc"),
    ]
    def f(x):
        return "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
    print(f"\nbaselines: {base_note}")
    print("\n" + "=" * 64)
    print(f"{'metric':32s} {'learned':>8s} {'random':>8s} {'identity':>8s}")
    print("-" * 64)
    for key, lbl in rows:
        print(f"{lbl:32s} {f(m_learned.get(key)):>8s} "
              f"{f(m_rand.get(key)):>8s} {f(m_idn.get(key)):>8s}")
    print("=" * 64)

    # ---- verdict vs gates ----
    nmi = m_learned["nmi_kmeans20"]
    chev = m_learned.get("knn_purity_chevron", float("nan"))
    def margin(key):
        base = max(
            m_rand.get(key, float("-inf")) if not np.isnan(m_rand.get(key, np.nan)) else float("-inf"),
            m_idn.get(key, float("-inf")) if not np.isnan(m_idn.get(key, np.nan)) else float("-inf"),
        )
        return m_learned[key] - base
    beats = margin("nmi_kmeans20")
    print(f"\nNMI={nmi:.3f}  chevron-kNN={chev:.3f}  "
          f"NMI margin over best baseline={beats:+.3f}")
    if binding:
        if nmi >= 0.25 and (not np.isnan(chev) and chev >= 0.50) and beats >= 0.10:
            print("VERDICT: SHIP  (NMI≥0.25 ∧ chevron-kNN≥0.50 ∧ beats baselines ≥0.10)")
        elif nmi < 0.15 or beats < 0.10:
            print("VERDICT: KILL  (NMI<0.15 OR fails baseline margin) → ship registration; γ is gold")
        else:
            print("VERDICT: BORDERLINE — escalate to a fresh α₃ β-VAE (the genuine bet)")
    else:
        print("VERDICT: (deferred — baselines non-binding; rerun with --baseline-features)")

    if a.out:
        import json
        payload = {"anchor": a.label_kind, "n": int(n), "gate": gate,
                   "agg": a.agg, "binding": binding,
                   "learned": m_learned, "random": m_rand, "identity": m_idn}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
