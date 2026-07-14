"""Score a trained global-code VQ-VAE against the standing shape-eval gates.

Judges the VQ codes on the IDENTICAL ruler used for registration / soft-DTW /
elastic-FPCA, so the comparison is apples-to-apples:

  * shape eta^2  = eta2(Sh, vq_code)   -- verbatim bake-off formula.
                   baselines: registration 0.577 | M8 1-D VAE 0.42-0.50 |
                   masked contour-VAE 0.099 | pure image-VAE 0.009 | kill gate 0.12
  * NMI          = normalized_mutual_info(vq_code, human_family)
                   baseline: incumbent K=20 alphabet = 0.178
  * kNN purity   = per-family bootstrap purity (k=10, 1000x) on
                   (a) the continuous latent z_e and (b) one-hot codes,
                   via the harness's own bootstrap_purity_ci.
                   baselines (jump / flat / complex / chevron):
                   registration 0.415/0.419/0.194/0.186;
                   soft-DTW     0.522/0.396/0.243/0.214

Usage:
    .venv/bin/python scripts/experiments/vqvae_shape/eval_vq_shape.py \
        --run results/vqvae_shape/k20 \
        --meta data/shape_substrate/true_registered_ridges_meta.npz \
        --lab  data/shape_substrate/true_registered_ridges.npz \
        --human data/manual_shape_labels.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

# Import the standing harness functions so we score on the exact same ruler.
HARNESS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, HARNESS_DIR)
from eval_shape_human_anchored import build_join, group_family, bootstrap_purity_ci  # noqa: E402

FAMILIES = ["chevron", "jump", "flat", "complex"]
BASE_RATE = {"chevron": 0.072, "jump": 0.334, "flat": 0.205, "complex": 0.110}
REG = {"chevron": 0.186, "jump": 0.415, "flat": 0.419, "complex": 0.194}
SOFTDTW = {"chevron": 0.214, "jump": 0.522, "flat": 0.396, "complex": 0.243}


def eta2(v, lab):
    """Between-group variance fraction 1 - SS_within/SS_total (verbatim)."""
    v = v if v.ndim == 2 else v[:, None]
    keep = lab >= 0
    v, lab = v[keep], lab[keep]
    if len(v) == 0:
        return 0.0
    g = v.mean(0)
    tot = float(((v - g) ** 2).sum())
    w = sum(float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum())
            for l in np.unique(lab))
    return 1 - w / tot if tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--human", required=True)
    a = ap.parse_args()

    codes = pd.read_parquet(os.path.join(a.run, "codes.parquet"))
    latents = np.load(os.path.join(a.run, "latents.npy"))
    vq = codes["vq_code"].to_numpy().astype(int)
    K = int(vq.max()) + 1

    m = np.load(a.meta, allow_pickle=True)
    lab = np.load(a.lab, allow_pickle=True)
    Sh = lab["shapes"].astype(np.float32)          # (N, 50) registered ridge
    assert Sh.shape[0] == len(vq) == latents.shape[0], "row misalignment"
    ws = m["wav_stem"].astype(str)
    cid = m["call_id"].astype(int)
    human = pd.read_csv(a.human)

    print("=" * 66)
    print(f"VQ-VAE SHAPE EVAL  --  {a.run}")
    print("=" * 66)
    print(f"  N ridges       : {len(vq)}   K codes used: {len(np.unique(vq))}/{K}")

    # ---- 1. shape eta^2 on the full corpus (partition vs ridge geometry) ----
    e_vq = eta2(Sh, vq)
    e_reg = eta2(Sh, lab["lab_shape"].astype(int))   # recompute registration ceiling locally
    print("-" * 66)
    print("  SHAPE eta^2 (variance of registered ridge explained by partition)")
    print(f"    VQ codes          : {e_vq:.3f}")
    print(f"    registration k20  : {e_reg:.3f}   [reference 0.577]")
    print(f"    references        : M8 1-D VAE 0.42-0.50 | contour-VAE 0.099 | "
          f"image-VAE 0.009 | KILL gate 0.12")

    # ---- 2. join to human labels, drop 'unclear', build families ----
    rows, joined = build_join(ws, cid, human, offset=-1)
    y = joined["shape_label"].to_numpy()
    keep = y != "unclear"
    rows_k = rows[keep]
    yf = np.array([group_family(v) for v in y[keep]])
    n_lab = len(yf)
    print("-" * 66)
    print(f"  human-anchored: {n_lab} labels joined (offset=-1, 'unclear' dropped)")
    fam_counts = {f: int((yf == f).sum()) for f in FAMILIES}
    print(f"    target-family counts: {fam_counts}")

    # ---- 3. NMI of hard codes vs human families ----
    nmi = normalized_mutual_info_score(yf, vq[rows_k])
    print("-" * 66)
    print(f"  NMI(vq_code, human_family) : {nmi:.3f}   [incumbent k20 = 0.178]")

    # ---- 4. kNN purity per family: continuous latent + one-hot codes ----
    Xc = latents[rows_k]                    # continuous pre-quant latent
    Xo = np.eye(K, dtype=np.float32)[vq[rows_k]]   # one-hot codes (Hamming)
    print("-" * 66)
    print("  kNN PURITY (k=10, 1000x bootstrap)   point [lo, hi]")
    print(f"  {'family':9s} {'base':>6s} {'VQ-latent':>20s} {'VQ-onehot':>20s} "
          f"{'registr.':>8s} {'softDTW':>8s}")
    purity = {"latent": {}, "onehot": {}}
    for fam in FAMILIES:
        pc, lc, hc = bootstrap_purity_ci(Xc, yf, fam, k=10, n_boot=1000, seed=42)
        po, lo, ho = bootstrap_purity_ci(Xo, yf, fam, k=10, n_boot=1000, seed=42)
        purity["latent"][fam] = [pc, lc, hc]
        purity["onehot"][fam] = [po, lo, ho]
        print(f"  {fam:9s} {BASE_RATE[fam]:6.3f} "
              f"{pc:6.3f} [{lc:.3f},{hc:.3f}] "
              f"{po:6.3f} [{lo:.3f},{ho:.3f}] "
              f"{REG[fam]:8.3f} {SOFTDTW[fam]:8.3f}")

    # ---- verdict heuristic (non-overlapping CI vs registration) ----
    print("-" * 66)
    beats = []
    for fam in FAMILIES:
        pc, lc, hc = purity["latent"][fam]
        # beat registration if our lower CI exceeds a registration point estimate
        if lc > REG[fam]:
            beats.append(f"{fam} (latent {pc:.3f} > reg {REG[fam]:.3f})")
    verdict = ("VQ latent beats registration on: " + ", ".join(beats)) if beats \
        else "VQ latent does NOT beat registration on any family (CI test)"
    eta_verdict = ("PASSES" if e_vq >= 0.12 else "FAILS") + " eta^2 kill gate (0.12)"
    print(f"  eta^2: {eta_verdict}")
    print(f"  purity: {verdict}")
    print("=" * 66)

    out = {
        "run": a.run, "K": K, "codes_used": int(len(np.unique(vq))),
        "n_labels": n_lab, "family_counts": fam_counts,
        "eta2_vq": e_vq, "eta2_registration": e_reg,
        "nmi_vq_vs_human": nmi, "nmi_incumbent_k20": 0.178,
        "purity": purity, "purity_registration": REG, "purity_softdtw": SOFTDTW,
        "base_rate": BASE_RATE,
    }
    with open(os.path.join(a.run, "eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {a.run}/eval.json")


if __name__ == "__main__":
    main()
