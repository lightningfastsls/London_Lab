"""Re-score VQ codes against the FULL VocalMat taxonomy (no coarse 'jump' merge).

The standing harness collapses the 12 VocalMat classes into 4 super-families
(chevron/jump/flat/complex) via group_family(); 'jump' merges Step up + Step
down + Two steps + Multi-steps. That merge hides the fact that the VQ codes
SPLIT the step-types. This script scores against the raw VocalMat labels.

Produces, per run: code x VocalMat-class heatmap (counts + row-normalized) and
NMI(vq_code, fine_vocalmat_label) alongside the coarse-family NMI for contrast.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

HARNESS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, HARNESS_DIR)
from eval_shape_human_anchored import build_join, group_family  # noqa: E402

# VocalMat 11-class order (+ Noise), coarse arcs first.
VM_CLASSES = ["Flat", "Short", "Up-FM", "Down-FM", "Chevron", "Reverse Chevron",
              "Complex", "Step up", "Step down", "Two steps", "Multi-steps", "Noise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    codes = pd.read_parquet(os.path.join(a.run, "codes.parquet"))
    vq = codes["vq_code"].to_numpy().astype(int)
    K = int(vq.max()) + 1
    m = np.load(a.meta, allow_pickle=True)

    rows, joined = build_join(m["wav_stem"].astype(str), m["call_id"].astype(int),
                              pd.read_csv(a.human), offset=-1)
    fine = joined["shape_label"].to_numpy()
    keep = fine != "unclear"
    rows_k, fine_k = rows[keep], fine[keep]
    coarse_k = np.array([group_family(v) for v in fine_k])
    vq_k = vq[rows_k]

    nmi_fine = normalized_mutual_info_score(fine_k, vq_k)
    nmi_coarse = normalized_mutual_info_score(coarse_k, vq_k)
    print(f"{a.run}: N labeled={len(fine_k)}  "
          f"NMI(fine VocalMat)={nmi_fine:.3f}  NMI(coarse 4-family)={nmi_coarse:.3f}")

    present = [c for c in VM_CLASSES if (fine_k == c).any()]
    M = np.zeros((K, len(present)))
    for ci in range(K):
        sub = fine_k[vq_k == ci]
        for j, c in enumerate(present):
            M[ci, j] = (sub == c).sum()
    order = sorted(range(K), key=lambda c: -M[c].sum())     # most-labeled codes first
    Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(1.05 * len(present) + 3, 0.36 * K + 2))
    im = ax.imshow(Mn[order], aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"code {c}" for c in order], fontsize=7)
    for ci, c in enumerate(order):
        for j in range(len(present)):
            if M[c, j] > 0:
                ax.text(j, ci, int(M[c, j]), ha="center", va="center",
                        color="w" if Mn[c, j] < 0.6 else "k", fontsize=6)
    ax.set_title(f"VQ code x FULL VocalMat taxonomy ({a.run.split('/')[-1]})\n"
                 f"NMI fine={nmi_fine:.3f} vs coarse-4family={nmi_coarse:.3f}  "
                 f"(incumbent k20 coarse=0.178). Row-normalized; counts overlaid.", fontsize=9)
    fig.colorbar(im, ax=ax, label="fraction of code's labeled members")
    fig.tight_layout()
    p = os.path.join(a.out, "heatmap_vocalmat_fine.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p)


if __name__ == "__main__":
    main()
