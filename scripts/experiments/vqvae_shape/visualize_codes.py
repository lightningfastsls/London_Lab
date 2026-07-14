"""Visualize the groups a trained global-code VQ-VAE created, so a human can
eyeball whether the codes are coherent shape families or just continuum tiles.

Three views:
  1. GALLERY  -- one panel per code: mean registered ridge (bold) + member
     ridges (faint) + count + dominant human label. The "prototype shape" of
     each group. Codes sorted by net slope (down-ramp -> flat -> up-ramp).
  2. HEATMAP  -- code x human-family: does any code concentrate a real family?
  3. MAP      -- 2-D UMAP/PCA of the latent, colored by code: separated islands
     or one blob?

Usage:
    .venv/bin/python scripts/experiments/vqvae_shape/visualize_codes.py \
        --run results/vqvae_shape/k20 \
        --meta data/shape_substrate/true_registered_ridges_meta.npz \
        --lab  data/shape_substrate/true_registered_ridges.npz \
        --human data/manual_shape_labels.csv --out results/vqvae_shape/k20/viz
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

HARNESS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, HARNESS_DIR)
from eval_shape_human_anchored import build_join, group_family  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=60, help="member ridges drawn per panel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    codes = pd.read_parquet(os.path.join(a.run, "codes.parquet"))
    latents = np.load(os.path.join(a.run, "latents.npy"))
    vq = codes["vq_code"].to_numpy().astype(int)
    K = int(vq.max()) + 1

    lab = np.load(a.lab, allow_pickle=True)
    m = np.load(a.meta, allow_pickle=True)
    Sh = lab["shapes"].astype(np.float32)               # (N,50) pitch-centered kHz
    t = np.linspace(0, 1, Sh.shape[1])

    # human labels per ridge row (sparse) -----------------------------------
    rows, joined = build_join(m["wav_stem"].astype(str), m["call_id"].astype(int),
                              pd.read_csv(a.human), offset=-1)
    fam_of_row = {}
    for r, lb in zip(rows, joined["shape_label"].to_numpy()):
        if lb != "unclear":
            fam_of_row[int(r)] = group_family(lb)
    families = ["flat", "jump", "chevron", "complex", "Down-FM", "Up-FM",
                "Short", "Noise"]

    # per-code stats + dominant human label ---------------------------------
    stats = []
    ylo, yhi = np.percentile(Sh, [2, 98])
    for c in range(K):
        idx = np.where(vq == c)[0]
        mean_ridge = Sh[idx].mean(0)
        net_slope = float(mean_ridge[-1] - mean_ridge[0])
        labs = [fam_of_row[i] for i in idx if i in fam_of_row]
        if labs:
            vc = pd.Series(labs).value_counts()
            dom = f"{vc.index[0]} {vc.iloc[0]}/{len(labs)}"
        else:
            dom = "(no labels)"
        stats.append({"code": c, "n": len(idx), "slope": net_slope,
                      "mean": mean_ridge, "idx": idx, "dom": dom})
    order = sorted(range(K), key=lambda c: stats[c]["slope"])

    # ---- VIEW 1: gallery ---------------------------------------------------
    ncol = 5 if K <= 25 else 8
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.2),
                             squeeze=False)
    for panel, c in enumerate(order):
        ax = axes[panel // ncol][panel % ncol]
        s = stats[c]
        draw = s["idx"] if len(s["idx"]) <= a.members else rng.choice(s["idx"], a.members, replace=False)
        for i in draw:
            ax.plot(t, Sh[i], color="#6f9dff", alpha=0.06, lw=0.6)
        ax.plot(t, s["mean"], color="#c0392b", lw=2.0)
        ax.axhline(0, color="#999", lw=0.5, ls=":")
        ax.set_ylim(ylo, yhi)
        ax.set_title(f"code {c} | n={s['n']}\n{s['dom']}", fontsize=8)
        ax.set_xticks([]); ax.tick_params(labelsize=6)
    for panel in range(K, nrow * ncol):
        axes[panel // ncol][panel % ncol].axis("off")
    fig.suptitle(f"VQ-VAE code prototypes ({a.run.split('/')[-1]}, K={K}) — "
                 f"mean registered ridge (red) over members (blue). "
                 f"y = pitch-centered kHz, x = normalized time. Sorted by net slope.",
                 fontsize=10)
    fig.supylabel("centered frequency (kHz)", fontsize=8)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    p1 = os.path.join(a.out, "gallery.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # ---- VIEW 2: code x family heatmap ------------------------------------
    present = [f for f in families if any(v == f for v in fam_of_row.values())]
    M = np.zeros((K, len(present)))
    for c in range(K):
        labs = [fam_of_row[i] for i in stats[c]["idx"] if i in fam_of_row]
        for j, f in enumerate(present):
            M[c, j] = labs.count(f)
    Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)      # row-normalized
    fig, ax = plt.subplots(figsize=(1.1 * len(present) + 3, 0.35 * K + 2))
    im = ax.imshow(Mn[order], aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"code {c} (n={stats[c]['n']})" for c in order], fontsize=7)
    for ci, c in enumerate(order):
        for j in range(len(present)):
            if M[c, j] > 0:
                ax.text(j, ci, int(M[c, j]), ha="center", va="center",
                        color="w" if Mn[c, j] < 0.6 else "k", fontsize=6)
    ax.set_title(f"Which human shape family lands in which code (row-normalized)\n"
                 f"{a.run.split('/')[-1]} — counts overlaid; only labeled calls", fontsize=9)
    fig.colorbar(im, ax=ax, label="fraction of code's labeled members")
    fig.tight_layout()
    p2 = os.path.join(a.out, "family_heatmap.png")
    fig.savefig(p2, dpi=130); plt.close(fig)

    # ---- VIEW 3: 2-D latent map -------------------------------------------
    try:
        import umap
        emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(latents)
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=0).fit_transform(latents)
        method = "PCA"
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=vq, cmap="tab20" if K <= 20 else "gist_ncar",
                    s=2, alpha=0.35, linewidths=0)
    ax.set_title(f"Latent {method} colored by VQ code ({a.run.split('/')[-1]}, K={K})\n"
                 f"islands = discrete kinds; one blob = continuum tiles", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, label="VQ code", ticks=range(0, K, max(1, K // 20)))
    fig.tight_layout()
    p3 = os.path.join(a.out, f"latent_{method.lower()}.png")
    fig.savefig(p3, dpi=130); plt.close(fig)

    print("wrote:")
    for p in (p1, p2, p3):
        print(" ", p)


if __name__ == "__main__":
    main()
