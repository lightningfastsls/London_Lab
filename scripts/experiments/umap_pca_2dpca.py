#!/usr/bin/env python
"""Quick UMAP of flattened-PCA features on the VocalMat 12-class corpus.

Visual companion to the 2DPCA verification: the linear method ceilings at
macro-F1 ~0.24. If the classes are genuinely not linearly separable, a UMAP of
the PCA features (the exact features the parity control fed to LDA/SVM) should
look like a smushed continuum, not 12 separable islands.

PCA(k=50) on flattened 64x64 grayscale -> UMAP(2D), colored by class.
Per repo convention: parameters printed up top.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402

SPLIT_DIR = _REPO_ROOT / "results/twodpca_vocalmat/split"
RESIZE = 64
PCA_K = 50
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
SEED = 1729
OUT_PNG = _REPO_ROOT / "results/twodpca_verify/umap_pca_features.png"


def load_images(csv: Path, resize: int):
    from PIL import Image

    df = pd.read_csv(csv)
    imgs = np.empty((len(df), resize * resize), dtype=np.float32)
    for i, rel in enumerate(df["path"].tolist()):
        p = Path(rel)
        if not p.is_absolute():
            p = _REPO_ROOT / rel
        with Image.open(p) as im:
            im = im.convert("L").resize((resize, resize), Image.BILINEAR)
            imgs[i] = (np.asarray(im, dtype=np.float32) / 255.0).ravel()
    return imgs, df["class"].to_numpy()


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import umap

    print("=" * 64)
    print("UMAP of flattened-PCA features (VocalMat 12-class)")
    print("=" * 64)
    print(f"split_dir     : {SPLIT_DIR}")
    print(f"resize        : {RESIZE}x{RESIZE} grayscale, flattened to {RESIZE*RESIZE}")
    print(f"PCA k         : {PCA_K}")
    print(f"UMAP          : n_neighbors={UMAP_NEIGHBORS} min_dist={UMAP_MIN_DIST} seed={SEED}")
    print(f"classes       : {list(GRIMSLEY_12_CLASSES)}")

    X_list, y_list = [], []
    for name in ("train", "val", "test"):
        xi, yi = load_images(SPLIT_DIR / f"{name}.csv", RESIZE)
        X_list.append(xi); y_list.append(yi)
    X = np.vstack(X_list); y = np.concatenate(y_list)
    print(f"total points  : {len(X)}")
    vals, cnts = np.unique(y, return_counts=True)
    print("per-class     : " + ", ".join(f"{v}={c}" for v, c in zip(vals, cnts)))

    t0 = time.perf_counter()
    pca = PCA(n_components=PCA_K, random_state=SEED)
    Xp = pca.fit_transform(X)
    print(f"PCA {PCA_K}-dim EVR : {pca.explained_variance_ratio_.sum():.3f} ({time.perf_counter()-t0:.1f}s)")

    t1 = time.perf_counter()
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        n_components=2, random_state=SEED, metric="euclidean",
    )
    emb = reducer.fit_transform(Xp)
    print(f"UMAP fit      : {time.perf_counter()-t1:.1f}s")

    # Plot: one panel colored by class.
    classes = list(GRIMSLEY_12_CLASSES)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 9))
    for ci, cls in enumerate(classes):
        m = y == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.45,
                   color=cmap(ci % 20), label=f"{cls} (n={int(m.sum())})", linewidths=0)
    ax.set_title(f"UMAP of PCA-{PCA_K} features (flattened 64x64) — VocalMat 12-class\n"
                 f"linear-method macro-F1 ceiling = 0.24; look for continuum vs separable islands")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=3, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
