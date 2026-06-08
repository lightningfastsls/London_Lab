#!/usr/bin/env python
"""Memory-safe 227x227 (no-downscale) run of the BEST 2DPCA config (2d2dpca/lda).

Why this script exists
----------------------
The experiment driver (``train_2dpca_classifier.py``) loads every image into a
float64 array and ``fit_2d2dpca`` then holds a second float64 copy
(``centered = arr - mean_image``). At 227x227 over ~12k images that is
~10 GB of float64 and OOM'd WSL during the §5 resize sweep.

This script reproduces ``TwoDPCAClassifier(variant="2d2dpca", classifier="lda")``
*exactly* (same covariance definitions, same projection, same StandardScaler+LDA
head) but accumulates the image-covariance matrices in CHUNKS, so peak memory is
one chunk (~500 imgs) at a time rather than two full float64 copies.

Faithfulness check: with ``--resize 64`` it must reproduce the library's recorded
0.2402 test macro-F1 (results/twodpca_vocalmat/summary.json). Run that first.

Per repo convention (feedback_analysis_print_params): all parameters printed up top.
"""
from __future__ import annotations

import argparse
import json
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

CLASS_TO_DISPLAY = {
    "noise": "Noise", "step_up": "Step up", "down_fm": "Down-FM", "short": "Short",
    "chevron": "Chevron", "up_fm": "Up-FM", "flat": "Flat", "two_steps": "Two steps",
    "step_down": "Step down", "complex": "Complex", "rev_chevron": "Reverse Chevron",
    "mult_steps": "Multi-steps",
}


def _iter_image_chunks(paths, image_root: Path, resize: int, chunk: int):
    """Yield (start_idx, np.float64 array (k, resize, resize)) chunks."""
    from PIL import Image

    buf = np.empty((chunk, resize, resize), dtype=np.float64)
    n = len(paths)
    for start in range(0, n, chunk):
        rels = paths[start : start + chunk]
        k = len(rels)
        for j, rel in enumerate(rels):
            p = Path(rel)
            if not p.is_absolute():
                p = image_root / rel
            with Image.open(p) as im:
                im = im.convert("L").resize((resize, resize), Image.BILINEAR)
                buf[j] = np.asarray(im, dtype=np.float64) / 255.0
        yield start, buf[:k]


def _load_paths_labels(split_csv: Path):
    df = pd.read_csv(split_csv)
    return df["path"].tolist(), df["class"].to_numpy()


def fit_2d2dpca_chunked(paths, image_root, resize, d, q, chunk):
    """Replicates fit_2d2dpca but accumulates Gt_col, Gt_row in chunks.

    Gt_col = (1/M) Σ (A-Ā)^T (A-Ā)   (n×n)  -> top-d eigvecs X (n×d)
    Gt_row = (1/M) Σ (A-Ā)(A-Ā)^T    (m×m)  -> top-q eigvecs Z (m×q)
    """
    m = n = resize
    M = len(paths)
    # Pass 1: mean image.
    mean_image = np.zeros((m, n), dtype=np.float64)
    for _, arr in _iter_image_chunks(paths, image_root, resize, chunk):
        mean_image += arr.sum(axis=0)
    mean_image /= M
    # Pass 2: covariances.
    gt_col = np.zeros((n, n), dtype=np.float64)
    gt_row = np.zeros((m, m), dtype=np.float64)
    for _, arr in _iter_image_chunks(paths, image_root, resize, chunk):
        c = arr - mean_image  # (k, m, n)
        gt_col += np.einsum("kij,kil->jl", c, c)
        gt_row += np.einsum("kij,klj->il", c, c)
    gt_col /= M
    gt_row /= M
    gt_col = (gt_col + gt_col.T) / 2.0
    gt_row = (gt_row + gt_row.T) / 2.0

    def _top(cov, k):
        w, v = np.linalg.eigh(cov)
        w = np.clip(w, 0.0, None)
        order = np.argsort(w)[::-1]
        return v[:, order][:, :k]

    X = _top(gt_col, d)  # (n, d)
    Z = _top(gt_row, q)  # (m, q)
    return mean_image, X, Z


def project_all_chunked(paths, image_root, resize, X, Z, chunk):
    """C_k = Z^T A_k X -> (q, d); return flattened features (N, q*d)."""
    q = Z.shape[1]
    d = X.shape[1]
    feats = np.empty((len(paths), q * d), dtype=np.float64)
    for start, arr in _iter_image_chunks(paths, image_root, resize, chunk):
        zt_a = np.einsum("pi,kij->kpj", Z.T, arr)  # (k,q,n)
        c = np.einsum("kpj,jd->kpd", zt_a, X)  # (k,q,d)
        feats[start : start + arr.shape[0]] = c.reshape(arr.shape[0], -1)
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", default="results/twodpca_vocalmat/split")
    ap.add_argument("--image-root", default=str(_REPO_ROOT))
    ap.add_argument("--resize", type=int, default=227)
    ap.add_argument("--d", type=int, default=9, help="column components (default 9 = energy-0.95 selection)")
    ap.add_argument("--q", type=int, default=10, help="row components (default 10 = energy-0.95 selection)")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--out", default="results/twodpca_verify/ablations/resize_227_bestconfig.json")
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import f1_score

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = _REPO_ROOT / split_dir
    image_root = Path(args.image_root)
    labels_order = list(GRIMSLEY_12_CLASSES)

    tr_paths, tr_lbls = _load_paths_labels(split_dir / "train.csv")
    va_paths, va_lbls = _load_paths_labels(split_dir / "val.csv")
    te_paths, te_lbls = _load_paths_labels(split_dir / "test.csv")

    print("=" * 72)
    print("Memory-safe 2d2dpca/lda BEST-config run (chunked covariance)")
    print("=" * 72)
    print(f"split_dir   : {split_dir}")
    print(f"resize      : {args.resize}x{args.resize} (bilinear, grayscale, /255)")
    print(f"d (col)     : {args.d}")
    print(f"q (row)     : {args.q}")
    print(f"chunk       : {args.chunk} images/chunk (peak RAM control)")
    print(f"variant     : 2d2dpca  classifier: lda  (replicates TwoDPCAClassifier best config)")
    print(f"scoring     : macro-F1 over {labels_order}")
    print(f"rows        : train={len(tr_paths)} val={len(va_paths)} test={len(te_paths)}")
    print("-" * 72)

    t0 = time.perf_counter()
    mean_image, X, Z = fit_2d2dpca_chunked(
        tr_paths, image_root, args.resize, args.d, args.q, args.chunk
    )
    tr_feat = project_all_chunked(tr_paths, image_root, args.resize, X, Z, args.chunk)
    va_feat = project_all_chunked(va_paths, image_root, args.resize, X, Z, args.chunk)
    te_feat = project_all_chunked(te_paths, image_root, args.resize, X, Z, args.chunk)

    scaler = StandardScaler().fit(tr_feat)
    lda = LinearDiscriminantAnalysis().fit(scaler.transform(tr_feat), tr_lbls)
    va_pred = lda.predict(scaler.transform(va_feat))
    te_pred = lda.predict(scaler.transform(te_feat))
    fit_seconds = time.perf_counter() - t0

    va_f1 = float(f1_score(va_lbls, va_pred, labels=labels_order, average="macro", zero_division=0))
    te_f1 = float(f1_score(te_lbls, te_pred, labels=labels_order, average="macro", zero_division=0))

    print(f"feature_dim : {tr_feat.shape[1]} (q*d = {args.q}*{args.d})")
    print(f"val macro-F1 : {va_f1:.4f}")
    print(f"test macro-F1: {te_f1:.4f}")
    print(f"elapsed      : {fit_seconds:.1f}s")
    print(f"ResNet-18    : 0.7669   |   2DPCA@64 best: 0.2402")

    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "variant": "2d2dpca", "classifier": "lda",
        "resize": args.resize, "d": args.d, "q": args.q,
        "feature_dim": int(tr_feat.shape[1]),
        "macro_f1_val": va_f1, "macro_f1_test": te_f1,
        "elapsed_s": fit_seconds,
        "resnet18_baseline": 0.7669,
        "twodpca_64_best": 0.2402,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
