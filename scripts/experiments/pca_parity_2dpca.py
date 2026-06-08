#!/usr/bin/env python
"""PARITY CONTROL: ordinary flattened sklearn PCA + linear heads vs 2DPCA.

Adversarial verification of the 2DPCA classifier (test macro-F1 ~0.24 on the
VocalMat 12-class corpus, vs ResNet-18's 0.767). Question: does ORDINARY
flattened sklearn PCA feeding the SAME kind of downstream linear classifier
(LDA / LinearSVC) ALSO land at ~0.24 on the IDENTICAL frozen split? If yes,
the ceiling is the data + linear-model class, not the 2DPCA implementation.

This script reuses the EXISTING frozen split (does not re-split):
    results/twodpca_vocalmat/split/{train,val,test}.csv

Image loading matches scripts/experiments/train_2dpca_classifier.py
:load_split_images EXACTLY at --resize 64: PIL open -> convert("L") ->
resize((64,64), BILINEAR) -> /255.0 -> float32. Then flatten to a 4096-vector.

Heads (both, per k):
    (a) LinearDiscriminantAnalysis
    (b) StandardScaler -> LinearSVC   (matches the driver's SVM path wrapping)

Scoring: sklearn.metrics.f1_score(labels=GRIMSLEY_12_CLASSES, average="macro",
zero_division=0) on the test split (val reported too).

Per repo convention (feedback_analysis_print_params) every parameter, threshold,
and per-split / per-class row count is printed at the top.
"""
from __future__ import annotations

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

# --------------------------------------------------------------------------- #
# Constants / parameters
# --------------------------------------------------------------------------- #
RESIZE = 64
K_SWEEP = (5, 9, 20, 40, 80, 160)
SEED = 1729
SPLIT_DIR = _REPO_ROOT / "results" / "twodpca_vocalmat" / "split"
IMAGE_ROOT = _REPO_ROOT
OUT_DIR = _REPO_ROOT / "results" / "twodpca_verify"
TWODPCA_BASELINE_TEST_MACRO_F1 = 0.24
RESNET18_BASELINE_TEST_MACRO_F1 = 0.7669
DISPLAY_ORDER = list(GRIMSLEY_12_CLASSES)


def load_split_images(split_csv: Path, image_root: Path, resize: int):
    """Match train_2dpca_classifier.load_split_images conventions exactly."""
    from PIL import Image

    df = pd.read_csv(split_csv)
    images = np.empty((len(df), resize, resize), dtype=np.float32)
    for i, rel in enumerate(df["path"].tolist()):
        img_path = Path(rel)
        if not img_path.is_absolute():
            img_path = image_root / rel
        with Image.open(img_path) as im:
            im = im.convert("L").resize((resize, resize), Image.BILINEAR)
            images[i] = np.asarray(im, dtype=np.float32) / 255.0
    labels = df["class"].to_numpy()
    return images, labels


def class_counts(labels: np.ndarray) -> dict[str, int]:
    vals, cnts = np.unique(labels, return_counts=True)
    counts = dict(zip(vals.tolist(), cnts.tolist()))
    return {cls: int(counts.get(cls, 0)) for cls in DISPLAY_ORDER}


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            y_true, y_pred, labels=DISPLAY_ORDER, average="macro", zero_division=0
        )
    )


def main() -> int:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_imgs, train_lbls = load_split_images(SPLIT_DIR / "train.csv", IMAGE_ROOT, RESIZE)
    val_imgs, val_lbls = load_split_images(SPLIT_DIR / "val.csv", IMAGE_ROOT, RESIZE)
    test_imgs, test_lbls = load_split_images(SPLIT_DIR / "test.csv", IMAGE_ROOT, RESIZE)

    n_feat = RESIZE * RESIZE
    Xtr = train_imgs.reshape(len(train_imgs), n_feat)
    Xva = val_imgs.reshape(len(val_imgs), n_feat)
    Xte = test_imgs.reshape(len(test_imgs), n_feat)

    train_counts = class_counts(train_lbls)
    val_counts = class_counts(val_lbls)
    test_counts = class_counts(test_lbls)

    # ---- MANDATORY parameter print ---- #
    print("=" * 72)
    print("PCA PARITY CONTROL (flattened sklearn PCA + linear heads) vs 2DPCA")
    print("=" * 72)
    print(f"split_dir         : {SPLIT_DIR}")
    print(f"image_root        : {IMAGE_ROOT}")
    print(f"out_dir           : {OUT_DIR}")
    print(f"resize            : {RESIZE}x{RESIZE} (bilinear, grayscale, /255) -> flatten {n_feat}")
    print(f"k sweep           : {list(K_SWEEP)}")
    print(f"seed              : {SEED}")
    print(f"heads             : LinearDiscriminantAnalysis ; StandardScaler->LinearSVC")
    print(f"scoring           : f1_score macro, labels=GRIMSLEY_12_CLASSES, zero_division=0")
    print(f"2dpca baseline    : test macro-F1 = {TWODPCA_BASELINE_TEST_MACRO_F1}")
    print(f"resnet18 baseline : test macro-F1 = {RESNET18_BASELINE_TEST_MACRO_F1}")
    print(f"classes (order)   : {DISPLAY_ORDER}")
    print("-" * 72)
    print(f"row counts        : train={len(train_lbls)} val={len(val_lbls)} test={len(test_lbls)}")
    print(f"{'class':<18}{'train':>8}{'val':>8}{'test':>8}")
    for cls in DISPLAY_ORDER:
        print(f"{cls:<18}{train_counts[cls]:>8}{val_counts[cls]:>8}{test_counts[cls]:>8}")
    print("-" * 72)

    results: list[dict] = []
    for k in K_SWEEP:
        if k >= min(len(Xtr), n_feat):
            print(f"[k={k}] SKIP (k >= min(n_train, n_feat))")
            continue
        t0 = time.perf_counter()
        pca = PCA(n_components=k, random_state=SEED)
        Ztr = pca.fit_transform(Xtr)
        Zva = pca.transform(Xva)
        Zte = pca.transform(Xte)
        pca_seconds = time.perf_counter() - t0
        evr = float(pca.explained_variance_ratio_.sum())

        # (a) LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(Ztr, train_lbls)
        lda_val = macro_f1(val_lbls, lda.predict(Zva))
        lda_test = macro_f1(test_lbls, lda.predict(Zte))

        # (b) StandardScaler -> LinearSVC
        svm = make_pipeline(
            StandardScaler(), LinearSVC(random_state=SEED, max_iter=20000)
        )
        svm.fit(Ztr, train_lbls)
        svm_val = macro_f1(val_lbls, svm.predict(Zva))
        svm_test = macro_f1(test_lbls, svm.predict(Zte))

        print(
            f"[k={k:>3}] EVR={evr:.3f}  pca={pca_seconds:5.2f}s  "
            f"LDA val={lda_val:.4f} test={lda_test:.4f}  |  "
            f"SVM val={svm_val:.4f} test={svm_test:.4f}"
        )
        results.append(
            {
                "k": int(k),
                "explained_variance_ratio_sum": evr,
                "pca_seconds": float(pca_seconds),
                "lda_macro_f1_val": lda_val,
                "lda_macro_f1_test": lda_test,
                "svm_macro_f1_val": svm_val,
                "svm_macro_f1_test": svm_test,
            }
        )

    # ---- Best config ---- #
    candidates = []
    for r in results:
        candidates.append((r["lda_macro_f1_test"], r["k"], "LDA"))
        candidates.append((r["svm_macro_f1_test"], r["k"], "SVM"))
    best_f1, best_k, best_head = max(candidates, key=lambda c: c[0])

    delta_vs_2dpca = best_f1 - TWODPCA_BASELINE_TEST_MACRO_F1
    if best_f1 > 0.45:
        verdict = "PCA MUCH HIGHER (2DPCA leaving signal)"
    else:
        verdict = "PARITY (ceiling confirmed)"

    print("-" * 72)
    print(
        f"BEST PCA: k={best_k} head={best_head} test macro-F1={best_f1:.4f} "
        f"(2DPCA={TWODPCA_BASELINE_TEST_MACRO_F1}, delta={delta_vs_2dpca:+.4f})"
    )
    print(f"VERDICT: {verdict}")
    print("-" * 72)

    out = {
        "parameters": {
            "resize": RESIZE,
            "flatten_dim": n_feat,
            "k_sweep": list(K_SWEEP),
            "seed": SEED,
            "heads": ["LinearDiscriminantAnalysis", "StandardScaler->LinearSVC"],
            "scoring": "f1_score macro, labels=GRIMSLEY_12_CLASSES, zero_division=0",
            "split_dir": str(SPLIT_DIR),
            "image_root": str(IMAGE_ROOT),
        },
        "baselines": {
            "twodpca_test_macro_f1": TWODPCA_BASELINE_TEST_MACRO_F1,
            "resnet18_test_macro_f1": RESNET18_BASELINE_TEST_MACRO_F1,
        },
        "row_counts": {
            "train": int(len(train_lbls)),
            "val": int(len(val_lbls)),
            "test": int(len(test_lbls)),
        },
        "per_class_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "sweep": results,
        "best": {
            "k": int(best_k),
            "head": best_head,
            "test_macro_f1": float(best_f1),
            "delta_vs_2dpca": float(delta_vs_2dpca),
        },
        "verdict": verdict,
    }
    out_json = OUT_DIR / "pca_parity.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
