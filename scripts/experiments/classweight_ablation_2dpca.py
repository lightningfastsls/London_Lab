#!/usr/bin/env python
"""Class-weight ablation for the (2D)^2PCA best config (energy=0.95, resize 64).

Adversarial-verification question: is the ~0.24 macro-F1 of the best 2DPCA head
suppressed by class imbalance handling? The driver's SVM/LDA heads use NO class
weighting. This script reuses the FROZEN public API of
``usv_spectrogram.classifier.twodpca`` (fit_2d2dpca + project_bilateral) to build
the exact projected feature matrices, then fits FOUR heads on the
flattened + StandardScaler'd features and compares test macro-F1:

    (a) LinearSVC()                          -- unweighted baseline (matches driver)
    (b) LinearSVC(class_weight="balanced")   -- balanced
    (c) LinearDiscriminantAnalysis()         -- empirical priors (matches driver)
    (d) LinearDiscriminantAnalysis(priors=[1/12]*12)  -- uniform priors

It does NOT modify twodpca.py / dataset.py / the driver. It loads the SAME frozen
split (results/twodpca_vocalmat/split/, resize 64) with the SAME PIL convention as
train_2dpca_classifier.load_split_images (grayscale 'L', bilinear, /255).

Per repo convention (feedback_analysis_print_params): all parameters printed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier.twodpca import (  # noqa: E402
    fit_2d2dpca,
    project_bilateral,
)

# Display order matching the driver (DISPLAY_ORDER / Grimsley 12). Pulled from
# the actual split labels to avoid hardcoding; sorted for determinism but we
# explicitly use the canonical 12-class list from dataset for scoring labels.
from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402

RESIZE = 64
ENERGY = 0.95
SPLIT_DIR = _REPO_ROOT / "results/twodpca_vocalmat/split"
OUT_PATH = _REPO_ROOT / "results/twodpca_verify/ablations/classweight_ablation.json"


def load_split_images(split_csv: Path, resize: int):
    """Replica of train_2dpca_classifier.load_split_images (same PIL convention)."""
    from PIL import Image

    df = pd.read_csv(split_csv)
    images = np.empty((len(df), resize, resize), dtype=np.float32)
    for i, rel in enumerate(df["path"].tolist()):
        img_path = Path(rel)
        if not img_path.is_absolute():
            img_path = _REPO_ROOT / rel
        with Image.open(img_path) as im:
            im = im.convert("L").resize((resize, resize), Image.BILINEAR)
            images[i] = np.asarray(im, dtype=np.float32) / 255.0
    labels = df["class"].to_numpy()
    return images, labels


def score(y_true, y_pred, labels):
    from sklearn.metrics import f1_score, recall_score

    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return macro, {l: float(f) for l, f in zip(labels, per_f1)}, {
        l: float(r) for l, r in zip(labels, per_rec)
    }


def main() -> int:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    train_imgs, train_lbls = load_split_images(SPLIT_DIR / "train.csv", RESIZE)
    test_imgs, test_lbls = load_split_images(SPLIT_DIR / "test.csv", RESIZE)

    # Canonical scoring label order = the 12 Grimsley classes that appear.
    labels = list(GRIMSLEY_12_CLASSES)
    # Keep only labels actually present (defensive), preserving canonical order.
    present = set(np.unique(np.concatenate([train_lbls, test_lbls])).tolist())
    labels = [l for l in labels if l in present]
    n_classes = len(labels)

    test_vals, test_cnts = np.unique(test_lbls, return_counts=True)
    test_count_map = dict(zip(test_vals.tolist(), test_cnts.tolist()))

    print("=" * 72)
    print("Class-weight ablation: (2D)^2PCA best config")
    print("=" * 72)
    print(f"split_dir       : {SPLIT_DIR}")
    print(f"resize          : {RESIZE}x{RESIZE} (grayscale 'L', bilinear, /255)")
    print(f"variant         : 2d2dpca")
    print(f"energy          : {ENERGY} (n_components/row energy-based)")
    print(f"train rows      : {len(train_lbls)}   test rows: {len(test_lbls)}")
    print(f"n_classes       : {n_classes}")
    print(f"scoring labels  : {labels}")
    print(f"test counts     : {[(l, test_count_map.get(l, 0)) for l in labels]}")
    print(f"heads           : LinearSVC (unw), LinearSVC(balanced), "
          f"LDA (empirical priors), LDA (uniform priors=[1/{n_classes}]*{n_classes})")
    print("-" * 72)

    # ---- Fit (2D)^2PCA on TRAIN only, energy=0.95 (frozen API) ---- #
    model = fit_2d2dpca(train_imgs.astype(np.float64), energy=ENERGY)
    d = model.n_components
    q = model.n_components_row
    print(f"chosen n_components (col d) : {d}")
    print(f"chosen n_components_row (q) : {q}")
    print(f"feature_dim (q*d)           : {q * d}")
    print(f"col energy retained         : {model.energy_ratio_col:.4f}")
    print(f"row energy retained         : {model.energy_ratio_row:.4f}")
    print("-" * 72)

    # ---- Project both splits -> (N, q, d) -> flatten -> StandardScaler ---- #
    def project_flat(imgs):
        feats = np.stack([project_bilateral(a.astype(np.float64), model) for a in imgs])
        return feats.reshape(feats.shape[0], -1)

    train_flat = project_flat(train_imgs)
    test_flat = project_flat(test_imgs)

    scaler = StandardScaler()
    train_std = scaler.fit_transform(train_flat)
    test_std = scaler.transform(test_flat)

    uniform_priors = np.full(n_classes, 1.0 / n_classes)

    heads = {
        "svm_unweighted": LinearSVC(),
        "svm_balanced": LinearSVC(class_weight="balanced"),
        "lda_empirical": LinearDiscriminantAnalysis(),
        "lda_uniform_priors": LinearDiscriminantAnalysis(priors=uniform_priors),
    }

    results = {}
    for name, est in heads.items():
        # LDA needs y aligned to its own class order for priors; sklearn LDA
        # maps priors to sorted classes_. To make uniform priors unambiguous we
        # rely on uniform == identical for every class regardless of order.
        est.fit(train_std, train_lbls)
        pred = est.predict(test_std)
        macro, per_f1, per_rec = score(test_lbls, pred, labels)
        results[name] = {
            "test_macro_f1": macro,
            "per_class_f1": per_f1,
            "per_class_recall": per_rec,
        }
        print(f"[{name:<20}] test macro-F1 = {macro:.4f}")

    # Focus comparison on Multi-steps (tiny class flagged in the brief).
    def get(name, cls, field):
        return results[name][field].get(cls, None)

    print("-" * 72)
    ms = "Multi-steps"
    if ms in labels:
        print(f"Multi-steps (test n={test_count_map.get(ms, 0)}) F1 / recall:")
        for name in heads:
            print(f"  {name:<20} F1={get(name, ms, 'per_class_f1'):.4f}  "
                  f"recall={get(name, ms, 'per_class_recall'):.4f}")

    summary = {
        "config": {
            "variant": "2d2dpca",
            "energy": ENERGY,
            "resize": RESIZE,
            "n_components_col_d": int(d),
            "n_components_row_q": int(q),
            "feature_dim": int(q * d),
            "split_dir": str(SPLIT_DIR),
            "n_classes": n_classes,
            "scoring_labels": labels,
            "test_counts": {l: test_count_map.get(l, 0) for l in labels},
        },
        "driver_unweighted_baseline_reference": 0.2402,
        "resnet18_baseline_test_macro_f1": 0.7669,
        "heads": results,
        "headline": {
            "svm_unweighted": results["svm_unweighted"]["test_macro_f1"],
            "svm_balanced": results["svm_balanced"]["test_macro_f1"],
            "lda_empirical": results["lda_empirical"]["test_macro_f1"],
            "lda_uniform_priors": results["lda_uniform_priors"]["test_macro_f1"],
            "multisteps_f1": {
                name: get(name, ms, "per_class_f1") if ms in labels else None
                for name in heads
            },
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print("-" * 72)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
