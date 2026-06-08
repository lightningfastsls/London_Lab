#!/usr/bin/env python
"""Experiment driver: 2DPCA / (2D)^2PCA classifier on the VocalMat 12-class corpus.

Runs the full config matrix against the recording-grouped 80/10/10 split and
compares macro-F1 to the production ResNet-18 v1 baseline (test macro-F1 0.7669,
``results/lab_classifier_v1/metrics.json``).

Algorithm: Yang et al. (2004) 2DPCA + Zhang & Zhou (2005) (2D)^2PCA, implemented
in ``src/usv_spectrogram/classifier/twodpca.py`` (pure NumPy; sklearn only in the
SVM/LDA classifier paths). This script is plumbing only — it loads images, calls
the frozen API, scores, and writes artifacts.

Config matrix (all four run in one invocation):
    1) variant=2dpca   classifier=nn
    2) variant=2dpca   classifier=svm
    3) variant=2d2dpca classifier=svm
    4) variant=2d2dpca classifier=lda

Usage (smoke test):
    .venv/bin/python scripts/experiments/train_2dpca_classifier.py \
        --limit-per-class 40 --resize 32 --out .claude/jobs/.../smoke_out

Per repo convention (feedback_analysis_print_params): every parameter, threshold,
sort key, and filter row count is printed at the top of the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Make src/ importable without installation.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier.dataset import (  # noqa: E402
    GRIMSLEY_12_CLASSES,
    build_stratified_split,
)
from usv_spectrogram.classifier.twodpca import TwoDPCAClassifier  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# manifest "class" string -> display name used everywhere downstream.
CLASS_TO_DISPLAY: dict[str, str] = {
    "noise": "Noise",
    "step_up": "Step up",
    "down_fm": "Down-FM",
    "short": "Short",
    "chevron": "Chevron",
    "up_fm": "Up-FM",
    "flat": "Flat",
    "two_steps": "Two steps",
    "step_down": "Step down",
    "complex": "Complex",
    "rev_chevron": "Reverse Chevron",
    "mult_steps": "Multi-steps",
}

# Order is the frozen 12-class display order (matches dataset.GRIMSLEY_12_CLASSES
# and the per_class_* lists in results/lab_classifier_v1/metrics.json).
DISPLAY_ORDER: tuple[str, ...] = GRIMSLEY_12_CLASSES

RESNET18_BASELINE_TEST_MACRO_F1 = 0.7669

# (variant, classifier) tuples in run order.
CONFIG_MATRIX: tuple[tuple[str, str], ...] = (
    ("2dpca", "nn"),
    ("2dpca", "svm"),
    ("2d2dpca", "svm"),
    ("2d2dpca", "lda"),
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_manifest(manifest_path: Path, limit_per_class: int) -> pd.DataFrame:
    """Load manifest, map class -> display name, add dummy duration_ms.

    ``build_stratified_split`` requires columns path, class, source_recording,
    duration_ms; the VocalMat manifest lacks duration_ms, so we inject a
    constant 1.0 (the split allocator is purely count-based, never reads it).
    """
    df = pd.read_csv(manifest_path)
    missing = {"path", "class", "source_recording"} - set(df.columns)
    if missing:
        raise KeyError(f"manifest missing columns: {sorted(missing)}")

    unknown = set(df["class"].unique()) - set(CLASS_TO_DISPLAY)
    if unknown:
        raise ValueError(f"manifest has unmapped class strings: {sorted(unknown)}")

    df = df.copy()
    df["class"] = df["class"].map(CLASS_TO_DISPLAY)
    df["duration_ms"] = 1.0

    if limit_per_class > 0:
        # Deterministic head() per class — keeps recording grouping intact-ish
        # for smoke tests (we are not making scientific claims at the limit).
        df = (
            df.groupby("class", group_keys=False, sort=False)
            .head(limit_per_class)
            .reset_index(drop=True)
        )
    return df


def load_split_images(
    split_csv: Path, image_root: Path, resize: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load a split CSV's PNGs -> (N, resize, resize) float32 in [0,1] + labels.

    Each PNG is opened with PIL, converted to grayscale ('L'), bilinear-resized
    to (resize, resize), and divided by 255.
    """
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


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, list[float], list[float], list[list[int]]]:
    """Macro-F1, per-class precision/recall, confusion matrix in DISPLAY_ORDER."""
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    labels = list(DISPLAY_ORDER)
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return (
        macro_f1,
        [float(x) for x in prec],
        [float(x) for x in rec],
        [[int(x) for x in r] for r in cm],
    )


def feature_dim(clf: TwoDPCAClassifier) -> int:
    """Flattened feature dimension p*d (the SVM/LDA vector length)."""
    model = clf.model_
    if clf.variant == "2dpca":
        m = model.mean_image.shape[0]
        return int(m * model.n_components)
    return int(model.n_components_row * model.n_components)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def write_confusion_png(cm: list[list[int]], title: str, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(DISPLAY_ORDER)))
    ax.set_yticks(range(len(DISPLAY_ORDER)))
    ax.set_xticklabels(DISPLAY_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(DISPLAY_ORDER, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 0.5
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                int(cm_arr[i, j]),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_eval_report(
    out_dir: Path, results: list[dict], best: dict
) -> None:
    lines: list[str] = []
    lines.append("# 2DPCA / (2D)^2PCA vs ResNet-18 on VocalMat (12-class)\n")
    lines.append(
        f"ResNet-18 v1 baseline test macro-F1: **{RESNET18_BASELINE_TEST_MACRO_F1:.4f}** "
        "(`results/lab_classifier_v1/metrics.json`)\n"
    )
    lines.append("## Config comparison\n")
    lines.append("| Variant | Classifier | n_components | feature_dim | val macro-F1 | test macro-F1 | vs ResNet-18 | fit (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        delta = r["macro_f1_test"] - RESNET18_BASELINE_TEST_MACRO_F1
        mark = " **(best)**" if r is best else ""
        ncomp = (
            f"{r['n_components']}"
            if r["variant"] == "2dpca"
            else f"{r['n_components_row']}x{r['n_components']}"
        )
        lines.append(
            f"| {r['variant']}{mark} | {r['classifier']} | {ncomp} | {r['feature_dim']} | "
            f"{r['macro_f1_val']:.4f} | {r['macro_f1_test']:.4f} | {delta:+.4f} | "
            f"{r['fit_seconds']:.2f} |"
        )
    lines.append("")
    lines.append(
        f"## Best config: {best['variant']} / {best['classifier']} "
        f"(test macro-F1 {best['macro_f1_test']:.4f})\n"
    )
    lines.append("| Class | Precision | Recall |")
    lines.append("|---|---|---|")
    for cls, p, rcl in zip(
        DISPLAY_ORDER, best["per_class_precision"], best["per_class_recall"]
    ):
        lines.append(f"| {cls} | {p:.3f} | {rcl:.3f} |")
    lines.append("")
    (out_dir / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/vocalmat_full/manifest.csv")
    ap.add_argument(
        "--image-root",
        default=str(_REPO_ROOT),
        help="prepended to manifest 'path' when not absolute",
    )
    ap.add_argument("--out", default="results/twodpca_vocalmat/")
    ap.add_argument("--resize", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="0 = all rows; >0 keeps the first N rows per class (smoke tests)",
    )
    ap.add_argument("--energy", type=float, default=0.95)
    ap.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="explicit column components d (overrides --energy if set); "
        "default None = energy-based selection (backward-compatible)",
    )
    ap.add_argument(
        "--n-components-row",
        type=int,
        default=None,
        help="explicit row components q for the 2d2dpca variant (overrides "
        "--energy for the row direction); default None = energy-based",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = _REPO_ROOT / manifest_path
    image_root = Path(args.image_root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / "split"

    # ---- Load + split ---- #
    manifest = load_manifest(manifest_path, args.limit_per_class)
    split = build_stratified_split(manifest, seed=args.seed, out_dir=split_dir)

    train_imgs, train_lbls = load_split_images(split.train_csv, image_root, args.resize)
    val_imgs, val_lbls = load_split_images(split.val_csv, image_root, args.resize)
    test_imgs, test_lbls = load_split_images(split.test_csv, image_root, args.resize)

    train_counts = class_counts(train_lbls)
    val_counts = class_counts(val_lbls)
    test_counts = class_counts(test_lbls)

    # ---- MANDATORY parameter print (feedback_analysis_print_params) ---- #
    print("=" * 72)
    print("2DPCA / (2D)^2PCA experiment driver")
    print("=" * 72)
    print(f"manifest          : {manifest_path}")
    print(f"image_root        : {image_root}")
    print(f"out_dir           : {out_dir}")
    print(f"resize            : {args.resize}x{args.resize} (bilinear, grayscale, /255)")
    print(f"seed              : {args.seed}")
    print(f"energy threshold  : {args.energy}")
    print(f"n_components       : {args.n_components} (None = energy-based)")
    print(f"n_components_row   : {args.n_components_row} (None = energy-based)")
    print(f"limit_per_class   : {args.limit_per_class} (0 = all)")
    print(f"split             : recording-grouped 80/10/10 (build_stratified_split)")
    print(f"classes (order)   : {list(DISPLAY_ORDER)}")
    print(f"distance (nn)     : Yang-2004 feature-matrix distance (sum of column L2)")
    print(f"resnet18 baseline : test macro-F1 = {RESNET18_BASELINE_TEST_MACRO_F1}")
    print("-" * 72)
    print(f"row counts        : train={len(train_lbls)} val={len(val_lbls)} test={len(test_lbls)}")
    print(f"{'class':<18}{'train':>8}{'val':>8}{'test':>8}")
    for cls in DISPLAY_ORDER:
        print(f"{cls:<18}{train_counts[cls]:>8}{val_counts[cls]:>8}{test_counts[cls]:>8}")
    print("-" * 72)

    # ---- Config matrix ---- #
    results: list[dict] = []
    for variant, classifier in CONFIG_MATRIX:
        clf = TwoDPCAClassifier(
            variant=variant,
            classifier=classifier,
            energy=args.energy,
            n_components=args.n_components,
            n_components_row=args.n_components_row,
        )
        t0 = time.perf_counter()
        clf.fit(train_imgs, train_lbls)
        fit_seconds = time.perf_counter() - t0

        val_pred = clf.predict(val_imgs)
        test_pred = clf.predict(test_imgs)

        macro_f1_val, _, _, _ = score_predictions(val_lbls, val_pred)
        macro_f1_test, prec, rec, cm = score_predictions(test_lbls, test_pred)
        fdim = feature_dim(clf)
        ncomp = clf.model_.n_components
        ncomp_row = (
            clf.model_.n_components_row if variant == "2d2dpca" else None
        )

        comp_str = (
            f"{ncomp}" if variant == "2dpca" else f"{ncomp_row}x{ncomp}"
        )
        print(
            f"[{variant:<8} {classifier:<3}] n_components={comp_str:<10} "
            f"feature_dim={fdim:<7} val_F1={macro_f1_val:.4f} "
            f"test_F1={macro_f1_test:.4f} fit={fit_seconds:.2f}s"
        )

        rec_dict = {
            "variant": variant,
            "classifier": classifier,
            "n_components": int(ncomp),
            "n_components_row": int(ncomp_row) if ncomp_row is not None else None,
            "feature_dim": fdim,
            "fit_seconds": float(fit_seconds),
            "macro_f1_val": macro_f1_val,
            "macro_f1_test": macro_f1_test,
            "per_class_precision": prec,
            "per_class_recall": rec,
            "confusion_matrix": cm,
        }
        results.append(rec_dict)

        # Per-config metrics JSON (same keys as lab_classifier_v1/metrics.json).
        metrics = {
            "macro_f1_val": macro_f1_val,
            "macro_f1_test": macro_f1_test,
            "per_class_precision": prec,
            "per_class_recall": rec,
            "confusion_matrix": cm,
            "variant": variant,
            "classifier": classifier,
            "n_components": int(ncomp),
            "feature_dim": fdim,
            "fit_seconds": float(fit_seconds),
        }
        (out_dir / f"metrics_{variant}_{classifier}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        write_confusion_png(
            cm,
            f"{variant} / {classifier}  (test macro-F1 {macro_f1_test:.4f})",
            out_dir / f"confusion_matrix_{variant}_{classifier}.png",
        )

    # ---- Summary + report ---- #
    best = max(results, key=lambda r: r["macro_f1_test"])
    summary = {
        "resnet18_baseline_test_macro_f1": RESNET18_BASELINE_TEST_MACRO_F1,
        "resize": args.resize,
        "seed": args.seed,
        "energy": args.energy,
        "limit_per_class": args.limit_per_class,
        "best_config": {
            "variant": best["variant"],
            "classifier": best["classifier"],
            "macro_f1_test": best["macro_f1_test"],
        },
        "configs": [
            {
                "variant": r["variant"],
                "classifier": r["classifier"],
                "n_components": r["n_components"],
                "n_components_row": r["n_components_row"],
                "feature_dim": r["feature_dim"],
                "macro_f1_val": r["macro_f1_val"],
                "macro_f1_test": r["macro_f1_test"],
                "fit_seconds": r["fit_seconds"],
            }
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_eval_report(out_dir, results, best)

    print("-" * 72)
    print(
        f"BEST: {best['variant']}/{best['classifier']} "
        f"test macro-F1={best['macro_f1_test']:.4f} "
        f"(ResNet-18 = {RESNET18_BASELINE_TEST_MACRO_F1})"
    )
    print(f"Artifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
