#!/usr/bin/env python
"""Adversarial leakage / chance-floor controls for the 2DPCA USV classifier.

Context: the 2DPCA math is CONFIRMED correct (reproduces the Olivetti benchmark
and a non-square orientation test passes). On the 12-class VocalMat USV corpus
the best config (variant=2d2dpca, classifier=lda, energy=0.95) scores test
macro-F1 ~0.24, far below ResNet-18's 0.767. This script asks: is that 0.24 a
real (if weak) signal at a genuine ceiling, or an artefact of leakage /
label-alignment bugs?

Three controls (all read the FROZEN split as-is; nothing is re-split):

  TASK 1 — SHUFFLE CONTROL (chance floor)
      Fit the best config on TRAIN with the train labels permuted by
      np.random.default_rng(0).permutation, predict TEST (true labels), score
      macro-F1. A correctly-wired pipeline must collapse to ~1/12 = 0.083.
      A shuffled model that still scores well (>0.15) implies leakage or a
      label-alignment bug. The UNSHUFFLED best config is scored on the same
      loaded arrays to confirm real >> shuffled.

  TASK 2 — SPLIT INTEGRITY
      (a) source_recording overlap sizes across train/val/test. These are
          EXPECTED to be non-zero because build_stratified_split allocates
          recordings to splits INDEPENDENTLY PER CLASS (see dataset.py
          ~line 260: it loops over each class and splits that class's
          recordings via _allocate_class). The same recording can land in
          train for one class and test for another, so the GLOBAL
          source_recording sets are non-disjoint. This is a form of
          cage-acoustic leakage that can only INFLATE a score — so a
          still-low 0.24 makes the "genuine ceiling" verdict STRONGER.
      (b) label->image alignment spot-check on 10 random test rows: the class
          folder embedded in the PNG path (data/vocalmat_full/<folder>/...)
          must map through CLASS_TO_DISPLAY to the same display value in the
          "class" column, AND the array the loader returns for that row must
          be the one decoded from that exact PNG.

  TASK 3 — write results/twodpca_verify/leakage_controls.json.

Per repo convention (feedback_analysis_print_params): every parameter,
threshold, and row count is printed at the top of the run.

This is a standalone verification script. It does NOT modify twodpca.py, the
driver, dataset.py, or any test file.
"""
from __future__ import annotations

import json
import sys
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
# Repo root on path so `scripts.experiments.train_2dpca_classifier` imports
# regardless of cwd (the package has no __init__.py; this is the documented
# pattern for running these standalone scripts).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402
from usv_spectrogram.classifier.twodpca import TwoDPCAClassifier  # noqa: E402

# Reuse the EXACT same loader and class map as the production driver so the
# image-loading convention (PIL -> L -> resize BILINEAR -> /255 float32) and
# the manifest-class -> display-name mapping are identical, not re-derived.
from scripts.experiments.train_2dpca_classifier import (  # noqa: E402
    CLASS_TO_DISPLAY,
    load_split_images,
)

# --------------------------------------------------------------------------- #
# Parameters (frozen)
# --------------------------------------------------------------------------- #
SPLIT_DIR = _REPO_ROOT / "results" / "twodpca_vocalmat" / "split"
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"
IMAGE_ROOT = _REPO_ROOT

RESIZE = 64
BEST_VARIANT = "2d2dpca"
BEST_CLASSIFIER = "lda"
BEST_ENERGY = 0.95
SHUFFLE_SEED = 0
SPOTCHECK_SEED = 0
N_SPOTCHECK = 10
CHANCE_FLOOR = 1.0 / 12.0  # 0.0833...
LEAKAGE_THRESHOLD = 0.15  # shuffled F1 above this => suspect leakage

OUT_DIR = _REPO_ROOT / "results" / "twodpca_verify"
OUT_JSON = OUT_DIR / "leakage_controls.json"

# Reverse map: display label -> manifest folder string (for the spot-check).
DISPLAY_TO_FOLDER = {disp: folder for folder, disp in CLASS_TO_DISPLAY.items()}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(GRIMSLEY_12_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def fit_best(train_imgs: np.ndarray, train_lbls: np.ndarray) -> TwoDPCAClassifier:
    clf = TwoDPCAClassifier(
        variant=BEST_VARIANT,
        classifier=BEST_CLASSIFIER,
        energy=BEST_ENERGY,
    )
    clf.fit(train_imgs, train_lbls)
    return clf


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load frozen split (resize 64, exact production convention) -------- #
    train_imgs, train_lbls = load_split_images(TRAIN_CSV, IMAGE_ROOT, RESIZE)
    val_imgs, val_lbls = load_split_images(VAL_CSV, IMAGE_ROOT, RESIZE)
    test_imgs, test_lbls = load_split_images(TEST_CSV, IMAGE_ROOT, RESIZE)

    # ---- MANDATORY parameter print --------------------------------------- #
    print("=" * 72)
    print("2DPCA leakage / chance-floor controls")
    print("=" * 72)
    print(f"split_dir         : {SPLIT_DIR}")
    print(f"image_root        : {IMAGE_ROOT}")
    print(f"resize            : {RESIZE}x{RESIZE} (bilinear, grayscale, /255)")
    print(f"best config       : variant={BEST_VARIANT} classifier={BEST_CLASSIFIER} energy={BEST_ENERGY}")
    print(f"shuffle seed      : np.random.default_rng({SHUFFLE_SEED}).permutation")
    print(f"spotcheck seed    : np.random.default_rng({SPOTCHECK_SEED}), n={N_SPOTCHECK}")
    print(f"chance floor      : 1/12 = {CHANCE_FLOOR:.4f}")
    print(f"leakage threshold : shuffled macro-F1 > {LEAKAGE_THRESHOLD} => suspect")
    print(f"scoring           : f1_score(average='macro', labels=GRIMSLEY_12_CLASSES, zero_division=0)")
    print(f"classes (order)   : {list(GRIMSLEY_12_CLASSES)}")
    print("-" * 72)
    print(f"row counts        : train={len(train_lbls)} val={len(val_lbls)} test={len(test_lbls)}")
    print("-" * 72)

    # ===================================================================== #
    # TASK 1 — shuffle control
    # ===================================================================== #
    print("TASK 1 — shuffle control (chance floor)")

    # Unshuffled best config on the SAME loaded arrays (reproduce ~0.24).
    clf_real = fit_best(train_imgs, train_lbls)
    unshuffled_f1 = macro_f1(test_lbls, clf_real.predict(test_imgs))
    print(f"  unshuffled best macro-F1 (test) : {unshuffled_f1:.4f}")

    # Shuffled-label control: permute TRAIN labels with rng(0).
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(train_lbls))
    train_lbls_shuffled = train_lbls[perm]
    n_changed = int((train_lbls_shuffled != train_lbls).sum())
    print(f"  shuffled labels changed         : {n_changed}/{len(train_lbls)} positions")
    clf_shuf = fit_best(train_imgs, train_lbls_shuffled)
    shuffled_f1 = macro_f1(test_lbls, clf_shuf.predict(test_imgs))
    print(f"  shuffled-label macro-F1 (test)  : {shuffled_f1:.4f}")
    near_chance = shuffled_f1 <= LEAKAGE_THRESHOLD
    real_beats_shuffled = unshuffled_f1 > shuffled_f1
    print(f"  shuffled <= {LEAKAGE_THRESHOLD} (near chance)   : {near_chance}")
    print(f"  unshuffled > shuffled           : {real_beats_shuffled}")
    print("-" * 72)

    # ===================================================================== #
    # TASK 2a — recording overlap
    # ===================================================================== #
    print("TASK 2a — source_recording overlap (per-class allocation => non-disjoint)")
    tr_df = pd.read_csv(TRAIN_CSV)
    va_df = pd.read_csv(VAL_CSV)
    te_df = pd.read_csv(TEST_CSV)
    tr_recs = set(tr_df["source_recording"].astype(str))
    va_recs = set(va_df["source_recording"].astype(str))
    te_recs = set(te_df["source_recording"].astype(str))
    overlap = {
        "tr_te": len(tr_recs & te_recs),
        "tr_va": len(tr_recs & va_recs),
        "va_te": len(va_recs & te_recs),
    }
    print(f"  |train recs|={len(tr_recs)} |val recs|={len(va_recs)} |test recs|={len(te_recs)}")
    print(f"  overlap train&test : {overlap['tr_te']}")
    print(f"  overlap train&val  : {overlap['tr_va']}")
    print(f"  overlap val&test   : {overlap['va_te']}")
    print("  why: build_stratified_split loops over each class and allocates that")
    print("       class's recordings independently, so one recording can be in")
    print("       train for class A and test for class B (cage-acoustic leakage).")
    print("       This can only INFLATE the score => a still-low 0.24 strengthens")
    print("       the genuine-ceiling verdict.")
    print("-" * 72)

    # ===================================================================== #
    # TASK 2b — label -> image alignment spot-check
    # ===================================================================== #
    print("TASK 2b — label->image alignment spot-check (10 random test rows)")
    from PIL import Image

    rng_sc = np.random.default_rng(SPOTCHECK_SEED)
    idxs = rng_sc.choice(len(te_df), size=min(N_SPOTCHECK, len(te_df)), replace=False)
    alignment_spotcheck: list[dict] = []
    for i in idxs.tolist():
        row = te_df.iloc[i]
        rel = str(row["path"])
        class_col = str(row["class"])
        # Folder embedded in the path: data/vocalmat_full/<folder>/<file>.png
        path_parts = Path(rel).parts
        path_folder = path_parts[-2] if len(path_parts) >= 2 else ""
        folder_display = CLASS_TO_DISPLAY.get(path_folder, None)
        folder_matches_class = folder_display == class_col

        # Confirm the loader's array for this row == the PNG decoded directly.
        img_path = Path(rel)
        if not img_path.is_absolute():
            img_path = IMAGE_ROOT / rel
        with Image.open(img_path) as im:
            direct = (
                np.asarray(
                    im.convert("L").resize((RESIZE, RESIZE), Image.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )
        loaded = test_imgs[i]  # loader's array at the same positional index
        array_matches = bool(np.array_equal(direct, loaded))

        # Also confirm the loader's label at this index == the csv class column.
        loader_label = str(test_lbls[i])
        label_matches = loader_label == class_col

        ok = folder_matches_class and array_matches and label_matches
        rec = {
            "row_index": int(i),
            "path": rel,
            "path_folder": path_folder,
            "folder_display": folder_display,
            "class_column": class_col,
            "loader_label": loader_label,
            "folder_matches_class": bool(folder_matches_class),
            "array_matches_png": array_matches,
            "label_matches_class": bool(label_matches),
            "result": "PASS" if ok else "FAIL",
        }
        alignment_spotcheck.append(rec)
        print(
            f"  row {i:>5} folder={path_folder:<12} class={class_col:<16} "
            f"arr={'Y' if array_matches else 'N'} lbl={'Y' if label_matches else 'N'} "
            f"-> {rec['result']}"
        )
    all_pass = all(r["result"] == "PASS" for r in alignment_spotcheck)
    print(f"  all 10 PASS : {all_pass}")
    print("-" * 72)

    # ===================================================================== #
    # TASK 3 — write JSON
    # ===================================================================== #
    out = {
        "parameters": {
            "resize": RESIZE,
            "best_variant": BEST_VARIANT,
            "best_classifier": BEST_CLASSIFIER,
            "best_energy": BEST_ENERGY,
            "shuffle_seed": SHUFFLE_SEED,
            "spotcheck_seed": SPOTCHECK_SEED,
            "n_spotcheck": N_SPOTCHECK,
            "chance_floor": CHANCE_FLOOR,
            "leakage_threshold": LEAKAGE_THRESHOLD,
            "row_counts": {
                "train": int(len(train_lbls)),
                "val": int(len(val_lbls)),
                "test": int(len(test_lbls)),
            },
        },
        "shuffled_macro_f1": shuffled_f1,
        "unshuffled_macro_f1": unshuffled_f1,
        "shuffled_near_chance": bool(near_chance),
        "unshuffled_beats_shuffled": bool(real_beats_shuffled),
        "recording_overlap": overlap,
        "recording_counts": {
            "train": len(tr_recs),
            "val": len(va_recs),
            "test": len(te_recs),
        },
        "alignment_spotcheck": alignment_spotcheck,
        "alignment_all_pass": bool(all_pass),
    }

    leakage_clean = near_chance and real_beats_shuffled and all_pass
    if leakage_clean:
        verdict = "NO LEAKAGE (chance floor + alignment confirmed)"
    else:
        reasons = []
        if not near_chance:
            reasons.append(
                f"shuffled macro-F1 {shuffled_f1:.4f} > {LEAKAGE_THRESHOLD} (above chance)"
            )
        if not real_beats_shuffled:
            reasons.append("unshuffled did not exceed shuffled")
        if not all_pass:
            reasons.append("alignment spot-check FAIL present")
        verdict = "LEAKAGE/ALIGNMENT ISSUE: " + "; ".join(reasons)
    out["verdict"] = verdict

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("VERDICT:", verdict)
    print(f"JSON written to: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
