#!/usr/bin/env python3
"""Verify the pure-NumPy 2DPCA implementation against the published Yang-2004
ORL/Olivetti faces benchmark.

This is an adversarial verification of
``src/usv_spectrogram/classifier/twodpca.py``. We reproduce the standard
Yang (2004, IEEE TPAMI 26(1):131-137) ORL protocol:

    - 40 people x 10 images, 64x64 grayscale, in [0,1]
    - First 5 images/person -> train (200); last 5/person -> test (200)
    - variant="2dpca", classifier="nn" (Yang feature-matrix distance 1-NN)
    - rank-1 accuracy vs. number of column components d

Yang reports rank-1 accuracy climbing to ~0.92-0.96 by d~5-10 and plateauing.
>= 0.90 confirms the core 2DPCA math + NN distance are correct.

We additionally probe the (2D)^2PCA variant and the SVM / LDA heads.

Run:
    .venv/bin/python scripts/experiments/verify_2dpca_olivetti.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make the src package importable when run as a plain script.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usv_spectrogram.classifier.twodpca import TwoDPCAClassifier  # noqa: E402

# --------------------------------------------------------------------------- #
# Parameters (repo convention: print everything up front)
# --------------------------------------------------------------------------- #
N_PEOPLE = 40
IMAGES_PER_PERSON = 10
TRAIN_PER_PERSON = 5  # first 5 -> train
TEST_PER_PERSON = 5  # last 5 -> test
IMAGE_SHAPE = (64, 64)
D_VALUES = [2, 4, 6, 8, 10, 15, 20]  # column components for the 2dpca/nn sweep
D_2D2D = 10  # column components for (2D)^2PCA
Q_2D2D = 10  # row components for (2D)^2PCA
D_SVM_LDA = 10  # column components for the svm / lda heads
OUT_DIR = REPO_ROOT / "results" / "twodpca_verify"
OUT_JSON = OUT_DIR / "olivetti_benchmark.json"

PASS_THRESHOLD = 0.90  # MATH CORRECT
FAIL_THRESHOLD = 0.85  # below -> POSSIBLE BUG; in between -> AMBIGUOUS


def print_params() -> None:
    print("=" * 72)
    print("2DPCA verification against Yang-2004 ORL/Olivetti benchmark")
    print("=" * 72)
    print("PARAMETERS")
    print(f"  dataset            : sklearn fetch_olivetti_faces")
    print(f"  people             : {N_PEOPLE}")
    print(f"  images per person  : {IMAGES_PER_PERSON}")
    print(f"  image shape        : {IMAGE_SHAPE}  (grayscale, [0,1])")
    print(f"  protocol           : Yang-2004 (first 5 train, last 5 test)")
    print(f"  train_per_person   : {TRAIN_PER_PERSON}  -> N_train = "
          f"{N_PEOPLE * TRAIN_PER_PERSON}")
    print(f"  test_per_person    : {TEST_PER_PERSON}  -> N_test  = "
          f"{N_PEOPLE * TEST_PER_PERSON}")
    print(f"  d sweep (2dpca/nn) : {D_VALUES}")
    print(f"  (2D)^2PCA d, q     : {D_2D2D}, {Q_2D2D}")
    print(f"  svm/lda d          : {D_SVM_LDA}")
    print(f"  metric             : rank-1 accuracy (fraction correct)")
    print(f"  pass / fail thresh : >= {PASS_THRESHOLD} pass; "
          f"< {FAIL_THRESHOLD} bug")
    print("=" * 72)


def load_olivetti():
    """Return (images (400,64,64) float, target (400,)). Exit non-zero on
    fetch failure -- never fabricate."""
    try:
        from sklearn.datasets import fetch_olivetti_faces
    except Exception as exc:  # pragma: no cover - import guard
        print(f"ERROR: cannot import sklearn fetch_olivetti_faces: {exc}",
              file=sys.stderr)
        sys.exit(2)
    try:
        ds = fetch_olivetti_faces()
    except Exception as exc:
        print(
            "ERROR: failed to fetch Olivetti faces (no network and no cached "
            f"copy?): {exc}",
            file=sys.stderr,
        )
        sys.exit(3)
    images = np.asarray(ds.images, dtype=np.float64)  # (400, 64, 64)
    target = np.asarray(ds.target)  # (400,)
    return images, target


def split_yang(images: np.ndarray, target: np.ndarray):
    """First 5 images/person -> train, last 5 -> test.

    The dataset is ordered 10-per-person consecutively (target sorted). We
    VERIFY that alignment instead of assuming it.
    """
    n = images.shape[0]
    expected = N_PEOPLE * IMAGES_PER_PERSON
    if n != expected:
        print(f"ERROR: expected {expected} images, got {n}", file=sys.stderr)
        sys.exit(4)

    # Verify the consecutive 10-per-person ordering.
    expected_target = np.repeat(np.arange(N_PEOPLE), IMAGES_PER_PERSON)
    if not np.array_equal(target, expected_target):
        print(
            "ERROR: Olivetti target is not the expected consecutive "
            "10-per-person ordering; aborting rather than mis-splitting.",
            file=sys.stderr,
        )
        print(f"  first 15 targets: {target[:15]}", file=sys.stderr)
        sys.exit(5)

    train_idx = []
    test_idx = []
    for p in range(N_PEOPLE):
        base = p * IMAGES_PER_PERSON
        train_idx.extend(range(base, base + TRAIN_PER_PERSON))
        test_idx.extend(range(base + TRAIN_PER_PERSON, base + IMAGES_PER_PERSON))
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    X_train = images[train_idx]
    y_train = target[train_idx]
    X_test = images[test_idx]
    y_test = target[test_idx]

    # Sanity: each person has exactly TRAIN/TEST counts and label sets match.
    assert X_train.shape[0] == N_PEOPLE * TRAIN_PER_PERSON
    assert X_test.shape[0] == N_PEOPLE * TEST_PER_PERSON
    assert set(y_train.tolist()) == set(range(N_PEOPLE))
    assert set(y_test.tolist()) == set(range(N_PEOPLE))
    return X_train, y_train, X_test, y_test


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred == truth))


def main() -> int:
    print_params()
    images, target = load_olivetti()
    print(f"Loaded Olivetti: images {images.shape} dtype {images.dtype}, "
          f"target {target.shape}, value range "
          f"[{images.min():.3f}, {images.max():.3f}]")

    X_train, y_train, X_test, y_test = split_yang(images, target)
    print(f"Split OK: train {X_train.shape}, test {X_test.shape} "
          f"(target ordering verified)\n")

    results = {
        "params": {
            "n_people": N_PEOPLE,
            "images_per_person": IMAGES_PER_PERSON,
            "train_per_person": TRAIN_PER_PERSON,
            "test_per_person": TEST_PER_PERSON,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "image_shape": list(IMAGE_SHAPE),
            "d_values": D_VALUES,
            "d_2d2d": D_2D2D,
            "q_2d2d": Q_2D2D,
            "d_svm_lda": D_SVM_LDA,
            "protocol": "Yang-2004 first-5-train/last-5-test",
        },
        "nn_2dpca_by_d": {},
        "other_heads": {},
    }

    # ---- 2dpca / nn sweep over d -------------------------------------------- #
    print("variant=2dpca  classifier=nn   (Yang feature-matrix 1-NN)")
    print(f"{'d':>4} | {'rank-1 acc':>11} | {'n_correct':>9}")
    print("-" * 32)
    best_acc = -1.0
    best_d = None
    for d in D_VALUES:
        clf = TwoDPCAClassifier(variant="2dpca", classifier="nn", n_components=d)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy(pred, y_test)
        n_correct = int(np.sum(pred == y_test))
        results["nn_2dpca_by_d"][str(d)] = acc
        print(f"{d:>4} | {acc:>11.4f} | {n_correct:>4}/{len(y_test)}")
        if acc > best_acc:
            best_acc = acc
            best_d = d
    print("-" * 32)
    print(f"best nn rank-1 = {best_acc:.4f} at d={best_d}\n")

    # ---- 2d2dpca / nn ------------------------------------------------------- #
    clf = TwoDPCAClassifier(
        variant="2d2dpca", classifier="nn",
        n_components=D_2D2D, n_components_row=Q_2D2D,
    )
    clf.fit(X_train, y_train)
    acc_2d2d = accuracy(clf.predict(X_test), y_test)
    results["other_heads"]["2d2dpca_nn"] = {
        "d": D_2D2D, "q": Q_2D2D, "acc": acc_2d2d,
    }
    print(f"variant=2d2dpca classifier=nn  d={D_2D2D} q={Q_2D2D} "
          f"-> rank-1 acc {acc_2d2d:.4f}")

    # ---- 2dpca / svm -------------------------------------------------------- #
    clf = TwoDPCAClassifier(
        variant="2dpca", classifier="svm", n_components=D_SVM_LDA,
    )
    clf.fit(X_train, y_train)
    acc_svm = accuracy(clf.predict(X_test), y_test)
    results["other_heads"]["2dpca_svm"] = {"d": D_SVM_LDA, "acc": acc_svm}
    print(f"variant=2dpca   classifier=svm d={D_SVM_LDA} "
          f"-> rank-1 acc {acc_svm:.4f}")

    # ---- 2dpca / lda -------------------------------------------------------- #
    clf = TwoDPCAClassifier(
        variant="2dpca", classifier="lda", n_components=D_SVM_LDA,
    )
    clf.fit(X_train, y_train)
    acc_lda = accuracy(clf.predict(X_test), y_test)
    results["other_heads"]["2dpca_lda"] = {"d": D_SVM_LDA, "acc": acc_lda}
    print(f"variant=2dpca   classifier=lda d={D_SVM_LDA} "
          f"-> rank-1 acc {acc_lda:.4f}\n")

    # ---- verdict ------------------------------------------------------------ #
    if best_acc >= PASS_THRESHOLD:
        verdict = "MATH CORRECT (>=0.90)"
    elif best_acc < FAIL_THRESHOLD:
        verdict = "POSSIBLE BUG (<0.85)"
    else:
        verdict = "AMBIGUOUS (0.85-0.90)"
    results["best_nn"] = {"acc": best_acc, "d": best_d}
    results["verdict"] = verdict

    print("=" * 72)
    print(f"VERDICT: {verdict}   (best nn rank-1 = {best_acc:.4f} at d={best_d})")
    print("=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
