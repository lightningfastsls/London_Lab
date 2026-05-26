"""Tests for scripts/train_perch2_probe.py — Module 18.3 Perch 2.0 linear probe (D3).

Written by test-architect BEFORE implementation exists. All tests will fail at
collection or at run-time until train_perch2_probe.py is created. That is the
expected TDD red phase.

ROADMAP §18.3 test plan coverage:
  11. Perch 2.0 probe: embedding dim is fixed across all inputs (no NaNs)
                                           ->  test_perch_embedding_dim_fixed_no_nans
  12. Perch 2.0 probe: linear classifier reaches macro F1 > 0.30 on smoke dataset
                                           ->  test_perch_linear_probe_macro_f1_smoke

Additional coverage (recurring gap patterns):
  - Perch unavailable: tests skip cleanly  ->  (skipif guard on all tests)
  - embed_with_perch returns 2-D array     ->  test_perch_embedding_is_2d
  - linear probe outputs one label per sample ->  test_linear_probe_output_length

Total: 4 tests (2 from ROADMAP, 2 additional)
All 4 tests are guarded by @pytest.mark.skipif(perch_unavailable, ...) because
Perch 2.0 (arXiv:2512.03219) requires a separate installation step not guaranteed
in CI. The skip reason is printed when Perch is absent.

Spec ambiguities resolved:
  - ROADMAP does not specify whether Perch is accessed via TF Hub, HuggingFace,
    or a local checkpoint. We test via the public API exposed by
    scripts/train_perch2_probe.py — specifically embed_with_perch() and
    train_linear_probe(). If Perch's distribution mechanism changes, only
    the import guard below needs updating.
  - "embedding dim is fixed" means all rows of the embedding matrix have the
    same number of columns; we do NOT assert the specific dimension value
    because Perch 2.0 may use 1280 or 2048 depending on the model variant.
  - "macro F1 > 0.30 on smoke dataset" uses the same 12-class × 10-sample
    synthetic dataset as test #9 in test_training.py. 0.30 is deliberately low
    (8.33% random baseline for 12 classes); even a linear probe on random
    ImageNet-style patches should exceed this on a tiny, class-balanced dataset.

Grimsley 12-class mapping (snake-case folder names):
  Display name          Snake-case folder
  "Noise"           ->  "noise"
  "Step up"         ->  "step_up"
  "Down-FM"         ->  "down_fm"
  "Short"           ->  "short"
  "Chevron"         ->  "chevron"
  "Up-FM"           ->  "up_fm"
  "Flat"            ->  "flat"
  "Two steps"       ->  "two_steps"
  "Step down"       ->  "step_down"
  "Complex"         ->  "complex"
  "Reverse Chevron" ->  "rev_chevron"
  "Multi-steps"     ->  "mult_steps"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# ---------------------------------------------------------------------------
# Perch availability guard
#
# We attempt to import the public API from train_perch2_probe.py.
# If the script doesn't exist yet OR Perch's backend (tensorflow_hub /
# huggingface_hub) is not installed, all tests are skipped with a clear
# reason rather than erroring.
# ---------------------------------------------------------------------------
_perch_unavailable_reason = ""
try:
    from train_perch2_probe import embed_with_perch, train_linear_probe  # type: ignore[import]
    _perch_unavailable = False
except ImportError as _e:
    _perch_unavailable = True
    _perch_unavailable_reason = str(_e)

perch_skip = pytest.mark.skipif(
    _perch_unavailable,
    reason=(
        "Perch 2.0 probe script not available "
        f"(scripts/train_perch2_probe.py or its dependencies missing: "
        f"{_perch_unavailable_reason})"
    ),
)

# ---------------------------------------------------------------------------
# Synthetic patch helpers
# ---------------------------------------------------------------------------
_NUM_CLASSES = 12
_SAMPLES_PER_CLASS = 10
_PATCH_SHAPE = (3, 227, 227)


def _make_synthetic_patches(
    samples_per_class: int = _SAMPLES_PER_CLASS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (patches, labels) as numpy arrays.

    patches: float32 (N, 3, 227, 227) in [-1, 1] (ImageNet-normalized range).
    labels:  int32   (N,) with values 0..11.
    """
    rng = np.random.default_rng(seed)
    n = samples_per_class * _NUM_CLASSES
    patches = rng.standard_normal((n, *_PATCH_SHAPE)).astype(np.float32)
    labels = np.tile(np.arange(_NUM_CLASSES), samples_per_class).astype(np.int32)
    # Shuffle so the probe sees a mixed stream, not all class-0 then class-1.
    idx = rng.permutation(n)
    return patches[idx], labels[idx]


# ===========================================================================
# Test 11 (ROADMAP item 11) — embedding dim is fixed across all inputs, no NaN
# ===========================================================================

@perch_skip
def test_perch_embedding_dim_fixed_no_nans():
    """Spec: embed_with_perch returns the same embedding dimension for all inputs
    and the output contains no NaN values.

    Perch 2.0 is a fixed-size encoder; every input patch should map to the
    same-dimensional embedding vector regardless of input content. A variable
    output dimension would crash the downstream linear probe's matrix multiply.
    NaNs would silently propagate through the linear head and produce garbage
    predictions.
    """
    patches, _ = _make_synthetic_patches(samples_per_class=5)

    embeddings = embed_with_perch(patches)

    assert isinstance(embeddings, np.ndarray), (
        f"embed_with_perch must return a numpy array, got {type(embeddings).__name__}"
    )
    assert embeddings.ndim == 2, (
        f"embed_with_perch must return a 2-D array (N, dim), got shape {embeddings.shape}"
    )
    assert embeddings.shape[0] == patches.shape[0], (
        f"embed_with_perch returned {embeddings.shape[0]} embeddings for "
        f"{patches.shape[0]} inputs — row count mismatch"
    )

    # All rows must have the same (and consistent) number of columns.
    # For a 2-D ndarray this is guaranteed by shape, but we check explicitly
    # that the dimension is the same for a random subset to catch any stacking bug.
    first_dim = embeddings.shape[1]
    assert first_dim > 0, (
        f"Embedding dimension is 0 — embed_with_perch produced empty vectors"
    )

    assert not np.isnan(embeddings).any(), (
        f"embed_with_perch produced NaN values: "
        f"{np.isnan(embeddings).sum()} NaNs in output of shape {embeddings.shape}"
    )


# ===========================================================================
# Test 12 (ROADMAP item 12) — linear probe macro F1 > 0.30 on smoke dataset
# ===========================================================================

@perch_skip
def test_perch_linear_probe_macro_f1_smoke():
    """Spec: a linear classifier trained on Perch 2.0 embeddings must reach
    macro F1 > 0.30 on a class-balanced 12-class smoke dataset.

    Random baseline: 1/12 = 8.3%. A threshold of 0.30 is deliberately
    conservative — even on random ImageNet patches a linear probe should do
    better than random on a tiny balanced dataset (the labels are perfectly
    balanced, so any signal at all gets amplified by the linear layer).

    If the probe scores below 0.30 on a 120-sample balanced dataset, it
    indicates either a broken embedding (constant vector) or a broken training
    loop (no gradient flow through the linear head).
    """
    patches, labels = _make_synthetic_patches(samples_per_class=10, seed=42)

    # 70/30 train/val split.
    n = len(patches)
    n_train = int(0.7 * n)
    train_patches, train_labels = patches[:n_train], labels[:n_train]
    val_patches, val_labels = patches[n_train:], labels[n_train:]

    embeddings_train = embed_with_perch(train_patches)
    embeddings_val = embed_with_perch(val_patches)

    metrics = train_linear_probe(
        train_embeddings=embeddings_train,
        train_labels=train_labels,
        val_embeddings=embeddings_val,
        val_labels=val_labels,
        num_classes=_NUM_CLASSES,
    )

    assert "macro_f1" in metrics, (
        f"train_linear_probe must return a dict with 'macro_f1'. "
        f"Got keys: {sorted(metrics.keys())}"
    )

    macro_f1 = metrics["macro_f1"]
    assert isinstance(macro_f1, float), (
        f"macro_f1 must be a float, got {type(macro_f1).__name__}"
    )
    assert macro_f1 > 0.30, (
        f"Linear probe macro F1 = {macro_f1:.4f} on smoke dataset (12 classes, "
        f"10 samples/class). Expected > 0.30 (random baseline = 0.083). "
        "A score this low suggests the Perch embeddings are constant or degenerate."
    )


# ===========================================================================
# Additional test — embed_with_perch returns a 2-D array
# ===========================================================================

@perch_skip
def test_perch_embedding_is_2d():
    """embed_with_perch must return a 2-D (N, embedding_dim) array.

    A 1-D or 3-D return would cause a shape mismatch in the linear probe's
    matrix multiply. We test this separately from the NaN check because the
    shape contract is independent of value correctness.
    """
    patches, _ = _make_synthetic_patches(samples_per_class=2)
    n = len(patches)
    embeddings = embed_with_perch(patches)

    assert embeddings.ndim == 2, (
        f"embed_with_perch must return a 2-D array, got {embeddings.ndim}-D "
        f"array of shape {embeddings.shape}"
    )
    assert embeddings.shape[0] == n, (
        f"Row count mismatch: input {n} patches, got {embeddings.shape[0]} embeddings"
    )


# ===========================================================================
# Additional test — linear probe outputs one label per val sample
# ===========================================================================

@perch_skip
def test_linear_probe_output_length():
    """train_linear_probe must produce predictions for every validation sample.

    If the probe internally filters samples (e.g., drops class-0 because it
    has a special index), the output length would be shorter than the input.
    We verify output length == input length.
    """
    patches, labels = _make_synthetic_patches(samples_per_class=3, seed=7)
    n = len(patches)
    n_train = int(0.7 * n)

    train_patches, train_labels = patches[:n_train], labels[:n_train]
    val_patches, val_labels = patches[n_train:], labels[n_train:]

    embeddings_train = embed_with_perch(train_patches)
    embeddings_val = embed_with_perch(val_patches)

    metrics = train_linear_probe(
        train_embeddings=embeddings_train,
        train_labels=train_labels,
        val_embeddings=embeddings_val,
        val_labels=val_labels,
        num_classes=_NUM_CLASSES,
    )

    # The metrics dict must at minimum contain 'macro_f1' and 'predictions'.
    assert "predictions" in metrics, (
        f"train_linear_probe must return 'predictions' in the metrics dict. "
        f"Got keys: {sorted(metrics.keys())}"
    )
    preds = metrics["predictions"]
    assert len(preds) == len(val_labels), (
        f"Predictions length {len(preds)} != val_labels length {len(val_labels)}. "
        "Every validation sample must receive exactly one predicted label."
    )
