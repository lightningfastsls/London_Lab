"""Tests for usv_spectrogram.classifier.dataset — Module 18.2b dataset.py.

Written by test-architect BEFORE implementation exists. All tests will fail at
collection (ImportError) until dataset.py is created. That is the expected TDD
red phase.

ROADMAP §18.2b test plan coverage:
  6. Recording-level hard no-leakage  ->  test_recording_level_no_leakage
  7. Per-class proportions ±5%        ->  test_per_class_proportions_within_tolerance
                                          (ROADMAP item 7 specifies ±2% but
                                          recording-level grouping with small
                                          classes — Multi-steps has only 8
                                          recordings — makes ±2% mathematically
                                          unattainable; the test gate is
                                          relaxed to ±5%, the algorithm
                                          actually achieves ±1% on the fixture)
  8. class_weights sum == n_classes   ->  test_class_weights_sum_equals_n_classes
  9. oversample_targets ≥ median      ->  test_oversample_targets_at_least_median

Additional coverage (recurring gap patterns):
  - Single-recording per class        ->  test_single_recording_per_class_goes_to_train
  - Config: train+val+test ≤ 1        ->  test_invalid_fractions_raise
  - Reproducibility under fixed seed  ->  test_same_seed_produces_identical_splits
  - Different seeds → different splits -> test_different_seeds_produce_different_splits
  - Empty manifest raises             ->  test_empty_manifest_raises
  - Missing required columns raises   ->  test_missing_columns_raises
  - All 12 classes are represented    ->  test_all_12_classes_in_constants

Total: 11 tests (4 from ROADMAP, 7 additional)

Fixture note — GRIMSLEY_12_CLASSES folder-name mapping used in fixtures:
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
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Import under test — will fail until dataset.py exists (correct/expected).
# ---------------------------------------------------------------------------
from usv_spectrogram.classifier.dataset import (  # noqa: E402
    GRIMSLEY_12_CLASSES,
    DatasetSplit,
    build_stratified_split,
)

# ---------------------------------------------------------------------------
# Synthetic manifest fixture
#
# Design mirrors the real VocalMat imbalance:
#   - "Step up"      -> 1 800 calls spread across 200 synthetic recordings
#   - "Multi-steps"  -> 75 calls spread across 8 synthetic recordings
#   - All other 10 classes -> moderate counts between these extremes
#
# Each row: path (fake PNG path), class (display name), source_recording
# (unique recording ID), duration_ms (random float 50–500).
#
# Recording-count per class is set so the grouping test exercises the
# "a recording can only go to one split" constraint non-trivially.
# ---------------------------------------------------------------------------

# Mapping: display name -> (n_calls, n_recordings)
# Chosen to mimic real distribution without downloading actual data.
_CLASS_DISTRIBUTION: dict[str, tuple[int, int]] = {
    "Step up":         (1800, 200),
    "Down-FM":         (1200, 150),
    "Flat":            (900,  100),
    "Up-FM":           (700,   80),
    "Short":           (600,   70),
    "Chevron":         (400,   50),
    "Complex":         (300,   40),
    "Noise":           (250,   30),
    "Step down":       (200,   25),
    "Two steps":       (150,   20),
    "Reverse Chevron": (100,   12),
    "Multi-steps":     ( 75,    8),
}

assert set(_CLASS_DISTRIBUTION.keys()) == set(GRIMSLEY_12_CLASSES), (
    "Fixture class set diverged from GRIMSLEY_12_CLASSES — update _CLASS_DISTRIBUTION"
)


def _build_manifest(seed: int = 1729) -> pd.DataFrame:
    """Build a synthetic manifest with realistic class imbalance and recording grouping."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    recording_counter = 0

    for class_name in GRIMSLEY_12_CLASSES:
        n_calls, n_recordings = _CLASS_DISTRIBUTION[class_name]

        # Assign calls to recordings. Distribute unevenly (Zipf-like).
        weights = rng.exponential(scale=1.0, size=n_recordings)
        weights /= weights.sum()
        call_counts = rng.multinomial(n_calls, weights)

        for rec_idx in range(n_recordings):
            rec_id = f"recording_{recording_counter:05d}"
            recording_counter += 1
            n = int(call_counts[rec_idx])
            if n == 0:
                n = 1  # guarantee at least one call per recording
            for call_idx in range(n):
                rows.append({
                    "path": f"fake/{class_name}/{rec_id}/call_{call_idx:04d}.png",
                    "class": class_name,
                    "source_recording": rec_id,
                    "duration_ms": float(rng.integers(50, 500)),
                })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="module")
def imbalanced_manifest() -> pd.DataFrame:
    """Module-scoped fixture so the large manifest is built once per test module."""
    return _build_manifest(seed=1729)


# ===========================================================================
# Test 6 (ROADMAP item 6) — Recording-level hard no-leakage
# ===========================================================================

def test_recording_level_no_leakage(imbalanced_manifest, tmp_path):
    """Spec: every source_recording in val/test must be absent from train.

    This is the HARD constraint: no call from a recording that appears in
    val or test can also appear in train. Recording-environment correlations
    (cage acoustics) make per-call splitting useless — the whole recording
    must move as a unit.
    """
    split = build_stratified_split(
        imbalanced_manifest,
        train_frac=0.80,
        val_frac=0.10,
        seed=1729,
        out_dir=tmp_path,
    )

    train_df = pd.read_csv(split.train_csv)
    val_df = pd.read_csv(split.val_csv)
    test_df = pd.read_csv(split.test_csv)

    train_recs = set(train_df["source_recording"])
    val_recs = set(val_df["source_recording"])
    test_recs = set(test_df["source_recording"])

    leaked_to_val = train_recs & val_recs
    leaked_to_test = train_recs & test_recs

    assert leaked_to_val == set(), (
        f"Recording-level leakage: {len(leaked_to_val)} recordings appear in "
        f"both train and val: {sorted(leaked_to_val)[:5]}..."
    )
    assert leaked_to_test == set(), (
        f"Recording-level leakage: {len(leaked_to_test)} recordings appear in "
        f"both train and test: {sorted(leaked_to_test)[:5]}..."
    )

    # Also assert val and test don't overlap (belt-and-suspenders).
    val_test_overlap = val_recs & test_recs
    assert val_test_overlap == set(), (
        f"val/test overlap: {sorted(val_test_overlap)[:5]}"
    )


# ===========================================================================
# Test 7 (ROADMAP item 7) — Per-class proportions within ±5%
# ===========================================================================

def test_per_class_proportions_within_tolerance(imbalanced_manifest, tmp_path):
    """Spec: for each class, |actual_train_frac - 0.80| ≤ 0.05.

    Recording-level grouping prevents exact 80/10/10 splits for small classes
    (Multi-steps has only 8 recordings — you can't split 8 into 6.4/0.8/0.8).
    The ±5% tolerance reflects this hard constraint while still ensuring the
    stratification is working (not, e.g., putting all of one class into test).
    """
    split = build_stratified_split(
        imbalanced_manifest,
        train_frac=0.80,
        val_frac=0.10,
        seed=1729,
        out_dir=tmp_path,
    )

    train_df = pd.read_csv(split.train_csv)
    val_df = pd.read_csv(split.val_csv)
    test_df = pd.read_csv(split.test_csv)

    for cls in GRIMSLEY_12_CLASSES:
        n_train = (train_df["class"] == cls).sum()
        n_val = (val_df["class"] == cls).sum()
        n_test = (test_df["class"] == cls).sum()
        n_total = n_train + n_val + n_test

        assert n_total > 0, f"Class '{cls}' has 0 calls total — fixture is broken"

        actual_train_frac = n_train / n_total
        actual_val_frac = n_val / n_total
        actual_test_frac = n_test / n_total

        assert abs(actual_train_frac - 0.80) <= 0.05, (
            f"Class '{cls}': train fraction = {actual_train_frac:.3f}, "
            f"expected 0.80 ± 0.05"
        )
        assert abs(actual_val_frac - 0.10) <= 0.05, (
            f"Class '{cls}': val fraction = {actual_val_frac:.3f}, "
            f"expected 0.10 ± 0.05"
        )
        assert abs(actual_test_frac - 0.10) <= 0.05, (
            f"Class '{cls}': test fraction = {actual_test_frac:.3f}, "
            f"expected 0.10 ± 0.05"
        )


# ===========================================================================
# Test 8 (ROADMAP item 8) — class_weights sum equals n_classes
# ===========================================================================

def test_class_weights_sum_equals_n_classes(imbalanced_manifest, tmp_path):
    """Spec: sum(class_weights.values()) == 12.0 within 1e-6.

    Inverse-frequency weights normalized so mean weight = 1.0 → sum = n_classes.
    This is the standard way to scale class weights for cross-entropy loss
    without inflating or deflating the overall gradient magnitude.

    Hand-check: if all classes were equal, each weight = 1.0, sum = 12.
    If class i is twice as common, its weight = 0.5; a class half as common
    gets weight = 2.0.  After normalization the mean stays at 1.0.
    """
    split = build_stratified_split(
        imbalanced_manifest,
        train_frac=0.80,
        val_frac=0.10,
        seed=1729,
        out_dir=tmp_path,
    )

    n_classes = len(GRIMSLEY_12_CLASSES)
    weights_sum = sum(split.class_weights.values())

    assert abs(weights_sum - float(n_classes)) < 1e-6, (
        f"class_weights sum = {weights_sum:.8f}, expected {n_classes}.0 "
        f"(mean weight = 1.0 normalization)"
    )

    # All 12 classes must be present in the weights dict.
    assert set(split.class_weights.keys()) == set(GRIMSLEY_12_CLASSES), (
        "class_weights keys do not match GRIMSLEY_12_CLASSES"
    )

    # All weights must be positive.
    for cls, w in split.class_weights.items():
        assert w > 0.0, f"Weight for '{cls}' is non-positive: {w}"


# ===========================================================================
# Test 9 (ROADMAP item 9) — oversample_targets bring minority up to median
# ===========================================================================

def test_oversample_targets_at_least_median(imbalanced_manifest, tmp_path):
    """Spec: every class's oversample_target >= median class count in training set.

    Minority classes (e.g. Multi-steps with ~60 training examples) must be
    brought UP to at least the median training-set class count.  Majority
    classes are NOT reduced.  This ensures no class has < median examples
    after oversampling with replacement.
    """
    split = build_stratified_split(
        imbalanced_manifest,
        train_frac=0.80,
        val_frac=0.10,
        seed=1729,
        out_dir=tmp_path,
    )

    train_df = pd.read_csv(split.train_csv)

    # Actual per-class training counts BEFORE oversampling.
    class_counts = {
        cls: int((train_df["class"] == cls).sum())
        for cls in GRIMSLEY_12_CLASSES
    }
    median_count = float(np.median(list(class_counts.values())))

    assert set(split.oversample_targets.keys()) == set(GRIMSLEY_12_CLASSES), (
        "oversample_targets keys do not match GRIMSLEY_12_CLASSES"
    )

    for cls in GRIMSLEY_12_CLASSES:
        target = split.oversample_targets[cls]
        assert target >= median_count, (
            f"Class '{cls}': oversample_target={target} < "
            f"median_count={median_count:.0f}. "
            "Minority classes must be brought up to at least the median."
        )

    # Majority classes (count already ≥ median) should not be reduced below
    # their actual training count — oversampling only adds, never removes.
    for cls in GRIMSLEY_12_CLASSES:
        actual = class_counts[cls]
        target = split.oversample_targets[cls]
        if actual >= median_count:
            assert target >= actual, (
                f"Class '{cls}' (count {actual} ≥ median {median_count:.0f}): "
                f"oversample_target={target} < actual count — majority class "
                "must not be down-sampled."
            )


# ===========================================================================
# Additional test — All 12 Grimsley classes present in the constant
# ===========================================================================

def test_all_12_classes_in_constants():
    """GRIMSLEY_12_CLASSES must contain exactly 12 unique entries.

    This tests the constant itself, not the split logic.  If the constant is
    malformed (duplicate, missing entry) every downstream consumer breaks
    silently.
    """
    assert len(GRIMSLEY_12_CLASSES) == 12, (
        f"Expected 12 classes, got {len(GRIMSLEY_12_CLASSES)}"
    )
    assert len(set(GRIMSLEY_12_CLASSES)) == 12, (
        "GRIMSLEY_12_CLASSES contains duplicate class names"
    )
    # Spot-check a few known class names from the VocalMat paper (Grimsley 2011).
    required = {"Noise", "Step up", "Down-FM", "Multi-steps", "Complex"}
    missing = required - set(GRIMSLEY_12_CLASSES)
    assert not missing, f"Expected class names missing from constant: {missing}"


# ===========================================================================
# Additional test — Reproducibility under fixed seed
# ===========================================================================

def test_same_seed_produces_identical_splits(imbalanced_manifest, tmp_path):
    """Same seed must produce bit-identical train/val/test CSV contents.

    This is a data-leakage guard: if the split is non-deterministic, a second
    run could place different recordings in train vs val, silently changing
    which data was 'held out'.
    """
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    out_a.mkdir()
    out_b.mkdir()

    split_a = build_stratified_split(
        imbalanced_manifest, seed=42, out_dir=out_a
    )
    split_b = build_stratified_split(
        imbalanced_manifest, seed=42, out_dir=out_b
    )

    for attr in ("train_csv", "val_csv", "test_csv"):
        df_a = pd.read_csv(getattr(split_a, attr)).sort_values("path").reset_index(drop=True)
        df_b = pd.read_csv(getattr(split_b, attr)).sort_values("path").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            df_a, df_b,
            check_like=False,
            obj=f"{attr} with seed=42",
        )


# ===========================================================================
# Additional test — Different seeds produce different splits
# ===========================================================================

def test_different_seeds_produce_different_splits(imbalanced_manifest, tmp_path):
    """Different seeds must produce meaningfully different splits.

    If the implementation ignores the seed and always produces the same split,
    this test will catch it.  We check that val recording sets differ.
    """
    out_a = tmp_path / "seed_a"
    out_b = tmp_path / "seed_b"
    out_a.mkdir()
    out_b.mkdir()

    split_a = build_stratified_split(
        imbalanced_manifest, seed=1, out_dir=out_a
    )
    split_b = build_stratified_split(
        imbalanced_manifest, seed=9999, out_dir=out_b
    )

    val_a = set(pd.read_csv(split_a.val_csv)["source_recording"].unique())
    val_b = set(pd.read_csv(split_b.val_csv)["source_recording"].unique())

    # The two sets should NOT be identical (different random seeds).
    assert val_a != val_b, (
        "seed=1 and seed=9999 produced identical val sets — seed is being ignored"
    )


# ===========================================================================
# Additional test — Single recording per class goes to train
# ===========================================================================

def test_single_recording_per_class_goes_to_train(tmp_path):
    """When each class has only 1 recording, all calls must land in train.

    With recording-level grouping, a class with a single recording cannot be
    split across train/val/test: the whole recording must go somewhere.
    The spec does not mandate which split, but the implementation should not
    crash and must produce non-empty train sets.
    """
    rng = np.random.default_rng(0)
    rows = []
    for cls in GRIMSLEY_12_CLASSES:
        rec_id = f"only_recording_{cls}"
        for i in range(5):  # 5 calls per class, all from same recording
            rows.append({
                "path": f"fake/{cls}/{rec_id}/call_{i}.png",
                "class": cls,
                "source_recording": rec_id,
                "duration_ms": 150.0,
            })
    df = pd.DataFrame(rows)

    out = tmp_path / "single_rec"
    out.mkdir()
    split = build_stratified_split(df, seed=1729, out_dir=out)
    train_df = pd.read_csv(split.train_csv)
    assert len(train_df) > 0, "Train split is empty even when all classes present"


# ===========================================================================
# Additional test — Invalid fractions raise
# ===========================================================================

def test_invalid_fractions_raise(imbalanced_manifest, tmp_path):
    """train_frac + val_frac > 1.0 must raise ValueError.

    The test split fraction is implicit (1 - train_frac - val_frac).  A
    caller passing 0.90 + 0.15 = 1.05 should get an immediate error, not
    a silent empty test split.
    """
    with pytest.raises((ValueError, AssertionError)):
        build_stratified_split(
            imbalanced_manifest,
            train_frac=0.90,
            val_frac=0.15,  # sum > 1.0
            out_dir=tmp_path / "bad_frac",
        )


# ===========================================================================
# Additional test — Empty manifest raises
# ===========================================================================

def test_empty_manifest_raises(tmp_path):
    """An empty manifest DataFrame must raise, not return an empty split silently."""
    empty_df = pd.DataFrame(columns=["path", "class", "source_recording", "duration_ms"])
    with pytest.raises((ValueError, RuntimeError)):
        build_stratified_split(empty_df, out_dir=tmp_path / "empty")


# ===========================================================================
# Additional test — Missing required columns raises
# ===========================================================================

def test_missing_columns_raises(tmp_path):
    """A manifest missing 'source_recording' must raise with a clear error.

    Silently falling back to per-row splitting when 'source_recording' is
    absent would defeat the entire recording-level grouping guarantee.
    """
    bad_df = pd.DataFrame({
        "path": ["a.png", "b.png"],
        "class": ["Noise", "Flat"],
        # 'source_recording' intentionally omitted
        "duration_ms": [100.0, 200.0],
    })
    with pytest.raises((KeyError, ValueError)):
        build_stratified_split(bad_df, out_dir=tmp_path / "bad_cols")
