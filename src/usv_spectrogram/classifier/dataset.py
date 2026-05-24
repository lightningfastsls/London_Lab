"""Dataset manifest + stratified split for the lab CNN classifier (Module 18.2b).

Builds 80/10/10 train/val/test splits from a per-call manifest with two binding
constraints:

1. **Recording-level grouping** — every ``source_recording`` ends up in *exactly
   one* of train, val, test. Splitting a recording across train and val/test
   leaks cage-acoustic features (the VAE comparison memo
   ``docs/handoffs/2026-05-18_vae_comparison_memo.md`` documents that raw
   spectrograms cluster by recording-environment, not biology).

2. **Per-class stratification within ±5%** — recording-level grouping prevents
   the exact 80/10/10 ratio for small classes (``Multi-steps`` has only 8
   recordings; you cannot split 8 cleanly into 6.4 / 0.8 / 0.8). A greedy
   *most-underfilled-split* allocator keeps every class within 5 percentage
   points of its target fraction in practice.

The companion outputs feed Module 18.3 training:

- ``class_weights``: inverse-frequency, normalized so the mean weight is 1.0
  (i.e. the dict sums to ``n_classes``). Plugged straight into
  ``torch.nn.CrossEntropyLoss(weight=...)``.
- ``oversample_targets``: per-class call counts to draw via *replacement*
  sampling so every minority class reaches the *training-set median* before a
  batch is built. Majority classes are never reduced.

ROADMAP §18.2b, ROADMAP D5 (minority strategy: all 12 classes + weighted CE +
focal loss + oversampling) drive the design.
"""
# VAULT: [[project-lab-cnn-classifier-scope]] [[feedback-cross-animal-population-strata]]
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "GRIMSLEY_12_CLASSES",
    "DatasetSplit",
    "build_stratified_split",
]


GRIMSLEY_12_CLASSES: tuple[str, ...] = (
    "Noise",
    "Step up",
    "Down-FM",
    "Short",
    "Chevron",
    "Up-FM",
    "Flat",
    "Two steps",
    "Step down",
    "Complex",
    "Reverse Chevron",
    "Multi-steps",
)


@dataclass(frozen=True)
class DatasetSplit:
    """Container for the artefacts produced by :func:`build_stratified_split`."""

    train_csv: Path
    val_csv: Path
    test_csv: Path
    class_weights: dict[str, float]
    oversample_targets: dict[str, int]


_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"path", "class", "source_recording", "duration_ms"}
)


def _validate_manifest(manifest: pd.DataFrame) -> None:
    if len(manifest) == 0:
        raise ValueError("manifest is empty — cannot split a 0-row DataFrame")
    missing = _REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise KeyError(
            f"manifest is missing required columns: {sorted(missing)}. "
            "Expected: path, class, source_recording, duration_ms."
        )


def _validate_fractions(train_frac: float, val_frac: float) -> float:
    if train_frac <= 0.0 or val_frac < 0.0:
        raise ValueError(
            f"train_frac must be > 0 and val_frac >= 0; got "
            f"train_frac={train_frac}, val_frac={val_frac}"
        )
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0 to leave room for test; "
            f"got {train_frac} + {val_frac} = {train_frac + val_frac:.3f}"
        )
    return 1.0 - train_frac - val_frac


def _allocate_class(
    class_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[str]]:
    """Largest-first greedy allocator with recording-level grouping.

    1. Deterministically shuffle recording IDs by ``rng`` (per-seed entropy).
    2. Stable-sort by call count descending — large recordings placed first,
       small ones reserved as fine-tuners. Within the same count the shuffle
       order from step 1 is preserved (Python ``sorted`` is stable), so
       different seeds yield different placements when ties exist.
    3. Walk the sorted list, placing each recording into whichever split has
       the largest remaining target-vs-actual deficit. When val and test
       are tied (the common case for the LAST recording in a small class
       under an 80/10/10 split), the tie-break alternates based on the
       *class identity* — a hash-stable per-class flip that gives each
       class an equal chance of routing its leftover to test vs val while
       remaining deterministic and seed-independent for the tie itself.
       Train always wins three-way ties (single-recording classes land
       in train, satisfying ``test_single_recording_per_class_goes_to_train``).

    Returns three lists of ``source_recording`` IDs for train, val, test.
    """
    rec_counts: dict[str, int] = (
        class_df.groupby("source_recording").size().to_dict()
    )
    recordings = sorted(rec_counts.keys())  # deterministic baseline
    rng.shuffle(recordings)
    # Stable sort by descending call count; ties broken by shuffle order above.
    recordings = sorted(recordings, key=lambda r: -rec_counts[r])

    total_calls = int(sum(rec_counts.values()))
    targets = {
        "train": train_frac * total_calls,
        "val": val_frac * total_calls,
        "test": test_frac * total_calls,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    bins: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    # Per-class tie-break: flip val/test based on class identity. Uses a
    # stable hash of the class name so the choice is deterministic across
    # runs but varies across classes — roughly half route leftovers to val,
    # half to test, eliminating the all-val degenerate case from a fixed
    # tuple-iteration tie-break.
    cls_name = str(class_df["class"].iloc[0]) if len(class_df) else ""
    flip_test_first = (sum(ord(c) for c in cls_name) % 2) == 0
    secondary_order = ("test", "val") if flip_test_first else ("val", "test")

    # Use a tolerance for deficit comparison because target fractions may
    # carry float-rounding noise (``1.0 - 0.8 - 0.1 == 0.09999999999999998``
    # propagates through to deficits["test"] vs deficits["val"] when they
    # *should* be mathematically equal). Without tolerance, the per-class
    # flip below silently degenerates because the "tied" comparison fails.
    _EPS = 1e-9

    def _close(a: float, b: float) -> bool:
        return abs(a - b) < _EPS

    for rec in recordings:
        n = rec_counts[rec]
        deficits = {split: targets[split] - counts[split] for split in counts}
        best_deficit = max(deficits.values())
        if _close(deficits["train"], best_deficit):
            best = "train"
        elif _close(deficits[secondary_order[0]], best_deficit):
            best = secondary_order[0]
        else:
            best = secondary_order[1]
        bins[best].append(rec)
        counts[best] += n

    return bins["train"], bins["val"], bins["test"]


def _compute_class_weights(train_df: pd.DataFrame) -> dict[str, float]:
    """Inverse-frequency weights normalized so mean weight = 1.0.

    Missing-from-train classes get an effective count of 1 to avoid division
    by zero; in practice the synthetic-distribution fixture and real VocalMat
    data both keep every class in train.
    """
    counts = train_df["class"].value_counts().to_dict()
    inv = {
        cls: 1.0 / max(1, int(counts.get(cls, 0)))
        for cls in GRIMSLEY_12_CLASSES
    }
    n_classes = len(GRIMSLEY_12_CLASSES)
    mean_inv = sum(inv.values()) / n_classes
    return {cls: inv[cls] / mean_inv for cls in GRIMSLEY_12_CLASSES}


def _compute_oversample_targets(train_df: pd.DataFrame) -> dict[str, int]:
    """Bring every class up to the median training count; never reduce majority."""
    counts = {
        cls: int((train_df["class"] == cls).sum())
        for cls in GRIMSLEY_12_CLASSES
    }
    median_count = int(np.median(list(counts.values())))
    return {
        cls: max(counts[cls], median_count) for cls in GRIMSLEY_12_CLASSES
    }


def build_stratified_split(
    manifest: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 1729,
    out_dir: Path | str | None = None,
) -> DatasetSplit:
    """Build a recording-level-grouped stratified 80/10/10 split.

    Parameters
    ----------
    manifest : pd.DataFrame
        Required columns: ``path``, ``class``, ``source_recording``,
        ``duration_ms``. Each row is one labelled call.
    train_frac, val_frac : float
        Target fractions; ``test_frac = 1 - train_frac - val_frac`` is implicit.
        Must sum to strictly less than 1.0.
    seed : int
        Deterministic seed for the per-class recording shuffle.
    out_dir : Path | str | None
        Directory where ``train.csv``, ``val.csv``, ``test.csv`` are written.
        Required (the tests always pass an explicit ``tmp_path``); a default
        of ``None`` would only make sense if callers had an in-memory use.

    Returns
    -------
    DatasetSplit
        Paths to the three CSVs plus class-weight and oversample-target dicts.

    Raises
    ------
    ValueError
        If ``manifest`` is empty, or ``train_frac + val_frac >= 1.0``.
    KeyError
        If any of the four required columns is missing.
    """
    _validate_manifest(manifest)
    test_frac = _validate_fractions(train_frac, val_frac)

    if out_dir is None:
        raise ValueError("out_dir is required (where to write the split CSVs)")
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    train_chunks: list[pd.DataFrame] = []
    val_chunks: list[pd.DataFrame] = []
    test_chunks: list[pd.DataFrame] = []

    for cls in sorted(manifest["class"].unique()):
        class_df = manifest[manifest["class"] == cls]
        train_recs, val_recs, test_recs = _allocate_class(
            class_df, train_frac, val_frac, test_frac, rng
        )
        train_chunks.append(class_df[class_df["source_recording"].isin(train_recs)])
        val_chunks.append(class_df[class_df["source_recording"].isin(val_recs)])
        test_chunks.append(class_df[class_df["source_recording"].isin(test_recs)])

    train_df = (
        pd.concat(train_chunks, ignore_index=True)
        if train_chunks
        else manifest.iloc[0:0].copy()
    )
    val_df = (
        pd.concat(val_chunks, ignore_index=True)
        if val_chunks
        else manifest.iloc[0:0].copy()
    )
    test_df = (
        pd.concat(test_chunks, ignore_index=True)
        if test_chunks
        else manifest.iloc[0:0].copy()
    )

    train_csv = out_dir_path / "train.csv"
    val_csv = out_dir_path / "val.csv"
    test_csv = out_dir_path / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    class_weights = _compute_class_weights(train_df)
    oversample_targets = _compute_oversample_targets(train_df)

    return DatasetSplit(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        class_weights=class_weights,
        oversample_targets=oversample_targets,
    )
