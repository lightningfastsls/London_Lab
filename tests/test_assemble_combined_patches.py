"""Regression tests for ``scripts/assemble_combined_patches.py``.

Anchors §1b of ``docs/handoffs/2026-05-20_path-c-cleanup.md``: after
concatenating per-cohort manifests (each restarting ``patch_idx`` at 0),
the combined manifest must carry a globally unique ``patch_idx`` and
preserve the cohort-relative id as ``patch_idx_per_cohort``.

Without that re-numbering, the downstream latents export in
``train_contour_vae_v2.py`` cross-joins on ``patch_idx`` and silently
scrambles the (z ↔ wav_stem) mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.assemble_combined_patches import main as assemble_main  # noqa: E402


def _write_fake_cohort(
    out_dir: Path, n_rows: int, cohort_label: str, patch_shape=(4, 5)
) -> None:
    """Create a minimal cohort dir with patches.npz + manifest matching
    the schema mass_apply_contour_mask.py produces.

    Each cohort starts patch_idx at 0 — the same condition the real
    pipeline hits, and the one §1b must handle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(cohort_label) & 0xFFFFFFFF)
    patches = rng.standard_normal((n_rows, *patch_shape)).astype(np.float32)
    freqs_kHz = np.linspace(20.0, 120.0, patch_shape[0], dtype=np.float32)
    np.savez_compressed(out_dir / "patches.npz", patches=patches, freqs_kHz=freqs_kHz)
    manifest = pd.DataFrame({
        "patch_idx": np.arange(n_rows, dtype=np.int64),
        "wav_stem": [f"{cohort_label}_wav_{i}" for i in range(n_rows)],
        "call_id": [f"{cohort_label}_call_{i}" for i in range(n_rows)],
        "window_idx": np.zeros(n_rows, dtype=np.int64),
    })
    manifest.to_parquet(out_dir / "patches_manifest.parquet", index=False)


def test_combined_manifest_patch_idx_globally_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After assembling two cohorts whose per-cohort patch_idx ranges
    overlap, the combined manifest must have a globally unique patch_idx."""
    cohort_a = tmp_path / "cohort_a"
    cohort_b = tmp_path / "cohort_b"
    _write_fake_cohort(cohort_a, n_rows=3, cohort_label="a")
    _write_fake_cohort(cohort_b, n_rows=4, cohort_label="b")

    out_dir = tmp_path / "combined"

    monkeypatch.setattr(sys, "argv", [
        "assemble_combined_patches.py",
        "--cohort-dirs", str(cohort_a), str(cohort_b),
        "--cohort-names", "a", "b",
        "--output-dir", str(out_dir),
    ])

    rc = assemble_main()
    assert rc == 0, "assemble_main() returned non-zero exit code"

    mf = pd.read_parquet(out_dir / "patches_manifest.parquet")

    # Same total row count as the per-cohort sum.
    assert len(mf) == 7

    # Globally unique patch_idx — the core invariant §1b enforces.
    assert mf["patch_idx"].nunique() == len(mf), (
        f"patch_idx collisions: {len(mf)} rows but only "
        f"{mf['patch_idx'].nunique()} unique values"
    )
    assert mf["patch_idx"].tolist() == list(range(7))

    # Cohort-relative id is preserved (0,1,2 then 0,1,2,3).
    assert "patch_idx_per_cohort" in mf.columns
    assert mf["patch_idx_per_cohort"].tolist() == [0, 1, 2, 0, 1, 2, 3]

    # Cohort label is present and partitions correctly.
    assert mf["cohort"].tolist() == ["a", "a", "a", "b", "b", "b", "b"]


def test_combined_patches_npz_row_count_matches_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The combined patches.npz must have the same axis-0 size as the
    combined manifest. Guards against a regression where streaming
    concat drops rows or duplicates them."""
    cohort_a = tmp_path / "cohort_a"
    cohort_b = tmp_path / "cohort_b"
    _write_fake_cohort(cohort_a, n_rows=5, cohort_label="a")
    _write_fake_cohort(cohort_b, n_rows=2, cohort_label="b")

    out_dir = tmp_path / "combined"

    monkeypatch.setattr(sys, "argv", [
        "assemble_combined_patches.py",
        "--cohort-dirs", str(cohort_a), str(cohort_b),
        "--cohort-names", "a", "b",
        "--output-dir", str(out_dir),
    ])

    assert assemble_main() == 0

    mf = pd.read_parquet(out_dir / "patches_manifest.parquet")
    with np.load(out_dir / "patches.npz", mmap_mode="r") as data:
        assert data["patches"].shape[0] == len(mf)
        assert data["patches"].shape[0] == 7
