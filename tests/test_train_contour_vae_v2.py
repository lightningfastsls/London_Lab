"""Regression tests for ``scripts/train_contour_vae_v2.py``.

Anchors §1a of ``docs/handoffs/2026-05-20_path-c-cleanup.md``: the
latents export must produce one row per encoded patch, even when the
manifest's ``patch_idx`` column is not globally unique (which is true
of any combined-cohort manifest, where each cohort restarts patch_idx
at 0).

The original bug merged on ``patch_idx``, cross-joining duplicates and
silently scrambling the (z ↔ wav_stem) mapping. On the combined 4-cohort
run this turned 69,293 manifest rows into 98,945 latents rows.
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

from scripts.train_contour_vae_v2 import _build_latents_df  # noqa: E402


def _make_manifest_with_duplicate_patch_idx() -> pd.DataFrame:
    """Build a tiny combined-style manifest with per-cohort patch_idx
    that collides across cohorts (0,1,2 ; 0,1 ; 0,1,2,3).

    This mirrors the exact shape of bug §1a: the manifest is a
    pd.concat() of three per-cohort frames, each restarting patch_idx
    at 0. Without the §1a fix, merge-on-patch_idx cross-joins.
    """
    frames = [
        pd.DataFrame({
            "patch_idx": np.arange(3, dtype=np.int64),
            "wav_stem": [f"cohortA_wav_{i}" for i in range(3)],
            "call_id": [f"cohortA_call_{i}" for i in range(3)],
            "window_idx": np.zeros(3, dtype=np.int64),
            "cohort": ["A"] * 3,
        }),
        pd.DataFrame({
            "patch_idx": np.arange(2, dtype=np.int64),
            "wav_stem": [f"cohortB_wav_{i}" for i in range(2)],
            "call_id": [f"cohortB_call_{i}" for i in range(2)],
            "window_idx": np.zeros(2, dtype=np.int64),
            "cohort": ["B"] * 2,
        }),
        pd.DataFrame({
            "patch_idx": np.arange(4, dtype=np.int64),
            "wav_stem": [f"cohortC_wav_{i}" for i in range(4)],
            "call_id": [f"cohortC_call_{i}" for i in range(4)],
            "window_idx": np.zeros(4, dtype=np.int64),
            "cohort": ["C"] * 4,
        }),
    ]
    return pd.concat(frames, ignore_index=True)


def test_latents_row_count_matches_manifest_with_duplicate_patch_idx() -> None:
    """Latents row count must equal manifest row count, even when
    cohort-relative patch_idx repeats across cohorts."""
    manifest = _make_manifest_with_duplicate_patch_idx()
    n = len(manifest)
    assert n == 9
    # Sanity: pre-fix code would have cross-joined here.
    assert manifest["patch_idx"].nunique() < n

    latent_dim = 4
    z_all = np.arange(n * latent_dim, dtype=np.float32).reshape(n, latent_dim)

    out = _build_latents_df(z_all, manifest, latent_dim)

    assert len(out) == n
    # New patch_idx is globally unique and ordinal.
    assert out["patch_idx"].nunique() == n
    assert out["patch_idx"].tolist() == list(range(n))


def test_latents_z_columns_match_positional_input() -> None:
    """Each row's z_k values must equal z_all[i, k] for the matching
    manifest row — proves there's no cross-join scrambling."""
    manifest = _make_manifest_with_duplicate_patch_idx()
    n = len(manifest)
    latent_dim = 3
    z_all = np.arange(n * latent_dim, dtype=np.float32).reshape(n, latent_dim)

    out = _build_latents_df(z_all, manifest, latent_dim)

    for i in range(n):
        for k in range(latent_dim):
            assert out.iloc[i][f"z_{k}"] == z_all[i, k], (
                f"z mismatch at row {i}, dim {k}: "
                f"latents={out.iloc[i][f'z_{k}']} vs z_all={z_all[i, k]}"
            )
        # wav_stem at row i must match manifest row i — the (z ↔ wav_stem)
        # mapping the original bug scrambled.
        assert out.iloc[i]["wav_stem"] == manifest.iloc[i]["wav_stem"]
        assert out.iloc[i]["cohort"] == manifest.iloc[i]["cohort"]


def test_latents_robust_to_non_trivial_dataframe_index() -> None:
    """A manifest whose pandas index isn't 0..N-1 (e.g. produced by
    pd.concat without ignore_index, or by filtering) must still produce
    a correctly-aligned latents frame. .reset_index(drop=True) on the
    column reads is what guarantees this."""
    manifest = _make_manifest_with_duplicate_patch_idx()
    # Inject a non-trivial DataFrame index.
    manifest.index = pd.Index([100 + 7 * i for i in range(len(manifest))])

    n = len(manifest)
    latent_dim = 2
    z_all = np.arange(n * latent_dim, dtype=np.float32).reshape(n, latent_dim)

    out = _build_latents_df(z_all, manifest, latent_dim)

    assert len(out) == n
    # Ordinal alignment preserved — row 0 of latents ↔ first row of manifest.
    assert out.iloc[0]["wav_stem"] == "cohortA_wav_0"
    assert out.iloc[-1]["wav_stem"] == "cohortC_wav_3"


def test_latents_without_cohort_column() -> None:
    """Single-cohort manifests have no ``cohort`` column. The helper must
    not crash and must not invent one."""
    manifest = pd.DataFrame({
        "patch_idx": np.arange(5, dtype=np.int64),
        "wav_stem": [f"wav_{i}" for i in range(5)],
        "call_id": [f"call_{i}" for i in range(5)],
        "window_idx": np.zeros(5, dtype=np.int64),
    })
    z_all = np.zeros((5, 2), dtype=np.float32)

    out = _build_latents_df(z_all, manifest, latent_dim=2)

    assert len(out) == 5
    assert "cohort" not in out.columns


def test_latents_rejects_length_mismatch() -> None:
    """If z_all and manifest disagree on row count, the helper must
    raise rather than silently truncate or recycle."""
    manifest = pd.DataFrame({
        "patch_idx": np.arange(5, dtype=np.int64),
        "wav_stem": [f"wav_{i}" for i in range(5)],
        "call_id": [f"call_{i}" for i in range(5)],
        "window_idx": np.zeros(5, dtype=np.int64),
    })
    z_all = np.zeros((4, 2), dtype=np.float32)  # one short

    with pytest.raises(ValueError, match="length"):
        _build_latents_df(z_all, manifest, latent_dim=2)
