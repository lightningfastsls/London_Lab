"""Tests for analyze_latent_dispersion — written by test-architect BEFORE implementation.

ROADMAP test plan coverage:
  1. U-statistic semantics, toy 3-point case -> test_mean_pairwise_euclidean_toy
  2. Single-row Z raises ValueError              -> test_mean_pairwise_euclidean_single_row_raises_value_error
  3. Equal-N subsample + seed reproducibility    -> test_subsample_per_cohort_exact_n_and_seed_reproducibility
  4. ValueError when cohort below requested N    -> test_subsample_raises_when_cohort_too_small
  5. Bootstrap CI shape, ordering, reproducibility -> test_bootstrap_ci_brackets_point_and_has_correct_shape
  6. Real-data schema / cohort counts            -> test_compute_cohort_dispersion_real_data_shape
  7. Seed reproducibility on synthetic data      -> test_compute_cohort_dispersion_seed_reproducibility

Additional coverage (recurring gap patterns):
  - Empty input (Z with 0 rows) raises cleanly  -> test_mean_pairwise_euclidean_empty_raises
  - Two-row Z returns the single distance exactly -> test_mean_pairwise_euclidean_two_rows_returns_exact_distance
  - subsample output rows are sampled WITHOUT replacement -> test_subsample_without_replacement
  - load_latents returns correct columns and dtypes -> test_load_latents_columns_and_dtypes

Total: 11 tests (7 from ROADMAP, 4 additional)

Decision pinned in spec:
  - test_mean_pairwise_euclidean_single_row_raises_value_error: spec offered ValueError OR 0.0;
    we pin ValueError because N=1 has no pairs and 0.0 would silently mask bugs in callers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is importable when running from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_latent_dispersion import (  # noqa: E402  # Will pass after module is created
    bootstrap_dispersion_ci,
    compute_cohort_dispersion,
    load_latents,
    mean_pairwise_euclidean,
    subsample_per_cohort,
)

# ---------------------------------------------------------------------------
# Path to the real parquet (symlinked in the worktree)
# ---------------------------------------------------------------------------
LATENTS_PARQUET = (
    Path(__file__).parent.parent / "results" / "contour_vae_combined" / "latents.parquet"
)

EXPECTED_COHORT_COUNTS = {
    "3452": 406,
    "9252": 584,
    "5970": 12440,
    "lab_131204": 55863,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(
    n_per_cohort: int = 1000,
    n_dims: int = 32,
    cohorts: tuple[str, ...] = ("A", "B", "C", "D"),
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic DataFrame that matches the latents parquet schema."""
    rng = np.random.default_rng(seed)
    rows = []
    call_id = 0
    for coh in cohorts:
        Z = rng.standard_normal((n_per_cohort, n_dims)).astype(np.float32)
        for i in range(n_per_cohort):
            row = {f"z_{d}": float(Z[i, d]) for d in range(n_dims)}
            row["cohort"] = coh
            row["wav_stem"] = f"{coh}_recording_{i // 100}"
            row["call_id"] = call_id
            row["window_idx"] = i % 10
            row["patch_idx"] = i
            call_id += 1
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# 1. mean_pairwise_euclidean — toy hand-computed case
# ===========================================================================

def test_mean_pairwise_euclidean_toy():
    """Spec: U-statistic over unique pairs, no self-pairs, no double-counting.

    Z = [[0,0],[3,4],[6,8]]
    Pairs:
      (0,1) dist = sqrt(9+16)  = 5
      (0,2) dist = sqrt(36+64) = 10
      (1,2) dist = sqrt(9+16)  = 5
    Mean = (5 + 10 + 5) / 3 = 20/3 ≈ 6.6666...
    """
    Z = np.array([[0, 0], [3, 4], [6, 8]], dtype=np.float64)
    result = mean_pairwise_euclidean(Z)
    expected = 20.0 / 3.0
    assert abs(result - expected) < 1e-9, (
        f"Expected 20/3={expected:.10f}, got {result:.10f}. "
        "Check that self-pairs are excluded and each pair counted once."
    )


# ===========================================================================
# 2. mean_pairwise_euclidean — single row → ValueError
# ===========================================================================

def test_mean_pairwise_euclidean_single_row_raises_value_error():
    """Spec: single-row Z has no pairs; function must raise ValueError.

    Pinned contract: ValueError (not 0.0), because 0.0 would silently mislead callers.
    """
    Z = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        mean_pairwise_euclidean(Z)


# ===========================================================================
# Additional: empty input raises (not AttributeError, not NaN)
# ===========================================================================

def test_mean_pairwise_euclidean_empty_raises():
    """Empty Z (0 rows) must raise ValueError; no silent NaN/inf."""
    Z = np.empty((0, 4), dtype=np.float64)
    with pytest.raises((ValueError, IndexError)):
        mean_pairwise_euclidean(Z)


# ===========================================================================
# Additional: two-row Z returns the single pairwise distance exactly
# ===========================================================================

def test_mean_pairwise_euclidean_two_rows_returns_exact_distance():
    """Two-point Z: the one pair's distance IS the mean. Verifies no averaging bug."""
    # dist([0,0,0], [3,4,0]) = 5.0
    Z = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    result = mean_pairwise_euclidean(Z)
    assert abs(result - 5.0) < 1e-9, f"Expected 5.0, got {result}"


# ===========================================================================
# 3. subsample_per_cohort — exact N, seed reproducibility
# ===========================================================================

def test_subsample_per_cohort_exact_n_and_seed_reproducibility():
    """Spec: equal-N sample; same seed → identical content; different seed → different rows."""
    df = _make_synthetic_df(n_per_cohort=1000, n_dims=4, cohorts=("X", "Y", "Z"), seed=0)
    n = 200

    result1 = subsample_per_cohort(df, n_per_cohort=n, seed=42)
    result2 = subsample_per_cohort(df, n_per_cohort=n, seed=42)
    result3 = subsample_per_cohort(df, n_per_cohort=n, seed=99)

    # Correct row count
    assert len(result1) == n * 3, f"Expected {n * 3} rows, got {len(result1)}"

    # All cohorts present
    assert set(result1["cohort"].unique()) == {"X", "Y", "Z"}

    # Each cohort has exactly n rows
    for coh in ("X", "Y", "Z"):
        count = (result1["cohort"] == coh).sum()
        assert count == n, f"Cohort {coh!r}: expected {n} rows, got {count}"

    # Same seed → identical selection (reset index before comparing)
    pd.testing.assert_frame_equal(
        result1.reset_index(drop=True),
        result2.reset_index(drop=True),
        check_like=False,
    )

    # Different seed → different rows (probabilistically certain with n=200 from 1000)
    assert not result1.reset_index(drop=True).equals(result3.reset_index(drop=True)), (
        "seed=42 and seed=99 produced identical output — seed is not being used"
    )


# ===========================================================================
# Additional: sampling is WITHOUT replacement per cohort
# ===========================================================================

def test_subsample_without_replacement():
    """Each cohort subsampled row must appear at most once (no duplicates)."""
    df = _make_synthetic_df(n_per_cohort=500, n_dims=4, cohorts=("P", "Q"), seed=7)
    result = subsample_per_cohort(df, n_per_cohort=300, seed=0)
    for coh in ("P", "Q"):
        sub = result[result["cohort"] == coh]
        assert sub["patch_idx"].nunique() == len(sub), (
            f"Duplicate rows in cohort {coh!r} — sampling may be with replacement"
        )


# ===========================================================================
# 4. subsample_per_cohort — ValueError when cohort is too small
# ===========================================================================

def test_subsample_raises_when_cohort_too_small():
    """Spec: requesting N > cohort size must raise ValueError (mentions offending cohort)."""
    big = pd.DataFrame({
        "cohort": ["A"] * 500 + ["B"] * 100,
        "patch_idx": range(600),
        "z_0": np.zeros(600),
    })
    with pytest.raises(ValueError, match="B"):
        subsample_per_cohort(big, n_per_cohort=200, seed=0)


# ===========================================================================
# 5. bootstrap_dispersion_ci — structure, ordering, reproducibility
# ===========================================================================

def test_bootstrap_ci_brackets_point_and_has_correct_shape():
    """Spec: CI has correct keys; ci_lo < point < ci_hi; reps length = n_reps; reproducible."""
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((200, 8)).astype(np.float32)

    result = bootstrap_dispersion_ci(Z, n_reps=500, seed=42, ci_pct=95.0)

    # Required keys
    assert set(result.keys()) == {"point", "ci_lo", "ci_hi", "reps"}, (
        f"Unexpected keys: {set(result.keys())}"
    )

    # reps length
    assert len(result["reps"]) == 500, f"Expected 500 reps, got {len(result['reps'])}"
    assert isinstance(result["reps"], np.ndarray)

    # point is mean_pairwise_euclidean on full Z
    expected_point = mean_pairwise_euclidean(Z)
    assert abs(result["point"] - expected_point) < 1e-6, (
        f"point={result['point']:.6f} but mean_pairwise_euclidean(Z)={expected_point:.6f}"
    )

    # CI ordering: 0 < ci_lo < point < ci_hi
    assert result["ci_lo"] > 0, f"ci_lo={result['ci_lo']} must be positive"
    assert result["ci_lo"] < result["point"], (
        f"ci_lo={result['ci_lo']} must be < point={result['point']}"
    )
    assert result["point"] < result["ci_hi"], (
        f"point={result['point']} must be < ci_hi={result['ci_hi']}"
    )

    # Reproducibility: same seed → same reps
    result2 = bootstrap_dispersion_ci(Z, n_reps=500, seed=42, ci_pct=95.0)
    np.testing.assert_array_equal(result["reps"], result2["reps"])


# ===========================================================================
# 6. compute_cohort_dispersion — real data shape and cohort counts
# ===========================================================================

def test_compute_cohort_dispersion_real_data_shape():
    """Spec: real parquet → 4 cohort rows, correct schema, n_subsampled=400 for all."""
    if not LATENTS_PARQUET.exists():
        pytest.skip(f"Real parquet not found at {LATENTS_PARQUET}")

    df = load_latents(str(LATENTS_PARQUET))
    out = compute_cohort_dispersion(df, n_per_cohort=400, n_boot=50, seed=42)

    # Four rows, one per cohort
    assert len(out) == 4, f"Expected 4 rows, got {len(out)}"

    # Required columns
    required_cols = {"cohort", "n_in_cohort", "n_subsampled", "point", "ci_lo", "ci_hi"}
    missing = required_cols - set(out.columns)
    assert not missing, f"Missing columns: {missing}"

    # n_subsampled == 400 for every cohort (we requested 400)
    assert (out["n_subsampled"] == 400).all(), (
        f"Expected n_subsampled=400 for all cohorts:\n{out[['cohort','n_subsampled']]}"
    )

    # point estimates are positive floats
    assert (out["point"] > 0).all(), (
        f"All point estimates must be positive:\n{out[['cohort','point']]}"
    )

    # n_in_cohort matches known counts
    cohort_counts = out.set_index("cohort")["n_in_cohort"].to_dict()
    for coh, expected in EXPECTED_COHORT_COUNTS.items():
        assert coh in cohort_counts, f"Cohort {coh!r} missing from output"
        assert cohort_counts[coh] == expected, (
            f"Cohort {coh!r}: expected n_in_cohort={expected}, got {cohort_counts[coh]}"
        )

    # n_boot and seed columns present and correct
    if "n_boot" in out.columns:
        assert (out["n_boot"] == 50).all()
    if "seed" in out.columns:
        assert (out["seed"] == 42).all()


# ===========================================================================
# 7. compute_cohort_dispersion — seed reproducibility on synthetic data
# ===========================================================================

def test_compute_cohort_dispersion_seed_reproducibility():
    """Spec: identical seed → identical output DataFrame (pd.testing.assert_frame_equal)."""
    df = _make_synthetic_df(n_per_cohort=1000, n_dims=32, cohorts=("A", "B", "C", "D"), seed=7)

    out1 = compute_cohort_dispersion(df, n_per_cohort=400, n_boot=20, seed=42)
    out2 = compute_cohort_dispersion(df, n_per_cohort=400, n_boot=20, seed=42)

    pd.testing.assert_frame_equal(
        out1.reset_index(drop=True),
        out2.reset_index(drop=True),
        check_like=True,   # allow column reordering
    )


# ===========================================================================
# Additional: load_latents — correct columns and float32 latent dtype
# ===========================================================================

def test_load_latents_columns_and_dtypes():
    """load_latents must return z_0..z_31, cohort, wav_stem, call_id columns."""
    if not LATENTS_PARQUET.exists():
        pytest.skip(f"Real parquet not found at {LATENTS_PARQUET}")

    df = load_latents(str(LATENTS_PARQUET))

    # All 32 latent dims present
    latent_cols = [f"z_{i}" for i in range(32)]
    missing = [c for c in latent_cols if c not in df.columns]
    assert not missing, f"Missing latent columns: {missing}"

    # Metadata columns present
    for col in ("cohort", "wav_stem", "call_id"):
        assert col in df.columns, f"Missing metadata column {col!r}"

    # Latent dims must be numeric (float16/32/64 all acceptable)
    for col in latent_cols:
        assert pd.api.types.is_float_dtype(df[col]), (
            f"Column {col!r} has dtype {df[col].dtype}, expected float"
        )

    # Row count matches known parquet size
    assert len(df) == 69293, f"Expected 69293 rows, got {len(df)}"
