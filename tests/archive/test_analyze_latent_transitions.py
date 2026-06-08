"""Tests for analyze_latent_transitions — written by test-architect BEFORE implementation.

ROADMAP test plan coverage:
  1. load_call_latents returns one row per call on real data, all columns present,
     timestamps finite, all 5 cohort_split categories present
     -> test_load_call_latents_one_row_per_call_real_data

  2. load_call_latents mean_z correct on synthetic mini-parquet (hand-computed values)
     -> test_load_call_latents_mean_z_correct_on_synthetic

  3. assign_call_clusters returns expected IDs for toy KMeans with known centroids
     -> test_assign_call_clusters_toy_kmeans

  4. segment_into_bouts basic: 6 calls, 4 bouts for threshold=0.25s
     -> test_segment_into_bouts_basic

  5. segment_into_bouts file-aware: two wav_stems never merged despite huge threshold
     -> test_segment_into_bouts_file_aware

  6. segment_into_bouts unsorted input is sorted within each wav_stem
     -> test_segment_into_bouts_unsorted_input_is_sorted

  7. build_transition_matrix toy: exact bigram counts matched to hand computation
     -> test_build_transition_matrix_toy

  8. build_transition_matrix zero-row fallback: uniform row for never-source cluster
     -> test_build_transition_matrix_zero_row_fallback

  9. entropy_rate_from_matrix uniform K=4: H = log2(4) = 2.0 bits
     -> test_entropy_rate_uniform_known_value

  10. entropy_rate_from_matrix identity matrix: H = 0 bits
      -> test_entropy_rate_deterministic_zero

  11. bootstrap_entropy_rate reproducibility: same seed → same results, diff seed → diff
      -> test_bootstrap_entropy_rate_reproducibility

  12. detect_idioms stationary iid sequences: < 2 idioms flagged (99th pct threshold)
      -> test_detect_idioms_stationary_no_idioms

  13. detect_idioms injected (0,1) bigram pattern is recovered as idiom
      -> test_detect_idioms_injected_pattern_recovers

  14. one symbol per call enforcement: bout sequence length = n_calls, not n_patches
      -> test_one_symbol_per_call_enforcement

  15. detect_idioms output schema: correct column names and dtypes
      -> test_idioms_output_schema

Additional coverage (recurring gap patterns):
  - build_transition_matrix rows sum to 1.0
      -> test_build_transition_matrix_rows_sum_to_one

  - segment_into_bouts single call per wav_stem doesn't crash
      -> test_segment_into_bouts_single_call_per_stem

  - entropy_rate_from_matrix K=1 trivial case
      -> test_entropy_rate_k1_trivial

  - build_transition_matrix empty sequences list returns uniform matrix
      -> test_build_transition_matrix_empty_sequences_returns_uniform

  - load_call_latents: join rate assertion >= 99% (tested via synthetic drop scenario)
      -> test_load_call_latents_assert_join_rate

Total: 19 tests (15 from ROADMAP, 4 additional)

Decisions beyond spec:
  - test_build_transition_matrix_toy: verified hand-computation by walking each bigram
    explicitly. Seq1 = [0,1,2,0,1]: pairs (0→1), (1→2), (2→0), (0→1). Seq2 = [1,2,2]:
    pairs (1→2), (2→2). Row 0: (0→1)=2, total=2, P[0]=[0,1,0]. Row 1: (1→2)=2, total=2,
    P[1]=[0,0,1]. Row 2: (2→0)=1, (2→2)=1, total=2, P[2]=[0.5,0,0.5].
  - test_entropy_rate_from_matrix: spec says "pi=None → compute as left eigenvector".
    We pass pi explicitly (uniform) in the uniform test and pi=None in the deterministic
    test, exercising both code paths.
  - test_detect_idioms_stationary_no_idioms: 99th percentile with 200 shuffles can
    spuriously flag up to ~1% of 25 bigrams. We allow up to 2 (not just 0) to avoid
    test flakiness at the boundary. This is documented as a spec assumption.
  - synthetic parquet uses only 3 latent dims (z_0..z_2) to keep assertions readable,
    with n_latent_dims=3 explicitly noted. The real module must tolerate any n_z.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is importable when running from any cwd
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Will raise ImportError until implementation exists — expected before implementation
from scripts.analyze_latent_transitions import (  # noqa: E402
    load_call_latents,
    assign_call_clusters,
    segment_into_bouts,
    build_transition_matrix,
    entropy_rate_from_matrix,
    bootstrap_entropy_rate,
    detect_idioms,
    DETECTION_CSV_PATHS,
)

# ---------------------------------------------------------------------------
# Paths to real artifacts
# ---------------------------------------------------------------------------

LATENTS_PARQUET = REPO_ROOT / "results" / "contour_vae_combined" / "latents.parquet"

REAL_DETECTION_CSV_PATHS = {
    "5970": str(REPO_ROOT / "classified_detections_full.csv"),
    "3452": str(REPO_ROOT / "classified_detections_3452.csv"),
    "9252": str(REPO_ROOT / "classified_detections_9252.csv"),
    "lab_131204": str(REPO_ROOT / "classified_detections_lab_131204_clean.csv"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_parquet(tmp_path: Path, n_latent_dims: int = 3) -> tuple[Path, Path]:
    """Build a minimal parquet + detection CSV for deterministic unit tests.

    Three calls:
      Call A: wav_stem="stemA", call_id=1, 2 patches, z_0=[1,3], z_1=[2,4], z_2=[3,5]
              → mean z = [2.0, 3.0, 4.0], n_patches=2
      Call B: wav_stem="stemA", call_id=2, 1 patch,  z_0=[10], z_1=[10], z_2=[10]
              → mean z = [10.0, 10.0, 10.0], n_patches=1
      Call C: wav_stem="stemB", call_id=1, 4 patches, z_0=[0,0,0,0] etc.
              → mean z = [0.0, 0.0, 0.0], n_patches=4
    """
    lat_cols = {f"z_{i}": [] for i in range(n_latent_dims)}

    rows = []
    # Call A — patch 0
    rows.append({"cohort": "5970", "wav_stem": "stemA", "call_id": 1,
                 "window_idx": 0, "patch_idx": 0, "z_0": 1.0, "z_1": 2.0, "z_2": 3.0})
    # Call A — patch 1
    rows.append({"cohort": "5970", "wav_stem": "stemA", "call_id": 1,
                 "window_idx": 1, "patch_idx": 1, "z_0": 3.0, "z_1": 4.0, "z_2": 5.0})
    # Call B — patch 0
    rows.append({"cohort": "5970", "wav_stem": "stemA", "call_id": 2,
                 "window_idx": 0, "patch_idx": 2, "z_0": 10.0, "z_1": 10.0, "z_2": 10.0})
    # Call C — 4 patches all zeros
    for p in range(4):
        rows.append({"cohort": "5970", "wav_stem": "stemB", "call_id": 1,
                     "window_idx": p, "patch_idx": 3 + p, "z_0": 0.0, "z_1": 0.0, "z_2": 0.0})

    parquet_path = tmp_path / "latents.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    # Detection CSV: 3 rows with timestamps, joined on (wav_stem, id=call_id)
    det_rows = [
        {"wav_stem": "stemA", "id": 1, "begin_time_s": 0.10, "end_time_s": 0.15},
        {"wav_stem": "stemA", "id": 2, "begin_time_s": 0.20, "end_time_s": 0.25},
        {"wav_stem": "stemB", "id": 1, "begin_time_s": 1.00, "end_time_s": 1.10},
    ]
    det_path = tmp_path / "det_5970.csv"
    pd.DataFrame(det_rows).to_csv(det_path, index=False)

    return parquet_path, det_path


def _make_toy_kmeans(centroids: np.ndarray):
    """Build a minimal KMeans-like object with fixed predict logic using L2 distance."""
    from sklearn.cluster import KMeans
    import joblib

    # Fit a KMeans with 1 sample per cluster to pin centroids exactly
    # n_init=1, max_iter=1, then force-set cluster_centers_
    k = len(centroids)
    km = KMeans(n_clusters=k, n_init=1, max_iter=1, random_state=0)
    # We can't set centers directly without fitting; instead use a trick:
    # Create one point per centroid and fit
    km.fit(centroids)
    km.cluster_centers_ = centroids.astype(float)
    return km


# ===========================================================================
# 1. load_call_latents — real data
# ===========================================================================

@pytest.mark.skipif(
    not LATENTS_PARQUET.exists(),
    reason="Real latents.parquet not available in this environment",
)
def test_load_call_latents_one_row_per_call_real_data():
    """Spec: load_call_latents returns one row per unique (wav_stem, call_id).

    Verifies:
    - n_rows ≈ 48917 (allow 1% drop from unmatched joins)
    - All required columns present
    - mean_z_0..mean_z_31 are float
    - begin_time_s is finite for > 99% of rows
    - All 5 cohort_split categories appear (5970, 3452, 9252, lab_matched, lab_swap)
    """
    result = load_call_latents(str(LATENTS_PARQUET), REAL_DETECTION_CSV_PATHS)

    expected_cols = {
        "cohort", "cohort_split", "wav_stem", "call_id",
        "begin_time_s", "end_time_s", "n_patches",
    } | {f"mean_z_{i}" for i in range(32)}
    missing = expected_cols - set(result.columns)
    assert not missing, f"Missing columns: {missing}"

    # One row per (wav_stem, call_id)
    n_unique = result.groupby(["wav_stem", "call_id"]).ngroups
    assert n_unique == len(result), "Duplicate (wav_stem, call_id) pairs found"

    # Row count close to 48917
    assert len(result) >= 48917 * 0.99, (
        f"Row count {len(result)} is below 99% of expected 48917"
    )
    assert len(result) <= 48917 * 1.01, (
        f"Row count {len(result)} exceeds expected 48917 by more than 1%"
    )

    # Timestamps finite
    finite_frac = np.isfinite(result["begin_time_s"].values).mean()
    assert finite_frac >= 0.99, f"begin_time_s finite fraction {finite_frac:.3f} < 0.99"

    # mean_z columns are float
    for i in range(32):
        col = f"mean_z_{i}"
        assert pd.api.types.is_float_dtype(result[col]), f"{col} is not float dtype"

    # All 5 cohort_split categories
    split_vals = set(result["cohort_split"].unique())
    expected_splits = {"5970", "3452", "9252", "lab_matched", "lab_swap"}
    assert expected_splits.issubset(split_vals), (
        f"Missing cohort_split categories: {expected_splits - split_vals}"
    )


# ===========================================================================
# 2. load_call_latents — mean_z hand-computed on synthetic data
# ===========================================================================

def test_load_call_latents_mean_z_correct_on_synthetic(tmp_path):
    """Spec: mean_z per call = mean of z-vectors across that call's patches.

    Call A (2 patches): patch0=[1,2,3], patch1=[3,4,5] → mean=[2.0,3.0,4.0]
    Call B (1 patch):   patch0=[10,10,10]               → mean=[10.0,10.0,10.0]
    Call C (4 patches): all zeros                        → mean=[0.0,0.0,0.0]
    """
    parquet_path, det_path = _make_synthetic_parquet(tmp_path, n_latent_dims=3)
    det_paths = {"5970": str(det_path)}

    result = load_call_latents(str(parquet_path), det_paths)

    assert len(result) == 3, f"Expected 3 calls, got {len(result)}"

    # Identify rows by (wav_stem, call_id)
    call_a = result[(result["wav_stem"] == "stemA") & (result["call_id"] == 1)]
    call_b = result[(result["wav_stem"] == "stemA") & (result["call_id"] == 2)]
    call_c = result[(result["wav_stem"] == "stemB") & (result["call_id"] == 1)]

    assert len(call_a) == 1, "Expected exactly 1 row for call A"
    assert len(call_b) == 1, "Expected exactly 1 row for call B"
    assert len(call_c) == 1, "Expected exactly 1 row for call C"

    # Call A: mean_z_0 = (1+3)/2 = 2.0
    assert call_a.iloc[0]["mean_z_0"] == pytest.approx(2.0, abs=1e-9), (
        f"Call A mean_z_0={call_a.iloc[0]['mean_z_0']}, expected 2.0"
    )
    assert call_a.iloc[0]["mean_z_1"] == pytest.approx(3.0, abs=1e-9)
    assert call_a.iloc[0]["mean_z_2"] == pytest.approx(4.0, abs=1e-9)
    assert call_a.iloc[0]["n_patches"] == 2

    # Call B: single patch, mean = value itself
    assert call_b.iloc[0]["mean_z_0"] == pytest.approx(10.0, abs=1e-9), (
        f"Call B mean_z_0={call_b.iloc[0]['mean_z_0']}, expected 10.0"
    )
    assert call_b.iloc[0]["n_patches"] == 1

    # Call C: 4 patches all zeros
    assert call_c.iloc[0]["mean_z_0"] == pytest.approx(0.0, abs=1e-9)
    assert call_c.iloc[0]["n_patches"] == 4


# ===========================================================================
# 3. load_call_latents — join rate assertion (synthetic drop scenario)
# ===========================================================================

def test_load_call_latents_assert_join_rate(tmp_path):
    """Spec: load_call_latents asserts >= 99% join rate per cohort.

    Build a synthetic parquet/CSV where only 1 of 3 calls matches
    (33% join rate). Expect an AssertionError to be raised.
    """
    # Parquet has 3 calls for cohort "5970"
    rows = []
    for call_id in [1, 2, 3]:
        rows.append({
            "cohort": "5970", "wav_stem": "stemX", "call_id": call_id,
            "window_idx": 0, "patch_idx": call_id - 1,
            "z_0": 0.0, "z_1": 0.0, "z_2": 0.0,
        })
    parquet_path = tmp_path / "latents_drop.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    # Detection CSV only has call_id=1 (33% match rate)
    det_rows = [{"wav_stem": "stemX", "id": 1, "begin_time_s": 0.0, "end_time_s": 0.1}]
    det_path = tmp_path / "det_drop.csv"
    pd.DataFrame(det_rows).to_csv(det_path, index=False)

    with pytest.raises((AssertionError, ValueError)):
        load_call_latents(str(parquet_path), {"5970": str(det_path)})


# ===========================================================================
# 4. assign_call_clusters — toy KMeans with known centroids
# ===========================================================================

def test_assign_call_clusters_toy_kmeans():
    """Spec: assign_call_clusters predicts cluster via kmeans.predict on mean_z_* columns.

    Centroids: [[0,0], [10,0], [0,10]]
    Call z-vectors: [0,0]→0, [10,0]→1, [0,10]→2, [9,1]→1, [1,9]→2
    """
    from sklearn.cluster import KMeans

    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    km = KMeans(n_clusters=3, n_init=1, max_iter=300, random_state=0)
    km.fit(centroids)
    km.cluster_centers_ = centroids

    call_df = pd.DataFrame({
        "mean_z_0": [0.0, 10.0, 0.0,  9.0,  1.0],
        "mean_z_1": [0.0,  0.0, 10.0, 1.0,  9.0],
        "cohort": ["5970"] * 5,
        "cohort_split": ["5970"] * 5,
        "wav_stem": ["stemA"] * 5,
        "call_id": list(range(5)),
    })

    labels = assign_call_clusters(call_df, km)

    assert len(labels) == 5, f"Expected 5 labels, got {len(labels)}"
    assert labels[0] == 0, f"[0,0] → cluster 0, got {labels[0]}"
    assert labels[1] == 1, f"[10,0] → cluster 1, got {labels[1]}"
    assert labels[2] == 2, f"[0,10] → cluster 2, got {labels[2]}"
    assert labels[3] == 1, f"[9,1] → cluster 1 (closer to [10,0]), got {labels[3]}"
    assert labels[4] == 2, f"[1,9] → cluster 2 (closer to [0,10]), got {labels[4]}"


# ===========================================================================
# 5. segment_into_bouts — basic gap-based segmentation
# ===========================================================================

def test_segment_into_bouts_basic():
    """Spec: bouts split where ICI > bout_threshold_s. File-aware: each file new bout.

    6 calls in 1 wav_stem at begin_time_s = [0.0, 0.1, 0.5, 0.8, 2.0, 2.1]
    Gaps between consecutive calls: 0.1, 0.4, 0.3, 1.2, 0.1
    With bout_threshold=0.25: splits at gap 0.4 (after call 1), gap 0.3 (after call 2),
    gap 1.2 (after call 3).
    Expected 4 bouts:
      bout 0: calls at [0.0, 0.1] (2 calls)
      bout 1: call  at [0.5]      (1 call)
      bout 2: call  at [0.8]      (1 call)
      bout 3: calls at [2.0, 2.1] (2 calls)
    """
    df = pd.DataFrame({
        "cohort": ["5970"] * 6,
        "cohort_split": ["5970"] * 6,
        "wav_stem": ["stemA"] * 6,
        "call_id": list(range(1, 7)),
        "begin_time_s": [0.0, 0.1, 0.5, 0.8, 2.0, 2.1],
        "end_time_s":   [0.05, 0.15, 0.55, 0.85, 2.05, 2.15],
    })

    result = segment_into_bouts(df, bout_threshold_s=0.25)

    assert "bout_id" in result.columns, "bout_id column must be added"

    bout_ids = result["bout_id"].tolist()
    unique_bouts = result["bout_id"].unique()
    assert len(unique_bouts) == 4, (
        f"Expected 4 unique bout_ids, got {len(unique_bouts)}: {unique_bouts}"
    )

    # Verify lengths by grouping
    bout_sizes = result.groupby("bout_id").size().sort_index()
    sizes = sorted(bout_sizes.values.tolist())
    assert sizes == [1, 1, 2, 2], (
        f"Expected bout sizes [1,1,2,2], got {sizes}"
    )


# ===========================================================================
# 6. segment_into_bouts — file-aware (wav_stem boundary always splits)
# ===========================================================================

def test_segment_into_bouts_file_aware():
    """Spec: every wav_stem starts a new bout regardless of inter-call gap.

    4 calls across 2 stems with a huge threshold (10.0s) that would merge
    everything if not file-aware. Expect 2 unique bout_ids (one per stem).
    """
    df = pd.DataFrame({
        "cohort": ["5970"] * 4,
        "cohort_split": ["5970"] * 4,
        "wav_stem": ["stem1", "stem1", "stem2", "stem2"],
        "call_id": [1, 2, 1, 2],
        "begin_time_s": [0.0, 0.1, 0.05, 0.15],
        "end_time_s":   [0.05, 0.15, 0.10, 0.20],
    })

    result = segment_into_bouts(df, bout_threshold_s=10.0)

    unique_bouts = result["bout_id"].unique()
    assert len(unique_bouts) == 2, (
        f"Expected 2 unique bout_ids (one per stem), got {len(unique_bouts)}: {unique_bouts}"
    )

    # Verify each stem maps to exactly one bout_id
    for stem in ["stem1", "stem2"]:
        bouts_for_stem = result[result["wav_stem"] == stem]["bout_id"].unique()
        assert len(bouts_for_stem) == 1, (
            f"stem {stem!r} should have 1 bout, found {len(bouts_for_stem)}: {bouts_for_stem}"
        )


# ===========================================================================
# 7. segment_into_bouts — unsorted input is sorted within each wav_stem
# ===========================================================================

def test_segment_into_bouts_unsorted_input_is_sorted():
    """Spec: segment_into_bouts sorts by begin_time_s within each wav_stem.

    Input rows arrive out of time order. After segmentation, rows within each
    wav_stem must be in ascending begin_time_s order.
    """
    df = pd.DataFrame({
        "cohort": ["5970"] * 4,
        "cohort_split": ["5970"] * 4,
        "wav_stem": ["stemA", "stemA", "stemA", "stemA"],
        "call_id": [3, 1, 4, 2],
        # out of order: 0.8, 0.0, 1.0, 0.1
        "begin_time_s": [0.8, 0.0, 1.0, 0.1],
        "end_time_s":   [0.9, 0.05, 1.1, 0.2],
    })

    result = segment_into_bouts(df, bout_threshold_s=0.25)

    stem_a = result[result["wav_stem"] == "stemA"]["begin_time_s"].values
    assert list(stem_a) == sorted(stem_a.tolist()), (
        f"Rows within stemA not sorted by begin_time_s: {stem_a}"
    )


# ===========================================================================
# 8. segment_into_bouts — single call per wav_stem (no crash edge case)
# ===========================================================================

def test_segment_into_bouts_single_call_per_stem():
    """Recurring gap: single-item list operations shouldn't crash.

    One call per stem should produce two bouts (one each).
    """
    df = pd.DataFrame({
        "cohort": ["5970", "5970"],
        "cohort_split": ["5970", "5970"],
        "wav_stem": ["stemA", "stemB"],
        "call_id": [1, 1],
        "begin_time_s": [0.0, 5.0],
        "end_time_s":   [0.1, 5.1],
    })

    result = segment_into_bouts(df, bout_threshold_s=0.25)

    assert len(result) == 2
    assert result["bout_id"].nunique() == 2


# ===========================================================================
# 9. build_transition_matrix — toy hand-computed expected values
# ===========================================================================

def test_build_transition_matrix_toy():
    """Spec: build bigram counts from sequences, row-normalize.

    sequences = [[0,1,2,0,1], [1,2,2]], K=3

    Manual bigram tally:
      seq1: (0,1), (1,2), (2,0), (0,1)  →  0→1: 2, 1→2: 1, 2→0: 1
      seq2: (1,2), (2,2)                 →  1→2: 1, 2→2: 1

    Row totals: row0: 2, row1: 2, row2: 2
    P[0] = [0, 1, 0]
    P[1] = [0, 0, 1]
    P[2] = [0.5, 0, 0.5]
    """
    seqs = [np.array([0, 1, 2, 0, 1]), np.array([1, 2, 2])]
    P = build_transition_matrix(seqs, k=3)

    assert P.shape == (3, 3), f"Expected (3,3), got {P.shape}"

    np.testing.assert_allclose(P[0], [0.0, 1.0, 0.0], atol=1e-9,
                               err_msg="Row 0 wrong")
    np.testing.assert_allclose(P[1], [0.0, 0.0, 1.0], atol=1e-9,
                               err_msg="Row 1 wrong")
    np.testing.assert_allclose(P[2], [0.5, 0.0, 0.5], atol=1e-9,
                               err_msg="Row 2 wrong")


# ===========================================================================
# 10. build_transition_matrix — zero row fallback to uniform
# ===========================================================================

def test_build_transition_matrix_zero_row_fallback():
    """Spec: rows that never appear as source get uniform distribution 1/K.

    sequences = [[0,0,0]], K=3
    Row 0: (0,0)=2, total=2 → [1, 0, 0]
    Rows 1, 2: never source → fallback uniform [1/3, 1/3, 1/3]
    """
    seqs = [np.array([0, 0, 0])]
    P = build_transition_matrix(seqs, k=3)

    assert P.shape == (3, 3)
    np.testing.assert_allclose(P[0], [1.0, 0.0, 0.0], atol=1e-9,
                               err_msg="Row 0 should be [1,0,0]")
    np.testing.assert_allclose(P[1], [1/3, 1/3, 1/3], atol=1e-9,
                               err_msg="Row 1 should be uniform (never source)")
    np.testing.assert_allclose(P[2], [1/3, 1/3, 1/3], atol=1e-9,
                               err_msg="Row 2 should be uniform (never source)")


# ===========================================================================
# 11. build_transition_matrix — rows sum to 1.0
# ===========================================================================

def test_build_transition_matrix_rows_sum_to_one():
    """Recurring invariant: every row of a transition matrix sums to 1.0."""
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 8, size=50) for _ in range(10)]
    P = build_transition_matrix(seqs, k=8)

    row_sums = P.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(8), atol=1e-9,
                               err_msg="Not all rows sum to 1")


# ===========================================================================
# 12. build_transition_matrix — empty sequences list returns uniform matrix
# ===========================================================================

def test_build_transition_matrix_empty_sequences_returns_uniform():
    """Recurring gap: empty input should not crash; all rows fall back to uniform."""
    P = build_transition_matrix([], k=4)

    assert P.shape == (4, 4)
    np.testing.assert_allclose(P, np.full((4, 4), 1/4), atol=1e-9,
                               err_msg="Empty sequences should produce uniform matrix")


# ===========================================================================
# 13. entropy_rate_from_matrix — uniform K=4 → H = 2.0 bits (hand-computed)
# ===========================================================================

def test_entropy_rate_uniform_known_value():
    """Spec: H = -sum_i pi_i * sum_j P_ij * log2(P_ij).

    Uniform K=4: P[i,j] = 1/4 for all i,j; stationary pi[i] = 1/4.
    H = -(4 * 1/4) * 4 * (1/4 * log2(1/4))
      = -1 * (-4 * 1/4 * 2)    [since log2(1/4) = -2]
      = -1 * (-2) = 2.0 bits
    """
    K = 4
    P = np.full((K, K), 1.0 / K)
    pi = np.full(K, 1.0 / K)

    H = entropy_rate_from_matrix(P, pi=pi)

    assert H == pytest.approx(2.0, abs=1e-9), (
        f"Uniform K=4 transition matrix should give H=2.0 bits, got {H}"
    )


# ===========================================================================
# 14. entropy_rate_from_matrix — identity matrix → H = 0 bits
# ===========================================================================

def test_entropy_rate_deterministic_zero():
    """Spec: deterministic transitions (identity matrix) give H = 0.

    P = I_K: each state always transitions to itself.
    H = -sum_i pi_i * (1 * log2(1)) = 0  (since log2(1) = 0).
    pi=None triggers eigenvector computation path.
    """
    K = 5
    P = np.eye(K)

    # Pass pi=None to exercise the eigenvector code path
    H = entropy_rate_from_matrix(P, pi=None)

    assert H == pytest.approx(0.0, abs=1e-9), (
        f"Identity transition matrix should give H=0.0 bits, got {H}"
    )


# ===========================================================================
# 15. entropy_rate_from_matrix — K=1 trivial case
# ===========================================================================

def test_entropy_rate_k1_trivial():
    """Recurring gap: K=1 alphabet has entropy rate 0 (only one symbol)."""
    P = np.array([[1.0]])
    pi = np.array([1.0])

    H = entropy_rate_from_matrix(P, pi=pi)

    assert H == pytest.approx(0.0, abs=1e-9), (
        f"K=1 should give H=0.0 bits, got {H}"
    )


# ===========================================================================
# 16. bootstrap_entropy_rate — reproducibility
# ===========================================================================

def test_bootstrap_entropy_rate_reproducibility():
    """Spec: same seed → identical results; different seed → different results.

    Uses synthetic 50-sequence list, K=5, n_reps=20.
    """
    rng = np.random.default_rng(99)
    seqs = [rng.integers(0, 5, size=rng.integers(8, 20)) for _ in range(50)]

    result1 = bootstrap_entropy_rate(seqs, k=5, n_reps=20, seed=42)
    result2 = bootstrap_entropy_rate(seqs, k=5, n_reps=20, seed=42)
    result3 = bootstrap_entropy_rate(seqs, k=5, n_reps=20, seed=7)

    # Same seed → identical
    np.testing.assert_array_equal(
        result1["reps"], result2["reps"],
        err_msg="Same seed produced different bootstrap replicates",
    )
    assert result1["point"] == result2["point"]
    assert result1["ci_lo"] == result2["ci_lo"]
    assert result1["ci_hi"] == result2["ci_hi"]

    # Different seed → different replicates (extremely unlikely to be equal by chance)
    assert not np.array_equal(result1["reps"], result3["reps"]), (
        "Different seeds should produce different replicates"
    )

    # Output keys present
    for key in ("point", "ci_lo", "ci_hi", "reps"):
        assert key in result1, f"Missing key {key!r} in bootstrap output"

    # CI bracket: lo <= point <= hi
    assert result1["ci_lo"] <= result1["point"] <= result1["ci_hi"], (
        f"CI bracket violated: lo={result1['ci_lo']}, "
        f"point={result1['point']}, hi={result1['ci_hi']}"
    )


# ===========================================================================
# 17. detect_idioms — stationary iid: no spurious idioms
# ===========================================================================

def test_detect_idioms_stationary_no_idioms():
    """Spec: iid uniform sequences should produce very few idioms at 99th percentile.

    5000-symbol iid uniform sequence over {0..4}. With n_shuffles=200 and
    percentile=99.0, expect at most 2 detected idioms (< 5% of 25 possible bigrams).
    """
    rng = np.random.default_rng(1234)
    seqs = [rng.integers(0, 5, size=200) for _ in range(25)]

    result = detect_idioms(seqs, k=5, n_shuffles=200, seed=0, percentile=99.0)

    assert "is_idiom" in result.columns
    n_idioms = int(result["is_idiom"].sum())
    assert n_idioms <= 2, (
        f"Stationary iid sequences should flag at most 2 idioms at 99th pct, "
        f"got {n_idioms}: {result[result['is_idiom']]}"
    )


# ===========================================================================
# 18. detect_idioms — injected (0,1) pattern is recovered
# ===========================================================================

def test_detect_idioms_injected_pattern_recovers():
    """Spec: a strongly over-represented bigram is flagged as an idiom.

    Sequences built so 80% of bigrams are (0,1). Running detect_idioms with
    n_shuffles=200 must flag (from_cluster=0, to_cluster=1) as is_idiom=True.
    """
    rng = np.random.default_rng(42)
    seqs = []
    for _ in range(30):
        seq = []
        for _ in range(40):
            # 80% chance of emitting (0,1) bigram
            if rng.random() < 0.8:
                seq.extend([0, 1])
            else:
                seq.append(int(rng.integers(0, 5)))
        seqs.append(np.array(seq, dtype=int))

    result = detect_idioms(seqs, k=5, n_shuffles=200, seed=0, percentile=99.0)

    assert "is_idiom" in result.columns
    row_01 = result[(result["from_cluster"] == 0) & (result["to_cluster"] == 1)]
    assert len(row_01) == 1, "Bigram (0,1) must appear in idioms output"
    assert bool(row_01.iloc[0]["is_idiom"]), (
        f"Bigram (0→1) should be flagged as idiom, enrichment_ratio="
        f"{row_01.iloc[0].get('enrichment_ratio', '?')}"
    )


# ===========================================================================
# 19. one symbol per call enforcement (not per patch)
# ===========================================================================

def test_one_symbol_per_call_enforcement(tmp_path):
    """Spec: after assign_call_clusters, each call contributes EXACTLY one symbol.

    Build synthetic data with 3 calls and 7 patches across those calls.
    The bout sequence must have length 3, not 7.

    Call layout in one wav_stem, one bout (all within 0.25s):
      Call 1: 3 patches → 1 symbol
      Call 2: 2 patches → 1 symbol
      Call 3: 2 patches → 1 symbol
    """
    from sklearn.cluster import KMeans

    # Build call_df with n_patches column and mean_z values
    # simulating the output of load_call_latents
    call_df = pd.DataFrame({
        "cohort": ["5970"] * 3,
        "cohort_split": ["5970"] * 3,
        "wav_stem": ["stemA"] * 3,
        "call_id": [1, 2, 3],
        "begin_time_s": [0.0, 0.05, 0.10],
        "end_time_s":   [0.04, 0.09, 0.14],
        "n_patches": [3, 2, 2],
        "mean_z_0": [0.0, 5.0, 0.0],
        "mean_z_1": [0.0, 0.0, 5.0],
    })

    # Toy KMeans
    centroids = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    km = KMeans(n_clusters=3, n_init=1, max_iter=300, random_state=0)
    km.fit(centroids)
    km.cluster_centers_ = centroids

    labels = assign_call_clusters(call_df, km)

    # One label per call, not per patch
    assert len(labels) == 3, (
        f"assign_call_clusters should return 3 labels (one per call), got {len(labels)}"
    )

    # When we segment and build a bout sequence, length = n_calls
    call_df_with_labels = call_df.copy()
    call_df_with_labels["cluster"] = labels
    bouts = segment_into_bouts(call_df_with_labels, bout_threshold_s=0.25)

    # Build sequences per bout
    sequences = []
    for bout_id, group in bouts.groupby("bout_id"):
        sequences.append(group["cluster"].values)

    total_symbols = sum(len(s) for s in sequences)
    total_patches = call_df["n_patches"].sum()  # = 7

    assert total_symbols == 3, (
        f"Bout sequences should sum to 3 (n_calls), not {total_symbols} (n_patches={total_patches})"
    )


# ===========================================================================
# 20. detect_idioms output schema
# ===========================================================================

def test_idioms_output_schema():
    """Spec: detect_idioms returns DataFrame with exact column names and sensible dtypes."""
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 4, size=30) for _ in range(10)]

    result = detect_idioms(seqs, k=4, n_shuffles=20, seed=0, percentile=99.0)

    required_cols = {
        "from_cluster", "to_cluster", "observed_count",
        "null_p99", "is_idiom", "enrichment_ratio",
    }
    missing = required_cols - set(result.columns)
    assert not missing, f"Missing columns: {missing}"

    # dtype checks
    assert pd.api.types.is_integer_dtype(result["from_cluster"]), (
        "from_cluster should be int dtype"
    )
    assert pd.api.types.is_integer_dtype(result["to_cluster"]), (
        "to_cluster should be int dtype"
    )
    assert pd.api.types.is_bool_dtype(result["is_idiom"]), (
        "is_idiom should be bool dtype"
    )
    assert pd.api.types.is_float_dtype(result["enrichment_ratio"]) or \
           pd.api.types.is_numeric_dtype(result["enrichment_ratio"]), (
        "enrichment_ratio should be numeric"
    )

    # Enrichment ratio >= 0 for all rows
    assert (result["enrichment_ratio"] >= 0).all(), (
        "enrichment_ratio must be non-negative"
    )

    # observed_count >= null_p99 wherever is_idiom=True
    idiom_rows = result[result["is_idiom"]]
    if len(idiom_rows) > 0:
        assert (idiom_rows["observed_count"] >= idiom_rows["null_p99"]).all(), (
            "idiom rows must have observed_count >= null_p99"
        )
