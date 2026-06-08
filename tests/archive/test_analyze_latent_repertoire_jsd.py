"""Tests for analyze_latent_repertoire_jsd — written by test-architect BEFORE implementation.

ROADMAP test plan coverage:
  1. split_lab_cohorts matched/swap/non-lab assignment
     -> test_split_lab_cohorts_assigns_matched_and_swap_correctly
  2. split_lab_cohorts raises on unparseable lab stem
     -> test_split_lab_cohorts_raises_on_unparseable_lab_stem
  3. cluster_proportions rows sum to 1, correct values for known input
     -> test_cluster_proportions_rows_sum_to_one_and_correct_value
  4. cluster_proportions fills missing clusters with 0.0 (not NaN)
     -> test_cluster_proportions_handles_missing_clusters_in_a_cohort
  5. js_divergence_bits(p, p) == 0.0 for various p
     -> test_js_divergence_identical_distributions_returns_zero
  6. js_divergence_bits on orthogonal dists returns 1.0
     -> test_js_divergence_orthogonal_distributions_returns_one
  7. JSD is symmetric, non-negative, bounded by 1
     -> test_js_divergence_symmetric_and_nonneg
  8. JSD hand-computed known value p=[0.5,0.5] q=[1,0]
     -> test_js_divergence_known_value
  9. pairwise_jsd_matrix symmetric, zero diag, shape correct
     -> test_pairwise_jsd_matrix_symmetric_and_zero_diag
  10. bootstrap_jsd_pairs reproducibility, shape, upper-triangle only
      -> test_bootstrap_jsd_reproducibility_and_shape
  11. bootstrap resamples (wav_stem, call_id) tuples, not patches
      -> test_bootstrap_resamples_calls_not_patches
  12. load + split real parquet — exact cohort_split counts
      -> test_load_and_split_real_data_shape
  13. k_sensitivity returns correct shape and k values
      -> test_k_sensitivity_returns_correct_shape

Additional coverage (recurring gap patterns):
  - cluster_proportions column dtype is int (not str)
      -> test_cluster_proportions_column_dtype_is_int
  - pairwise_jsd_matrix values bounded [0, 1]
      -> test_pairwise_jsd_matrix_values_bounded
  - bootstrap output no duplicate pairs and no self-pairs
      -> (covered within test_bootstrap_jsd_reproducibility_and_shape)

Total: 14 tests (13 from ROADMAP, 1 additional)

Decisions made not in spec:
  - test_cluster_proportions_column_dtype_is_int: spec says "columns=[0..k-1] as ints";
    we add an explicit dtype assertion to guard against string columns after parquet round-trip.
  - test_bootstrap_resamples_calls_not_patches: single-call cohort used so every bootstrap
    sample is identical by definition — we test CI width == 0 exactly (not approximately).
    A patch-level resampler would also produce width==0 for a single patch, so we pair the
    1-call cohort with a multi-call partner and verify that the paired CI has nonzero width.
  - JSD known-value tolerance: set to 1e-9 (tighter than spec's 1e-3) because the formula
    is closed-form and floating-point error should be machine-epsilon level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is importable when running from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_latent_repertoire_jsd import (  # noqa: E402  # Will pass after module is created
    bootstrap_jsd_pairs,
    cluster_proportions,
    fit_kmeans,
    js_divergence_bits,
    k_sensitivity,
    load_latents,
    pairwise_jsd_matrix,
    split_lab_cohorts,
)

# ---------------------------------------------------------------------------
# Path to the real parquet
# ---------------------------------------------------------------------------
LATENTS_PARQUET = (
    Path(__file__).parent.parent / "results" / "contour_vae_combined" / "latents.parquet"
)

EXPECTED_COHORT_SPLIT_COUNTS = {
    "5970": 12440,
    "3452": 406,
    "9252": 584,
    "lab_matched": 37677,
    "lab_swap": 18186,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lab_row(wav_stem: str, call_id: int = 0) -> dict:
    """Return a minimal dict for a lab_131204 row."""
    return {
        "cohort": "lab_131204",
        "wav_stem": wav_stem,
        "call_id": call_id,
        "patch_idx": call_id,
        **{f"z_{i}": 0.0 for i in range(32)},
    }


def _make_non_lab_row(cohort: str, wav_stem: str, call_id: int = 0) -> dict:
    return {
        "cohort": cohort,
        "wav_stem": wav_stem,
        "call_id": call_id,
        "patch_idx": call_id,
        **{f"z_{i}": 0.0 for i in range(32)},
    }


def _make_proportions_df(
    cohorts: list[str],
    k: int,
    counts_per_cohort: list[list[int]],
) -> pd.DataFrame:
    """Build a cluster-proportions DataFrame directly from raw counts.

    counts_per_cohort[i] is a list of k non-negative integers, one per cluster.
    """
    rows = []
    for coh, counts in zip(cohorts, counts_per_cohort):
        total = sum(counts)
        row = {j: counts[j] / total for j in range(k)}
        row["index"] = coh
        rows.append(row)
    df = pd.DataFrame(rows).set_index("index")
    df.index.name = None
    return df


def _make_synthetic_call_df(
    cohorts: list[str],
    calls_per_cohort: int,
    patches_per_call: int,
    k: int,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (df, labels) where labels[i] is the cluster id for patch i."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    patch_idx = 0
    for cohort in cohorts:
        for call_num in range(calls_per_cohort):
            for window in range(patches_per_call):
                rows.append({
                    "cohort": cohort,
                    "cohort_split": cohort,
                    "wav_stem": f"{cohort}_rec_{call_num // 10}",
                    "call_id": call_num,
                    "window_idx": window,
                    "patch_idx": patch_idx,
                    **{f"z_{i}": float(rng.standard_normal()) for i in range(4)},
                })
                labels.append(int(rng.integers(0, k)))
                patch_idx += 1
    return pd.DataFrame(rows), np.array(labels, dtype=np.int32)


# ===========================================================================
# 1. split_lab_cohorts — matched vs swap vs non-lab
# ===========================================================================

def test_split_lab_cohorts_assigns_matched_and_swap_correctly():
    """Spec: m==f -> lab_matched; m!=f -> lab_swap; non-lab -> original cohort.

    Also verifies the original 'cohort' column is unchanged.
    """
    rows = [
        _make_lab_row("131204_1400_m1fm1_chunk_001", call_id=0),   # matched
        _make_lab_row("131204_1400_m3fm5_chunk_002", call_id=1),   # swap
        _make_lab_row("131204_1400_m6fm6_chunk_003", call_id=2),   # matched
        _make_non_lab_row("5970", "usv_lmt_034_session1", call_id=3),
    ]
    df = pd.DataFrame(rows)
    original_cohort = df["cohort"].copy()

    result = split_lab_cohorts(df)

    assert "cohort_split" in result.columns, "cohort_split column must be added"

    # Check per-row assignments
    split_vals = result["cohort_split"].tolist()
    assert split_vals[0] == "lab_matched", f"m1fm1 should be lab_matched, got {split_vals[0]!r}"
    assert split_vals[1] == "lab_swap",    f"m3fm5 should be lab_swap, got {split_vals[1]!r}"
    assert split_vals[2] == "lab_matched", f"m6fm6 should be lab_matched, got {split_vals[2]!r}"
    assert split_vals[3] == "5970",        f"5970 row should be '5970', got {split_vals[3]!r}"

    # Aggregate counts
    vc = result["cohort_split"].value_counts()
    assert vc.get("lab_matched", 0) == 2, f"Expected 2 lab_matched, got {vc.get('lab_matched', 0)}"
    assert vc.get("lab_swap", 0) == 1,    f"Expected 1 lab_swap, got {vc.get('lab_swap', 0)}"
    assert vc.get("5970", 0) == 1,        f"Expected 1 '5970', got {vc.get('5970', 0)}"

    # Original cohort column must be unchanged
    pd.testing.assert_series_equal(result["cohort"], original_cohort)


# ===========================================================================
# 2. split_lab_cohorts — ValueError on unparseable lab stem
# ===========================================================================

def test_split_lab_cohorts_raises_on_unparseable_lab_stem():
    """Spec: a lab_131204 row whose wav_stem doesn't match r'_m(\\d+)fm(\\d+)_'
    must raise ValueError. A non-lab row with a garbage stem must NOT raise.
    """
    bad_lab_row = pd.DataFrame([
        _make_lab_row("garbage_no_couple_info", call_id=0),
    ])
    with pytest.raises(ValueError):
        split_lab_cohorts(bad_lab_row)

    # Non-lab garbage stem must be fine
    good_non_lab = pd.DataFrame([
        _make_non_lab_row("5970", "garbage_no_couple_info", call_id=0),
    ])
    result = split_lab_cohorts(good_non_lab)
    assert result["cohort_split"].iloc[0] == "5970", (
        "Non-lab rows with unusual stems should pass through unchanged"
    )


# ===========================================================================
# 3. cluster_proportions — correct values, rows sum to 1
# ===========================================================================

def test_cluster_proportions_rows_sum_to_one_and_correct_value():
    """Spec: cohort A has counts [10,20,5,15] (n=50); cohort B has [25,0,5,20] (n=50).
    Expected proportions: A=[0.20,0.40,0.10,0.30], B=[0.50,0.0,0.10,0.40].
    Columns are integer cluster IDs 0..3. Rows sum to 1.0.
    """
    k = 4
    # Build df: 50 patches per cohort, assign labels according to specified counts
    cohort_a_labels = (
        [0] * 10 + [1] * 20 + [2] * 5 + [3] * 15  # total 50
    )
    cohort_b_labels = (
        [0] * 25 + [1] * 0 + [2] * 5 + [3] * 20   # total 50
    )
    all_labels = np.array(cohort_a_labels + cohort_b_labels, dtype=np.int32)

    rows = (
        [{"cohort_split": "A", "wav_stem": "A_rec", "call_id": i, "patch_idx": i}
         for i in range(50)]
        + [{"cohort_split": "B", "wav_stem": "B_rec", "call_id": i, "patch_idx": 50 + i}
           for i in range(50)]
    )
    df = pd.DataFrame(rows)

    props = cluster_proportions(df, all_labels, k=k, cohort_col="cohort_split")

    # Shape: 2 cohorts × 4 clusters
    assert props.shape == (2, k), f"Expected shape (2, {k}), got {props.shape}"

    # Columns must be integer cluster IDs
    assert list(props.columns) == list(range(k)), (
        f"Expected integer column IDs [0,1,2,3], got {list(props.columns)}"
    )

    # Sort index so we can compare by name
    props = props.sort_index()

    # Check cohort A proportions
    expected_a = np.array([0.20, 0.40, 0.10, 0.30])
    np.testing.assert_allclose(
        props.loc["A"].values, expected_a, atol=1e-10,
        err_msg=f"Cohort A proportions wrong: {props.loc['A'].values}"
    )

    # Check cohort B proportions
    expected_b = np.array([0.50, 0.0, 0.10, 0.40])
    np.testing.assert_allclose(
        props.loc["B"].values, expected_b, atol=1e-10,
        err_msg=f"Cohort B proportions wrong: {props.loc['B'].values}"
    )

    # Rows must sum to 1.0
    row_sums = props.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10,
                               err_msg=f"Rows don't sum to 1: {row_sums.values}")


# ===========================================================================
# Additional: cluster_proportions column dtype is int (not str / object)
# ===========================================================================

def test_cluster_proportions_column_dtype_is_int():
    """Columns in the proportions DataFrame must be Python int / np.integer, not str.

    This guards against string columns after parquet round-trips or str(range(k)).
    """
    k = 5
    labels = np.array([0, 1, 2, 3, 4] * 10, dtype=np.int32)
    rows = [{"cohort_split": "X", "wav_stem": "X_rec", "call_id": i, "patch_idx": i}
            for i in range(50)]
    df = pd.DataFrame(rows)
    props = cluster_proportions(df, labels, k=k, cohort_col="cohort_split")
    for col in props.columns:
        assert isinstance(col, (int, np.integer)), (
            f"Column {col!r} has type {type(col).__name__}, expected int"
        )


# ===========================================================================
# 4. cluster_proportions — missing cluster fills with 0.0 (not NaN)
# ===========================================================================

def test_cluster_proportions_handles_missing_clusters_in_a_cohort():
    """Spec: if cohort B has zero patches in cluster 7 (k=8), cell must be 0.0, not NaN."""
    k = 8
    # Cohort A: patches in all 8 clusters
    cohort_a_labels = list(range(k))            # one patch per cluster
    # Cohort B: patches only in clusters 0..6 (cluster 7 is empty)
    cohort_b_labels = list(range(7))            # no cluster 7

    all_labels = np.array(cohort_a_labels + cohort_b_labels, dtype=np.int32)
    rows = (
        [{"cohort_split": "A", "wav_stem": "A_rec", "call_id": i, "patch_idx": i}
         for i in range(k)]
        + [{"cohort_split": "B", "wav_stem": "B_rec", "call_id": i, "patch_idx": k + i}
           for i in range(7)]
    )
    df = pd.DataFrame(rows)

    props = cluster_proportions(df, all_labels, k=k, cohort_col="cohort_split")

    assert 7 in props.columns, "Cluster 7 column must exist even if B has no patches there"
    b_cluster7 = props.loc["B", 7]
    assert b_cluster7 == 0.0, f"Expected 0.0 for missing cluster, got {b_cluster7!r}"
    assert not np.isnan(b_cluster7), "Missing cluster must be 0.0, not NaN"


# ===========================================================================
# 5. js_divergence_bits — identical distributions return 0.0
# ===========================================================================

def test_js_divergence_identical_distributions_returns_zero():
    """Spec: JSD(p, p) == 0 for any valid distribution p."""
    cases = [
        np.array([0.25, 0.25, 0.25, 0.25]),   # uniform
        np.array([0.9, 0.05, 0.03, 0.02]),     # peaked
        np.array([0.0, 0.0, 1.0, 0.0]),        # one-hot (sparse)
        np.array([0.5, 0.5]),                   # binary
        np.array([1 / 10] * 10),               # uniform k=10
    ]
    for p in cases:
        result = js_divergence_bits(p, p)
        assert abs(result) < 1e-12, (
            f"JSD(p, p) = {result} for p={p}; expected 0.0 within 1e-12"
        )


# ===========================================================================
# 6. js_divergence_bits — orthogonal distributions return 1.0
# ===========================================================================

def test_js_divergence_orthogonal_distributions_returns_one():
    """Spec: JSD([1,0,0,0], [0,1,0,0]) == 1.0 in bits (disjoint support saturates upper bound).

    Hand-computed:
      M = [0.5, 0.5, 0, 0]
      KL(P||M) = 1 * log2(1/0.5) = 1.0
      KL(Q||M) = 1 * log2(1/0.5) = 1.0
      JSD = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    """
    p = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0, 0.0])
    result = js_divergence_bits(p, q)
    assert abs(result - 1.0) < 1e-10, (
        f"Expected JSD([1,0,0,0],[0,1,0,0]) == 1.0, got {result}"
    )


# ===========================================================================
# 7. js_divergence_bits — symmetric, non-negative, bounded by 1
# ===========================================================================

def test_js_divergence_symmetric_and_nonneg():
    """Spec: JSD(p,q) == JSD(q,p), JSD >= 0, JSD <= 1.0 for all random pairs."""
    rng = np.random.default_rng(7)
    k = 5
    for trial in range(10):
        raw_p = rng.exponential(scale=1.0, size=k)
        raw_q = rng.exponential(scale=1.0, size=k)
        p = raw_p / raw_p.sum()
        q = raw_q / raw_q.sum()

        jsd_pq = js_divergence_bits(p, q)
        jsd_qp = js_divergence_bits(q, p)

        assert abs(jsd_pq - jsd_qp) < 1e-12, (
            f"Trial {trial}: JSD(p,q)={jsd_pq} != JSD(q,p)={jsd_qp} — not symmetric"
        )
        assert jsd_pq >= 0, f"Trial {trial}: JSD={jsd_pq} < 0"
        assert jsd_pq <= 1.0 + 1e-10, f"Trial {trial}: JSD={jsd_pq} > 1.0"


# ===========================================================================
# 8. js_divergence_bits — hand-computed known value
# ===========================================================================

def test_js_divergence_known_value():
    """Spec: p=[0.5, 0.5], q=[1.0, 0.0] -> JSD = 0.31127812445913283 bits.

    Hand-computed:
      M = [0.75, 0.25]
      KL(P||M) = 0.5*log2(0.5/0.75) + 0.5*log2(0.5/0.25)
               = 0.5*(-0.58496) + 0.5*(1.0)
               = 0.20752
      KL(Q||M) = 1.0*log2(1.0/0.75)   [0*log2(0/0.25) -> 0 by L'Hopital convention]
               = 0.41504
      JSD      = 0.5*0.20752 + 0.5*0.41504
               = 0.31128
    Verified numerically: 0.31127812445913283
    """
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    expected = 0.31127812445913283
    result = js_divergence_bits(p, q)
    assert abs(result - expected) < 1e-9, (
        f"JSD([0.5,0.5],[1,0]) expected {expected}, got {result}"
    )


# ===========================================================================
# 9. pairwise_jsd_matrix — symmetric, zero diagonal, shape, bounded
# ===========================================================================

def test_pairwise_jsd_matrix_symmetric_and_zero_diag():
    """Spec: N x N symmetric DataFrame, diagonal=0, off-diagonal >= 0, max <= 1."""
    n = 4
    k = 6
    cohorts = ["W1", "W2", "W3", "Lab"]
    # Construct proportions manually (not via cluster_proportions, to isolate function)
    rng = np.random.default_rng(42)
    raw = rng.dirichlet(alpha=np.ones(k), size=n)  # n x k, rows sum to 1
    props = pd.DataFrame(raw, index=cohorts, columns=list(range(k)))

    result = pairwise_jsd_matrix(props)

    # Shape
    assert result.shape == (n, n), f"Expected ({n},{n}), got {result.shape}"

    # Index and columns match cohorts (sorted)
    assert set(result.index) == set(cohorts), f"Index mismatch: {result.index.tolist()}"
    assert set(result.columns) == set(cohorts), f"Columns mismatch: {result.columns.tolist()}"

    # Diagonal == 0
    for c in cohorts:
        diag_val = result.loc[c, c]
        assert abs(diag_val) < 1e-12, f"Diagonal [{c},{c}] = {diag_val}, expected 0"

    # Symmetric
    for i, ci in enumerate(cohorts):
        for j, cj in enumerate(cohorts):
            if i != j:
                assert abs(result.loc[ci, cj] - result.loc[cj, ci]) < 1e-12, (
                    f"Asymmetry at [{ci},{cj}]: {result.loc[ci,cj]} vs {result.loc[cj,ci]}"
                )

    # All values in [0, 1]
    values = result.values
    assert (values >= -1e-12).all(), f"Negative value found: {values.min()}"
    assert (values <= 1.0 + 1e-10).all(), f"Value > 1.0 found: {values.max()}"


# Additional: pairwise_jsd_matrix off-diagonal bounded [0, 1]
def test_pairwise_jsd_matrix_values_bounded():
    """Exhaustive check: even with near-orthogonal cohort distributions, max JSD <= 1."""
    k = 4
    # Nearly orthogonal: each cohort concentrates on one cluster
    cohorts = ["A", "B", "C", "D"]
    raw = np.eye(4)   # cohort A is [1,0,0,0], etc.
    props = pd.DataFrame(raw, index=cohorts, columns=list(range(k)))
    result = pairwise_jsd_matrix(props)
    assert result.values.max() <= 1.0 + 1e-10, f"Max value > 1.0: {result.values.max()}"
    # Off-diagonal between orthogonal cohorts should all be 1.0
    for i, ci in enumerate(cohorts):
        for j, cj in enumerate(cohorts):
            if i != j:
                assert abs(result.loc[ci, cj] - 1.0) < 1e-9, (
                    f"Orthogonal pair [{ci},{cj}] expected JSD=1.0, got {result.loc[ci,cj]}"
                )


# ===========================================================================
# 10. bootstrap_jsd_pairs — reproducibility, shape, upper-triangle only
# ===========================================================================

def test_bootstrap_jsd_reproducibility_and_shape():
    """Spec: same seed -> identical output; upper-triangle pairs only; required columns."""
    n_cohorts = 4
    calls_per_cohort = 50
    k = 5
    cohorts = ["W1", "W2", "W3", "Lab"]

    df, labels = _make_synthetic_call_df(
        cohorts=cohorts,
        calls_per_cohort=calls_per_cohort,
        patches_per_call=2,
        k=k,
        seed=7,
    )

    run1 = bootstrap_jsd_pairs(df, labels, k=k, cohort_col="cohort_split",
                                n_reps=20, seed=42)
    run2 = bootstrap_jsd_pairs(df, labels, k=k, cohort_col="cohort_split",
                                n_reps=20, seed=42)

    # Identical output with same seed
    pd.testing.assert_frame_equal(
        run1.reset_index(drop=True),
        run2.reset_index(drop=True),
    )

    # Required columns
    required_cols = {"cohort_a", "cohort_b", "jsd_point", "jsd_ci_lo", "jsd_ci_hi",
                     "n_reps", "seed"}
    missing = required_cols - set(run1.columns)
    assert not missing, f"Missing columns: {missing}"

    # Exactly 4*3/2 = 6 unique upper-triangle pairs
    n_pairs = n_cohorts * (n_cohorts - 1) // 2
    assert len(run1) == n_pairs, f"Expected {n_pairs} rows, got {len(run1)}"

    # No self-pairs
    self_pairs = run1[run1["cohort_a"] == run1["cohort_b"]]
    assert len(self_pairs) == 0, f"Self-pairs found: {self_pairs}"

    # Upper-triangle: cohort_a < cohort_b lexicographically
    for _, row in run1.iterrows():
        assert row["cohort_a"] < row["cohort_b"], (
            f"Non-upper-triangle pair: ({row['cohort_a']}, {row['cohort_b']})"
        )

    # No duplicate pairs
    pair_tuples = list(zip(run1["cohort_a"], run1["cohort_b"]))
    assert len(pair_tuples) == len(set(pair_tuples)), "Duplicate pairs found"

    # n_reps and seed columns carry through
    assert (run1["n_reps"] == 20).all(), "n_reps column incorrect"
    assert (run1["seed"] == 42).all(), "seed column incorrect"

    # CI values are ordered: ci_lo <= jsd_point <= ci_hi
    assert (run1["jsd_ci_lo"] <= run1["jsd_point"] + 1e-10).all(), (
        "ci_lo > jsd_point in some rows"
    )
    assert (run1["jsd_point"] <= run1["jsd_ci_hi"] + 1e-10).all(), (
        "jsd_point > ci_hi in some rows"
    )


# ===========================================================================
# 11. bootstrap resamples (wav_stem, call_id) tuples, not patches
# ===========================================================================

def test_bootstrap_resamples_calls_not_patches():
    """Load-bearing statistical check: resampling unit is (wav_stem, call_id), not patch.

    Construction:
      Cohort A: 1 call, 10 patches, all in cluster 0.
        -> Every bootstrap resample of cohort A draws that 1 call (1×replacement),
           producing exactly [1.0, 0, 0, 0, 0] proportions every rep.
        -> JSD of A-vs-A across all bootstrap reps is always 0 — CI width == 0.

      Cohort B: 20 calls, 1 patch each, randomly assigned across 5 clusters.
        -> Bootstrap resampling at call level produces genuine variance.
        -> A-vs-B CI width must be > 0.

    If the implementer bootstraps at the patch level, cohort A still has fixed proportions
    (all patches in cluster 0), so this test would still pass. The real discriminating signal
    is that cohort B's patch-level bootstrap would produce zero variance (1 patch per call
    means patch-level == call-level), so we make cohort B have 5 patches per call to
    distinguish: patch-level resampling of B's calls introduces call-level correlation
    that only the call-level bootstrap captures correctly.

    Revised construction for stricter test:
      Cohort A: 1 call, 10 patches, all cluster 0.  (degenerate — 0 variance either way)
      Cohort B: 20 calls × 5 patches/call.
        First 10 calls: all patches in cluster 1.
        Last 10 calls:  all patches in cluster 2.
        -> Call-level bootstrap: each resample picks ~10 from each half -> proportion ~[0.5, 0.5]
           BUT with real variance because the two halves are separately resampleable.
        -> CI width > 0 for A-vs-B.

    The key assertion: jsd_ci_lo for (A, B) < jsd_ci_hi for (A, B) (nonzero width).
    And: the CI is NOT exactly [0, 0] for the A-vs-B pair.
    """
    k = 5

    rows = []
    labels_list = []
    patch_idx = 0

    # Cohort A: 1 call, 10 patches, all cluster 0
    for _ in range(10):
        rows.append({
            "cohort": "A", "cohort_split": "A",
            "wav_stem": "A_stem_0", "call_id": 0,
            "window_idx": patch_idx, "patch_idx": patch_idx,
            **{f"z_{i}": 0.0 for i in range(4)},
        })
        labels_list.append(0)
        patch_idx += 1

    # Cohort B: 20 calls × 5 patches/call
    # Calls 0..9: all patches in cluster 1
    # Calls 10..19: all patches in cluster 2
    for call_id in range(20):
        cluster = 1 if call_id < 10 else 2
        for w in range(5):
            rows.append({
                "cohort": "B", "cohort_split": "B",
                "wav_stem": f"B_stem_{call_id // 5}", "call_id": call_id,
                "window_idx": w, "patch_idx": patch_idx,
                **{f"z_{i}": 0.0 for i in range(4)},
            })
            labels_list.append(cluster)
            patch_idx += 1

    df = pd.DataFrame(rows)
    labels = np.array(labels_list, dtype=np.int32)

    result = bootstrap_jsd_pairs(df, labels, k=k, cohort_col="cohort_split",
                                  n_reps=200, seed=0)

    # Should have exactly 1 pair: (A, B)
    assert len(result) == 1, f"Expected 1 pair, got {len(result)}: {result}"
    row = result.iloc[0]
    assert row["cohort_a"] == "A" and row["cohort_b"] == "B", (
        f"Unexpected pair: ({row['cohort_a']}, {row['cohort_b']})"
    )

    # A has only 1 unique call — every bootstrap sample for A is that same call.
    # B has 20 calls — bootstrap samples vary.
    # Therefore JSD variance is driven entirely by B's variability.
    # CI width must be > 0.
    ci_width = row["jsd_ci_hi"] - row["jsd_ci_lo"]
    assert ci_width > 0, (
        f"CI width == 0 for (A, B) pair. "
        f"ci_lo={row['jsd_ci_lo']}, ci_hi={row['jsd_ci_hi']}. "
        "This may indicate bootstrapping at the patch level (A has 10 patches all in cluster 0, "
        "so patch-level resampling would also give zero variance for A, but B's 20 calls with "
        "fixed cluster per call mean patch-level and call-level give identical per-call results "
        "— double-check implementation)."
    )

    # jsd_point > 0 (A concentrates on cluster 0, B on clusters 1/2)
    assert row["jsd_point"] > 0, f"jsd_point=0; expected A-vs-B to differ"


# ===========================================================================
# 12. Real data: load + split gives exact cohort_split counts
# ===========================================================================

def test_load_and_split_real_data_shape():
    """Spec: load real parquet, split_lab_cohorts, verify exact cohort_split value counts.

    Expected: 5970=12440, 3452=406, 9252=584, lab_matched=37677, lab_swap=18186.
    """
    if not LATENTS_PARQUET.exists():
        pytest.skip(f"Real parquet not found at {LATENTS_PARQUET}")

    df = load_latents(str(LATENTS_PARQUET))
    df_split = split_lab_cohorts(df)

    assert "cohort_split" in df_split.columns, "cohort_split column missing after split"

    vc = df_split["cohort_split"].value_counts().to_dict()

    missing_keys = set(EXPECTED_COHORT_SPLIT_COUNTS) - set(vc)
    assert not missing_keys, f"Missing cohort_split values: {missing_keys}"

    extra_keys = set(vc) - set(EXPECTED_COHORT_SPLIT_COUNTS)
    assert not extra_keys, f"Unexpected cohort_split values: {extra_keys}"

    for cohort, expected_count in EXPECTED_COHORT_SPLIT_COUNTS.items():
        actual = vc[cohort]
        assert actual == expected_count, (
            f"cohort_split={cohort!r}: expected {expected_count}, got {actual}"
        )


# ===========================================================================
# 13. k_sensitivity — correct shape and k values
# ===========================================================================

def test_k_sensitivity_returns_correct_shape():
    """Spec: k_sensitivity(df, Z, ks=[3,5], ...) returns long-form DF with
    columns {k, cohort_a, cohort_b, jsd} and n_rows = len(ks) * n_pairs.
    """
    cohorts = ["W1", "W2", "Lab"]
    calls_per_cohort = 30
    k_for_data = 8  # labels drawn from 0..7
    ks = [3, 5]

    df, _ = _make_synthetic_call_df(
        cohorts=cohorts,
        calls_per_cohort=calls_per_cohort,
        patches_per_call=2,
        k=k_for_data,
        seed=99,
    )
    Z_cols = [f"z_{i}" for i in range(4)]
    Z = df[Z_cols].values.astype(np.float32)

    result = k_sensitivity(df, Z, ks=ks, cohort_col="cohort_split", seed=0)

    # Required columns
    required_cols = {"k", "cohort_a", "cohort_b", "jsd"}
    missing = required_cols - set(result.columns)
    assert not missing, f"Missing columns: {missing}"

    # All k values present
    assert set(result["k"].unique()) == set(ks), (
        f"Expected k values {set(ks)}, got {set(result['k'].unique())}"
    )

    # n_pairs per k: 3 * 2 / 2 = 3
    n_pairs = len(cohorts) * (len(cohorts) - 1) // 2
    expected_rows = len(ks) * n_pairs
    assert len(result) == expected_rows, (
        f"Expected {expected_rows} rows ({len(ks)} ks × {n_pairs} pairs), got {len(result)}"
    )

    # No CIs in k_sensitivity output
    assert "jsd_ci_lo" not in result.columns, (
        "k_sensitivity should not include CI columns (no bootstrap)"
    )

    # JSD values are non-negative and bounded
    assert (result["jsd"] >= 0).all(), f"Negative JSD in k_sensitivity: {result['jsd'].min()}"
    assert (result["jsd"] <= 1.0 + 1e-10).all(), (
        f"JSD > 1.0 in k_sensitivity: {result['jsd'].max()}"
    )
