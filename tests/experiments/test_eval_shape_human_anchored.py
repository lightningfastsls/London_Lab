"""Tests for scripts/experiments/eval_shape_human_anchored.py — written by
test-architect BEFORE implementation exists.

This module is the standing gate for deciding whether an elastic (soft-DTW)
metric beats the incumbent registration-Euclidean metric for 1-D ridge shape
representations, evaluated against ~200 human shape labels using leave-one-out
kNN retrieval purity with bootstrap confidence intervals.

ROADMAP test plan coverage:
  1. group_family chevron/reverse-chevron mapping       -> test_group_family_chevron_variants
  2. group_family jump-family mapping (4 labels)        -> test_group_family_jump_variants
  3. group_family flat mapping                          -> test_group_family_flat
  4. group_family complex mapping                       -> test_group_family_complex
  5. group_family identity for unknown labels           -> test_group_family_identity_for_unknown_labels
  6. build_join basic matching with offset=-1           -> test_build_join_basic_offset_minus_one
  7. build_join offset comparison (offset 0 vs -1)      -> test_build_join_offset_affects_match_count
  8. build_join dedup: first index wins                 -> test_build_join_dedup_first_index_wins
  9. build_join returns correct joined_df subset        -> test_build_join_joined_df_is_subset_of_human_df
 10. build_join row column in joined_df is integer      -> test_build_join_row_column_dtype_integer
 11. loo_knn_purity two-blob purity near 1.0            -> test_loo_knn_purity_separated_blobs_near_one
 12. loo_knn_purity random labels near base rate        -> test_loo_knn_purity_random_labels_near_base_rate
 13. loo_knn_purity absent target returns (nan, 0)      -> test_loo_knn_purity_absent_target_returns_nan_zero
 14. loo_knn_purity determinism                         -> test_loo_knn_purity_deterministic
 15. loo_knn_purity k capped at n-1                     -> test_loo_knn_purity_k_capped_at_n_minus_one
 16. knn_purity_from_distance separated blobs ~1.0      -> test_knn_purity_from_distance_separated_blobs
 17. knn_purity_from_distance matches loo_knn_purity    -> test_knn_purity_from_distance_matches_euclidean_path
 18. knn_purity_from_distance absent target returns nan -> test_knn_purity_from_distance_absent_target_nan
 19. bootstrap_purity_ci reproducibility with seed      -> test_bootstrap_purity_ci_reproducible_with_seed
 20. bootstrap_purity_ci ordering ci_lo <= pt <= ci_hi  -> test_bootstrap_purity_ci_ordering
 21. bootstrap_purity_ci bounds [0, 1]                  -> test_bootstrap_purity_ci_bounds_zero_to_one
 22. bootstrap_purity_ci tight CI on degenerate case    -> test_bootstrap_purity_ci_tight_on_degenerate

Additional coverage (recurring gap patterns):
  - group_family is case-sensitive (no accidental lowercasing) -> test_group_family_case_sensitive
  - build_join empty human_df -> empty results               -> test_build_join_empty_human_df
  - build_join no matches -> empty results                    -> test_build_join_no_matches_returns_empty
  - loo_knn_purity single point per target class             -> test_loo_knn_purity_single_target_point
  - loo_knn_purity purity value is in [0, 1]                 -> test_loo_knn_purity_output_in_unit_interval
  - bootstrap_purity_ci different seeds differ               -> test_bootstrap_purity_ci_different_seeds_differ
  - knn_purity_from_distance diagonal treated as +inf        -> test_knn_purity_from_distance_self_excluded

Total: 29 tests (22 from ROADMAP test plan items, 7 additional)

Spec ambiguities resolved:
  - "Bootstrap resamples the SET OF TARGET-CLASS points": this means the n_boot
    bootstrap samples are each a resample (with replacement) of the indices that
    belong to the target class; the purity across the full n is evaluated for
    each resample. We test this by checking CI width narrows with more target
    points in a degenerate identical-neighbour scenario.
  - "diagonal treated as +inf": the spec says self-exclusion is by treating the
    diagonal as +inf before sorting; we verify a point is never its own nearest
    neighbour in the D-based path.
  - "k is capped at n-1 if fewer points than k+1": tested with a 3-point
    dataset and k=10 — should not crash and should use k=2 effectively.
  - group_family is expected to be case-sensitive per spec ("Chevron" not
    "chevron"); we test that "chevron" (lowercase) falls through to identity.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap  (Pattern 8, consistent with existing tests)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
_REPO = REPO_ROOT
for _p in (_SRC, str(_REPO)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Import the module under test.
# This will fail with ModuleNotFoundError until the implementation exists.
# That is the expected red-phase failure — do NOT add a skip guard here.
# ---------------------------------------------------------------------------
from scripts.experiments.eval_shape_human_anchored import (  # noqa: E402
    bootstrap_purity_ci,
    build_join,
    group_family,
    knn_purity_from_distance,
    loo_knn_purity,
)

# ===========================================================================
# Helper factories (all synthetic — no rig data, no npz files)
# ===========================================================================

def _two_blob_dataset(n_per_class: int = 60, dim: int = 8, sep: float = 20.0, seed: int = 0):
    """Return (X, labels) for two well-separated Gaussian blobs.

    Class 'A' centred at +sep/2 along dim-0; class 'B' at -sep/2.
    With sep=20 and std=1, min inter-blob distance >> intra-blob distance.
    """
    rng = np.random.default_rng(seed)
    X_a = rng.standard_normal((n_per_class, dim)) + np.array([sep / 2] + [0.0] * (dim - 1))
    X_b = rng.standard_normal((n_per_class, dim)) + np.array([-sep / 2] + [0.0] * (dim - 1))
    X = np.vstack([X_a, X_b]).astype(np.float64)
    labels = np.array(["A"] * n_per_class + ["B"] * n_per_class)
    return X, labels


def _euclidean_pairwise(X: np.ndarray) -> np.ndarray:
    """Return (n, n) symmetric Euclidean distance matrix."""
    return cdist(X, X, metric="euclidean")


def _make_human_df(call_ids: list[str], shape_labels: list[str]) -> pd.DataFrame:
    """Build a minimal human-label DataFrame."""
    return pd.DataFrame({"call_id": call_ids, "shape_label": shape_labels})


# ===========================================================================
# group_family  — label taxonomy tests
# ===========================================================================

class TestGroupFamily:
    """Spec: group_family maps Grimsley display names to family buckets."""

    def test_group_family_chevron_variants(self):
        """Both 'Chevron' and 'Reverse Chevron' should map to 'chevron'."""
        assert group_family("Chevron") == "chevron"
        assert group_family("Reverse Chevron") == "chevron"

    def test_group_family_jump_variants(self):
        """All four step/jump types map to 'jump'."""
        for label in ("Step up", "Step down", "Two steps", "Multi-steps"):
            result = group_family(label)
            assert result == "jump", (
                f"Expected 'jump' for label {label!r}, got {result!r}"
            )

    def test_group_family_flat(self):
        """'Flat' maps to 'flat'."""
        assert group_family("Flat") == "flat"

    def test_group_family_complex(self):
        """'Complex' maps to 'complex'."""
        assert group_family("Complex") == "complex"

    def test_group_family_identity_for_unknown_labels(self):
        """Labels outside the mapping are returned unchanged (identity)."""
        for label in ("Noise", "FM", "Short", "Up-FM", "Down-FM", "Custom123"):
            assert group_family(label) == label, (
                f"Expected identity for {label!r}, got {group_family(label)!r}"
            )

    def test_group_family_case_sensitive(self):
        """group_family is case-sensitive; lowercase 'chevron' is NOT a known key."""
        # "chevron" (lowercase) should fall through to identity, not map to "chevron"
        # because the spec keys are title-case display names.
        result = group_family("chevron")
        assert result == "chevron"  # identity (unchanged), not the bucket

        result_rc = group_family("reverse chevron")
        assert result_rc == "reverse chevron"  # identity, not "chevron"

    def test_group_family_empty_string_identity(self):
        """Empty string is an unknown label and returns itself."""
        assert group_family("") == ""


# ===========================================================================
# build_join  — ridge-to-label join tests
# ===========================================================================

class TestBuildJoin:
    """Spec: build_join links ridge rows to human-labeled call_ids."""

    def _simple_setup(self):
        """
        Minimal synthetic setup:
          wav_stem = ['rec_001', 'rec_001', 'rec_002']
          call_id  = [  5,         6,         3       ]
          With offset=-1, composite ids are:
            'rec_001__det4', 'rec_001__det5', 'rec_002__det2'
          With offset=0, composite ids are:
            'rec_001__det5', 'rec_001__det6', 'rec_002__det3'
        """
        wav_stem = np.array(["rec_001", "rec_001", "rec_002"])
        call_id = np.array([5, 6, 3])
        return wav_stem, call_id

    def test_build_join_basic_offset_minus_one(self):
        """With offset=-1, composite ids are f'{stem}__det{id-1}' and must match human_df."""
        wav_stem, call_id = self._simple_setup()
        # Human df uses 0-indexed detection ids matching offset=-1 composites
        human_df = _make_human_df(
            call_ids=["rec_001__det4", "rec_001__det5", "rec_002__det2"],
            shape_labels=["Flat", "Chevron", "Step up"],
        )
        rows, joined_df = build_join(wav_stem, call_id, human_df, offset=-1)

        assert len(rows) == 3, f"Expected 3 matched rows, got {len(rows)}"
        assert len(joined_df) == 3, f"Expected joined_df length 3, got {len(joined_df)}"

        # Verify row indices point to the correct ridge rows
        assert rows[0] == 0, f"First match should be ridge row 0 (rec_001__det4)"
        assert rows[1] == 1, f"Second match should be ridge row 1 (rec_001__det5)"
        assert rows[2] == 2, f"Third match should be ridge row 2 (rec_002__det2)"

    def test_build_join_offset_affects_match_count(self):
        """offset=-1 matches call_ids built from (call_id-1); offset=0 matches (call_id+0).

        Specifically: for a human_df that only has offset-0 composite ids,
        offset=0 should match all; offset=-1 should match fewer (or none).
        """
        wav_stem = np.array(["rec_001", "rec_001"])
        call_id = np.array([5, 6])

        # Human df uses offset=0 ids (call_id unchanged)
        human_df = _make_human_df(
            call_ids=["rec_001__det5", "rec_001__det6"],
            shape_labels=["Flat", "Chevron"],
        )
        rows_0, joined_0 = build_join(wav_stem, call_id, human_df, offset=0)
        rows_m1, joined_m1 = build_join(wav_stem, call_id, human_df, offset=-1)

        # With offset=0, composite ids are rec_001__det5 and rec_001__det6 — both match
        assert len(rows_0) == 2, f"offset=0 should match 2, got {len(rows_0)}"
        # With offset=-1, composite ids are rec_001__det4 and rec_001__det5 — only det5 matches
        assert len(rows_m1) == 1, f"offset=-1 should match 1, got {len(rows_m1)}"

    def test_build_join_dedup_first_index_wins(self):
        """If the same composite id appears at two ridge rows, the FIRST index wins."""
        # Two ridge rows produce the same composite id with offset=-1
        wav_stem = np.array(["rec_001", "rec_001", "rec_001"])
        call_id = np.array([5, 5, 6])  # rows 0 and 1 both produce rec_001__det4
        human_df = _make_human_df(
            call_ids=["rec_001__det4", "rec_001__det5"],
            shape_labels=["Flat", "Chevron"],
        )
        rows, joined_df = build_join(wav_stem, call_id, human_df, offset=-1)

        # rec_001__det4 should map to row index 0 (first occurrence), not 1
        rec4_row = joined_df.loc[joined_df["call_id"] == "rec_001__det4", "row"].iloc[0]
        assert rec4_row == 0, f"Expected first-occurrence index 0, got {rec4_row}"

    def test_build_join_joined_df_is_subset_of_human_df(self):
        """joined_df contains only rows from human_df whose call_id matched."""
        wav_stem = np.array(["rec_001"])
        call_id = np.array([5])
        human_df = _make_human_df(
            call_ids=["rec_001__det4", "rec_999__det99"],  # only first matches
            shape_labels=["Flat", "Noise"],
        )
        rows, joined_df = build_join(wav_stem, call_id, human_df, offset=-1)

        assert len(joined_df) == 1
        assert joined_df.iloc[0]["call_id"] == "rec_001__det4"
        assert joined_df.iloc[0]["shape_label"] == "Flat"

    def test_build_join_row_column_dtype_integer(self):
        """The added 'row' column in joined_df must hold integer indices, not floats."""
        wav_stem = np.array(["rec_001", "rec_002"])
        call_id = np.array([5, 3])
        human_df = _make_human_df(
            call_ids=["rec_001__det4", "rec_002__det2"],
            shape_labels=["Flat", "Chevron"],
        )
        rows, joined_df = build_join(wav_stem, call_id, human_df, offset=-1)

        assert "row" in joined_df.columns, "'row' column missing from joined_df"
        assert np.issubdtype(joined_df["row"].dtype, np.integer), (
            f"'row' column dtype should be integer, got {joined_df['row'].dtype}"
        )

    def test_build_join_empty_human_df(self):
        """Empty human_df -> no matches -> empty rows and empty joined_df."""
        wav_stem = np.array(["rec_001", "rec_002"])
        call_id = np.array([5, 3])
        human_df = _make_human_df(call_ids=[], shape_labels=[])

        rows, joined_df = build_join(wav_stem, call_id, human_df, offset=-1)

        assert len(rows) == 0, f"Expected 0 rows, got {len(rows)}"
        assert len(joined_df) == 0, f"Expected empty joined_df, got {len(joined_df)}"

    def test_build_join_no_matches_returns_empty(self):
        """When no composite id matches any human_df call_id, return empty arrays."""
        wav_stem = np.array(["rec_001"])
        call_id = np.array([5])
        human_df = _make_human_df(
            call_ids=["totally_different__det99"],
            shape_labels=["Noise"],
        )
        rows, joined_df = build_join(wav_stem, call_id, human_df, offset=-1)

        assert len(rows) == 0
        assert len(joined_df) == 0


# ===========================================================================
# loo_knn_purity  — leave-one-out kNN purity tests
# ===========================================================================

class TestLooKnnPurity:
    """Spec: loo_knn_purity computes LOO kNN retrieval purity for a target family."""

    def test_loo_knn_purity_separated_blobs_near_one(self):
        """Two well-separated Gaussian blobs -> purity for each class should be > 0.9."""
        X, labels = _two_blob_dataset(n_per_class=60, sep=20.0, seed=1)
        purity_a, n_a = loo_knn_purity(X, labels, target="A", k=10)
        purity_b, n_b = loo_knn_purity(X, labels, target="B", k=10)

        assert n_a == 60, f"Expected 60 points with label A, got {n_a}"
        assert n_b == 60, f"Expected 60 points with label B, got {n_b}"
        assert purity_a > 0.9, f"Expected purity > 0.9 for well-separated blob A, got {purity_a:.4f}"
        assert purity_b > 0.9, f"Expected purity > 0.9 for well-separated blob B, got {purity_b:.4f}"

    def test_loo_knn_purity_random_labels_near_base_rate(self):
        """Randomly shuffled labels on one blob -> purity near class fraction (base rate).

        With 30 A and 70 B points on a single isotropic Gaussian, expected purity
        for A is approximately 30/100 = 0.30. We assert within ±0.15 of that.
        """
        rng = np.random.default_rng(42)
        n_total = 100
        n_a = 30
        X = rng.standard_normal((n_total, 8))
        labels = np.array(["A"] * n_a + ["B"] * (n_total - n_a))
        rng.shuffle(labels)

        purity, n_target = loo_knn_purity(X, labels, target="A", k=10)
        expected_base_rate = n_a / n_total

        assert n_target == n_a, f"Expected {n_a} target points, got {n_target}"
        assert not math.isnan(purity), "Purity should not be NaN for a non-empty target"
        # Purity should be near base rate for random labels
        assert abs(purity - expected_base_rate) < 0.15, (
            f"Random-label purity {purity:.3f} too far from base rate {expected_base_rate:.3f}"
        )

    def test_loo_knn_purity_absent_target_returns_nan_zero(self):
        """If no points have label==target, return (nan, 0)."""
        X, labels = _two_blob_dataset(n_per_class=20, seed=2)
        purity, n_target = loo_knn_purity(X, labels, target="MISSING_LABEL", k=5)

        assert n_target == 0, f"Expected 0 target points, got {n_target}"
        assert math.isnan(purity), f"Expected NaN purity for absent target, got {purity}"

    def test_loo_knn_purity_deterministic(self):
        """Same inputs produce identical output on repeated calls."""
        X, labels = _two_blob_dataset(n_per_class=30, seed=3)
        r1 = loo_knn_purity(X, labels, target="A", k=5)
        r2 = loo_knn_purity(X, labels, target="A", k=5)

        assert r1[0] == r2[0], "Purity should be deterministic"
        assert r1[1] == r2[1], "n_target should be deterministic"

    def test_loo_knn_purity_k_capped_at_n_minus_one(self):
        """With only 3 points and k=10, function should not crash (cap k at n-1=2)."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = np.array(["A", "A", "B"])

        # Should not raise; with k capped at 2 there is 1 target A per LOO query
        purity, n_target = loo_knn_purity(X, labels, target="A", k=10)

        assert n_target == 2
        assert not math.isnan(purity)
        assert 0.0 <= purity <= 1.0

    def test_loo_knn_purity_output_in_unit_interval(self):
        """Purity must always be in [0.0, 1.0] for non-trivial inputs."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((50, 4))
        labels = np.array(["A"] * 25 + ["B"] * 25)

        purity, _ = loo_knn_purity(X, labels, target="A", k=5)
        assert 0.0 <= purity <= 1.0, f"Purity {purity} outside [0, 1]"

    def test_loo_knn_purity_single_target_point(self):
        """A target with exactly 1 point: purity is well-defined (fraction of k neighbours
        that are in the target class); should not crash."""
        # 1 target point ('A') and 10 non-target points ('B')
        rng = np.random.default_rng(7)
        X = rng.standard_normal((11, 4))
        labels = np.array(["A"] + ["B"] * 10)

        purity, n_target = loo_knn_purity(X, labels, target="A", k=5)
        assert n_target == 1
        assert not math.isnan(purity)
        assert 0.0 <= purity <= 1.0

    def test_loo_knn_purity_perfect_isolation(self):
        """Hand-computed spot-check: 4 target points all mutually closest to each other.

        Layout: A=[0,0],[1,0],[0,1],[1,1]; B=[100,100],[101,100],[100,101],[101,101]
        With k=3, each A's 3 nearest neighbours are the other 3 A's -> purity = 1.0.
        """
        X = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],  # A cluster
            [100.0, 100.0], [101.0, 100.0], [100.0, 101.0], [101.0, 101.0],  # B cluster
        ])
        labels = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        purity, n_target = loo_knn_purity(X, labels, target="A", k=3)

        assert n_target == 4
        assert abs(purity - 1.0) < 1e-9, f"Expected purity 1.0, got {purity}"

    def test_loo_knn_purity_hand_computed_spot_check(self):
        """Exact spot-check with k=1 LOO.

        Setup: 3 points. A=[0,0],[3,0]; B=[4,0].
        LOO for A[0]=[0,0]: nearest non-self = A[1]=[3,0] (dist=3) vs B[0]=[4,0] (dist=4)
            -> nearest is A -> contributes 1/1 = 1.0
        LOO for A[1]=[3,0]: nearest non-self = B[0]=[4,0] (dist=1) vs A[0]=[0,0] (dist=3)
            -> nearest is B -> contributes 0/1 = 0.0
        Mean purity for target A = (1.0 + 0.0) / 2 = 0.5
        """
        X = np.array([[0.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        labels = np.array(["A", "A", "B"])
        purity, n_target = loo_knn_purity(X, labels, target="A", k=1)

        assert n_target == 2
        assert abs(purity - 0.5) < 1e-9, f"Expected purity 0.5, got {purity}"


# ===========================================================================
# knn_purity_from_distance  — precomputed distance matrix path
# ===========================================================================

class TestKnnPurityFromDistance:
    """Spec: knn_purity_from_distance mirrors loo_knn_purity but via a precomputed D matrix."""

    def test_knn_purity_from_distance_separated_blobs(self):
        """Well-separated blobs -> purity > 0.9 via precomputed Euclidean D."""
        X, labels = _two_blob_dataset(n_per_class=60, sep=20.0, seed=4)
        D = _euclidean_pairwise(X)

        purity_a = knn_purity_from_distance(D, labels, target="A", k=10)
        assert purity_a > 0.9, f"Expected > 0.9 from D-based path, got {purity_a:.4f}"

    def test_knn_purity_from_distance_matches_euclidean_path(self):
        """D built from Euclidean pairwise distances must yield the same purity as
        loo_knn_purity to within 1e-9 (bit-for-bit consistency)."""
        X, labels = _two_blob_dataset(n_per_class=40, sep=15.0, seed=5)
        D = _euclidean_pairwise(X)

        for target in ("A", "B"):
            purity_x, _ = loo_knn_purity(X, labels, target=target, k=5)
            purity_d = knn_purity_from_distance(D, labels, target=target, k=5)
            assert abs(purity_x - purity_d) < 1e-9, (
                f"Target {target}: loo_knn_purity={purity_x:.10f} vs "
                f"knn_purity_from_distance={purity_d:.10f} — should be identical"
            )

    def test_knn_purity_from_distance_absent_target_nan(self):
        """Absent target returns nan from the D-based path too."""
        X, labels = _two_blob_dataset(n_per_class=20, seed=6)
        D = _euclidean_pairwise(X)

        result = knn_purity_from_distance(D, labels, target="NONEXISTENT", k=5)
        assert math.isnan(result), f"Expected NaN for absent target, got {result}"

    def test_knn_purity_from_distance_self_excluded(self):
        """The diagonal (self-distance) is treated as +inf so a point is never
        its own nearest neighbour.

        Hand-check: 3 points, D[0,0]=0 (would be self), but nearest of point 0 is
        determined by D[0,1] vs D[0,2]. If the diagonal were NOT excluded,
        point 0 would always 'retrieve' itself, making purity artificially 1.
        We use a D where D[i,i] = 0 explicitly and verify the result does not
        count self-retrieval.
        """
        # 3 points: A at rows 0,1; B at row 2
        # D: A[0] is closest to B[2] (dist=1) then A[1] (dist=5)
        # If self were included, D[0,0]=0 would be the nearest -> wrong purity
        D = np.array([
            [0.0, 5.0, 1.0],
            [5.0, 0.0, 2.0],
            [1.0, 2.0, 0.0],
        ])
        labels = np.array(["A", "A", "B"])

        # LOO k=1 for target A:
        #   A[0]: nearest non-self = B[2] (dist=1) -> 0 A-neighbours -> contributes 0.0
        #   A[1]: nearest non-self = B[2] (dist=2) -> 0 A-neighbours -> contributes 0.0
        # purity = 0.0
        purity = knn_purity_from_distance(D, labels, target="A", k=1)
        assert abs(purity - 0.0) < 1e-9, (
            f"Expected purity 0.0 (self excluded, nearest is B), got {purity}"
        )


# ===========================================================================
# bootstrap_purity_ci  — bootstrap confidence interval tests
# ===========================================================================

class TestBootstrapPurityCi:
    """Spec: bootstrap_purity_ci wraps loo_knn_purity with bootstrap CIs."""

    def test_bootstrap_purity_ci_reproducible_with_seed(self):
        """Same seed=42 gives identical (point, lo, hi) on repeated calls."""
        X, labels = _two_blob_dataset(n_per_class=30, sep=15.0, seed=10)
        r1 = bootstrap_purity_ci(X, labels, target="A", k=5, n_boot=200, seed=42)
        r2 = bootstrap_purity_ci(X, labels, target="A", k=5, n_boot=200, seed=42)

        assert r1[0] == r2[0], "point_estimate not reproducible"
        assert r1[1] == r2[1], "ci_lo not reproducible"
        assert r1[2] == r2[2], "ci_hi not reproducible"

    def test_bootstrap_purity_ci_ordering(self):
        """ci_lo <= point_estimate <= ci_hi for well-separated blobs."""
        X, labels = _two_blob_dataset(n_per_class=40, sep=18.0, seed=11)
        point, lo, hi = bootstrap_purity_ci(X, labels, target="A", k=5, n_boot=300, seed=42)

        assert lo <= point, f"ci_lo={lo:.4f} > point_estimate={point:.4f}"
        assert point <= hi, f"point_estimate={point:.4f} > ci_hi={hi:.4f}"

    def test_bootstrap_purity_ci_bounds_zero_to_one(self):
        """CI endpoints must be in [0.0, 1.0] since purity is a fraction."""
        rng = np.random.default_rng(55)
        X = rng.standard_normal((60, 6))
        labels = np.array(["chevron"] * 30 + ["flat"] * 30)
        point, lo, hi = bootstrap_purity_ci(X, labels, target="chevron", k=5, n_boot=200, seed=42)

        assert lo >= 0.0, f"ci_lo={lo} below 0"
        assert hi <= 1.0, f"ci_hi={hi} above 1"
        assert 0.0 <= point <= 1.0, f"point_estimate={point} outside [0,1]"

    def test_bootstrap_purity_ci_tight_on_degenerate(self):
        """A perfectly separated dataset gives purity 1.0; the CI should be [1.0, 1.0]
        (zero width) because all bootstrap samples also give purity 1.0.

        This tests the invariant: degenerate all-same-neighbour case -> tight CI.
        """
        # Maximally separated: A cluster far from B, so every resample of A points
        # still has purity 1.0 (each A's k=5 nearest neighbours are all A)
        X, labels = _two_blob_dataset(n_per_class=50, sep=1000.0, seed=12)
        point, lo, hi = bootstrap_purity_ci(X, labels, target="A", k=5, n_boot=200, seed=42)

        assert abs(point - 1.0) < 1e-9, f"Expected point_estimate=1.0, got {point}"
        # CI width should be tiny (0.0 for perfectly separable data)
        ci_width = hi - lo
        assert ci_width < 0.01, (
            f"Expected near-zero CI width for perfectly separated data, got {ci_width:.4f}"
        )

    def test_bootstrap_purity_ci_different_seeds_differ(self):
        """Different seeds should produce different CI bounds (not necessarily different
        point estimates, but the bootstrap distribution should differ).

        Fixture note (Shachar-approved 2026-06-03): the original sep=5.0/k=5 fixture
        was *too discrete* — target per-point purities took only 2 values {0.8, 1.0},
        so the percentile-bootstrap endpoints land on the same discrete level for any
        seed (verified to persist at n_boot=5000). That is correct percentile-bootstrap
        behaviour on a near-degenerate statistic, not a code bug. We use a less-discrete
        fixture (npc=100, sep=2.5, k=10 -> ~9 per-point levels) so the seed genuinely
        drives the CI endpoints; the assertion below is unchanged.
        """
        X, labels = _two_blob_dataset(n_per_class=100, sep=2.5, seed=13)
        r_42 = bootstrap_purity_ci(X, labels, target="A", k=10, n_boot=500, seed=42)
        r_99 = bootstrap_purity_ci(X, labels, target="A", k=10, n_boot=500, seed=99)

        # Point estimates should be identical (same LOO computation)
        assert r_42[0] == r_99[0], "Point estimate should be seed-independent"

        # With 500 bootstrap samples and non-degenerate data, lo or hi should differ
        # (probability of identical results with different seeds is astronomically small)
        ci_differ = (r_42[1] != r_99[1]) or (r_42[2] != r_99[2])
        assert ci_differ, (
            "CI bounds should differ with different seeds on non-degenerate data; "
            f"seed=42: ({r_42[1]:.6f}, {r_42[2]:.6f}), "
            f"seed=99: ({r_99[1]:.6f}, {r_99[2]:.6f})"
        )

    def test_bootstrap_purity_ci_point_estimate_equals_loo(self):
        """point_estimate must exactly equal loo_knn_purity(X, labels, target, k)[0]."""
        X, labels = _two_blob_dataset(n_per_class=25, sep=10.0, seed=14)
        point, lo, hi = bootstrap_purity_ci(X, labels, target="A", k=5, n_boot=100, seed=42)
        expected_point, _ = loo_knn_purity(X, labels, target="A", k=5)

        assert abs(point - expected_point) < 1e-12, (
            f"bootstrap_purity_ci point_estimate={point} != "
            f"loo_knn_purity={expected_point}"
        )
