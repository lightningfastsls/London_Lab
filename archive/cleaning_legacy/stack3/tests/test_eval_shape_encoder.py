"""Tests for scripts/eval_shape_encoder.py

Pre-implementation spec tests for Pathway-B contrastive shape encoder evaluation
utilities. Written by test-architect BEFORE any implementation exists. All tests
are expected to fail with ImportError or ModuleNotFoundError until the module
is created.

ROADMAP test plan coverage:
  1. eta2: two distinct groups -> η² ≈ 1.0                  -> test_eta2_perfect_separation
  2. eta2: single cluster -> η² ≈ 0.0                       -> test_eta2_single_cluster
  3. eta2: lab<0 rows excluded                               -> test_eta2_ignores_noise_labels
  4. eta2: random labels/values -> η² small                  -> test_eta2_random_labels_small
  5. knn_purity: well-separated blobs -> overall ≥ 0.95      -> test_knn_purity_well_separated
  6. knn_purity: overlapping blob, random types -> ~0.5      -> test_knn_purity_random_assignment
  7. knn_purity: self excluded from neighbors                -> test_knn_purity_self_excluded

Additional coverage (recurring gap patterns):
  - eta2: empty keep-set (all labels <0) -> returns 0.0     -> test_eta2_all_noise_labels
  - eta2: 1D input (ndim==1) auto-promoted to 2D            -> test_eta2_1d_input
  - eta2: exact reference impl match on hand-computed case  -> test_eta2_hand_computed_exact
  - knn_purity: per-type keys present in return dict        -> test_knn_purity_returns_per_type_keys
  - knn_purity: overall key present in return dict          -> test_knn_purity_returns_overall_key
  - knn_purity: k >= n edge handled (no crash)              -> test_knn_purity_k_larger_than_n
  - knn_purity: single type -> per-type purity=1.0          -> test_knn_purity_single_type
  - eta2: two groups, tot=0 (constant vector) -> 0.0        -> test_eta2_zero_variance_returns_0

Total: 15 tests (7 from ROADMAP, 8 additional)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module loading via importlib (scripts/ is not an installed package)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = REPO_ROOT / "scripts" / "eval_shape_encoder.py"

_MODULE: ModuleType | None = None
_IMPORT_ERROR: str | None = None

if _MODULE_PATH.exists():
    try:
        spec = importlib.util.spec_from_file_location(
            "eval_shape_encoder", _MODULE_PATH
        )
        _MODULE = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules["eval_shape_encoder"] = _MODULE
        spec.loader.exec_module(_MODULE)  # type: ignore[union-attr]
    except Exception as exc:
        _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = (
        f"Module file not found: {_MODULE_PATH}  "
        "(expected after implementation)"
    )


def _require_module() -> ModuleType:
    """Return the loaded module or raise a clear skip."""
    if _MODULE is None:
        pytest.skip(
            f"eval_shape_encoder not yet implemented: {_IMPORT_ERROR}"
        )
    return _MODULE  # type: ignore[return-value]


# ===========================================================================
# eta2 tests
# ===========================================================================


class TestEta2:
    """Tests for eta2(v, lab) -> float."""

    def test_eta2_perfect_separation(self):
        """Spec: two groups, each a distinct constant vector -> η² ≈ 1.0.

        Group 0: all rows = [0, 0]
        Group 1: all rows = [10, 10]
        Within-group variance is zero; all variance is between groups -> η² = 1.0.
        """
        mod = _require_module()
        rng = np.random.default_rng(0)
        n = 100
        v = np.zeros((n, 2), dtype=float)
        lab = np.zeros(n, dtype=int)
        v[n // 2 :] = 10.0
        lab[n // 2 :] = 1

        result = mod.eta2(v, lab)

        assert isinstance(result, float), f"eta2 should return float, got {type(result)}"
        assert abs(result - 1.0) < 1e-6, (
            f"Perfect separation: expected η²=1.0, got {result:.8f}"
        )

    def test_eta2_single_cluster(self):
        """Spec: single cluster (all same label) -> η² ≈ 0.0.

        With one cluster the group mean equals the global mean; all variance is
        'within', so 1 - within/total = 0.
        """
        mod = _require_module()
        rng = np.random.default_rng(1)
        v = rng.standard_normal((200, 4))
        lab = np.zeros(200, dtype=int)  # single label

        result = mod.eta2(v, lab)

        assert abs(result) < 1e-10, (
            f"Single cluster: expected η²=0.0, got {result:.2e}"
        )

    def test_eta2_ignores_noise_labels(self):
        """Spec: rows with lab < 0 are excluded; their presence must not change score.

        Construct two perfectly separated groups (lab 0 and 1). Adding many noisy
        rows labeled -1 should leave η² at 1.0 unchanged.
        """
        mod = _require_module()
        n_per_group = 50
        v_clean = np.vstack([
            np.zeros((n_per_group, 2)),
            np.full((n_per_group, 2), 10.0),
        ])
        lab_clean = np.array([0] * n_per_group + [1] * n_per_group)
        score_clean = mod.eta2(v_clean, lab_clean)

        # Add 200 noise rows labeled -1 with arbitrary values
        rng = np.random.default_rng(2)
        v_noisy = np.vstack([v_clean, rng.standard_normal((200, 2))])
        lab_noisy = np.concatenate([lab_clean, np.full(200, -1, dtype=int)])
        score_noisy = mod.eta2(v_noisy, lab_noisy)

        assert abs(score_clean - 1.0) < 1e-6, (
            f"Baseline clean score should be 1.0, got {score_clean}"
        )
        assert abs(score_noisy - score_clean) < 1e-6, (
            f"Noise rows (lab=-1) changed η²: clean={score_clean:.8f}, "
            f"noisy={score_noisy:.8f}"
        )

    def test_eta2_random_labels_small(self):
        """Spec: random labels over random values -> η² small (< 0.2) for n≥2000.

        With large n and random group assignment, the group means converge to
        the global mean, so between-group variance is negligible.
        """
        mod = _require_module()
        rng = np.random.default_rng(42)
        n = 2000
        v = rng.standard_normal((n, 8))
        lab = rng.integers(0, 10, size=n)

        result = mod.eta2(v, lab)

        assert result < 0.2, (
            f"Random labels (n={n}): expected η² < 0.2 (near-zero), got {result:.4f}"
        )

    def test_eta2_all_noise_labels(self):
        """Edge case: all labels < 0 -> keep-set is empty -> return 0.0."""
        mod = _require_module()
        v = np.random.default_rng(3).standard_normal((50, 4))
        lab = np.full(50, -1, dtype=int)

        result = mod.eta2(v, lab)

        assert result == 0.0, (
            f"All-noise labels: expected 0.0, got {result}"
        )

    def test_eta2_1d_input(self):
        """Edge case: 1-D v (ndim==1) should be auto-promoted to (n,1).

        The spec says: v = v if v.ndim==2 else v[:,None]. The function must
        not crash and must return the correct η² for a scalar feature.
        """
        mod = _require_module()
        # 1-D: two groups at 0 and 10 -> perfect separation
        v = np.array([0.0] * 50 + [10.0] * 50)
        lab = np.array([0] * 50 + [1] * 50)

        result = mod.eta2(v, lab)

        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert abs(result - 1.0) < 1e-6, (
            f"1-D perfect separation: expected η²=1.0, got {result:.8f}"
        )

    def test_eta2_hand_computed_exact(self):
        """Spec invariant: verify η² against a hand-computed reference.

        Setup: 3 groups, each of 4 scalar values.
          Group 0: [0, 0, 0, 0]  (mean=0)
          Group 1: [2, 2, 2, 2]  (mean=2)
          Group 2: [4, 4, 4, 4]  (mean=4)

        Global mean = (0*4 + 2*4 + 4*4) / 12 = 24/12 = 2.0
        Total SS = 4*(0-2)^2 + 4*(2-2)^2 + 4*(4-2)^2
                 = 4*4 + 0 + 4*4 = 32
        Within SS = sum of within-group variance = 0 (each group is constant)
        η² = 1 - 0/32 = 1.0

        Now perturb: Group 0 = [0,0,1,1] (mean=0.5), Group 1=[2,2,3,3] (mean=2.5),
                     Group 2 = [4,4,5,5] (mean=4.5)
        Global mean = (0+0+1+1+2+2+3+3+4+4+5+5)/12 = 30/12 = 2.5
        Total SS = (0-2.5)^2*2+(1-2.5)^2*2+(2-2.5)^2*2+(3-2.5)^2*2+(4-2.5)^2*2+(5-2.5)^2*2
                 = 2*(6.25+2.25+0.25+0.25+2.25+6.25) = 2*17.5 = 35
        Within SS = [2*(0-0.5)^2+2*(1-0.5)^2] + [2*(2-2.5)^2+2*(3-2.5)^2]
                  + [2*(4-4.5)^2+2*(5-4.5)^2]
                  = 3 * (2*0.25 + 2*0.25) = 3 * 1.0 = 3
        η² = 1 - 3/35 = 32/35 ≈ 0.914286
        """
        mod = _require_module()

        v = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
        lab = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        result = mod.eta2(v, lab)

        expected = 32.0 / 35.0  # hand-computed above
        assert abs(result - expected) < 1e-6, (
            f"Hand-computed check: expected {expected:.6f}, got {result:.6f}"
        )

    def test_eta2_zero_variance_returns_0(self):
        """Edge case: all values identical (tot SS = 0) -> returns 0.0, not NaN."""
        mod = _require_module()
        v = np.ones((40, 3))
        lab = np.array([0] * 20 + [1] * 20)

        result = mod.eta2(v, lab)

        assert result == 0.0, (
            f"Zero-variance input: expected 0.0, got {result}"
        )
        assert not (result != result), "eta2 returned NaN on zero-variance input"


# ===========================================================================
# knn_purity tests
# ===========================================================================


def _make_blobs(
    n_per_cluster: int,
    centers: list[list[float]],
    std: float,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create Gaussian blobs at given centers. Returns (Z, types)."""
    rng = np.random.default_rng(seed)
    parts_z, parts_t = [], []
    for label, center in enumerate(centers):
        z = rng.standard_normal((n_per_cluster, len(center))) * std
        z += np.array(center)
        parts_z.append(z)
        parts_t.append(np.full(n_per_cluster, label))
    return np.vstack(parts_z), np.concatenate(parts_t)


class TestKnnPurity:
    """Tests for knn_purity(Z, types, k) -> dict."""

    def test_knn_purity_well_separated(self):
        """Spec: two well-separated Gaussian blobs -> overall purity >= 0.95."""
        mod = _require_module()
        n = 200
        Z, types = _make_blobs(
            n, centers=[[0.0, 0.0], [100.0, 0.0]], std=1.0, seed=10
        )
        result = mod.knn_purity(Z, types, k=10)

        assert "overall" in result, f"knn_purity must return 'overall' key, got {result.keys()}"
        assert result["overall"] >= 0.95, (
            f"Well-separated blobs: overall purity={result['overall']:.4f}, expected >= 0.95"
        )

    def test_knn_purity_random_assignment(self):
        """Spec: overlapping blob with random 50/50 type labels -> purity ≈ 0.5 (±0.1)."""
        mod = _require_module()
        rng = np.random.default_rng(20)
        n = 400
        Z = rng.standard_normal((n, 4))  # single blob, no separation
        types = rng.integers(0, 2, size=n)  # random 50/50 labels

        result = mod.knn_purity(Z, types, k=10)

        assert "overall" in result, f"knn_purity must return 'overall' key"
        assert abs(result["overall"] - 0.5) <= 0.1, (
            f"Random 50/50 labels in single blob: overall purity={result['overall']:.4f}, "
            f"expected 0.5 ± 0.1"
        )

    def test_knn_purity_self_excluded(self):
        """Spec: self is excluded from the k neighbors (no identity contamination).

        Strategy: construct a dataset where each point's ONLY identical neighbor
        is itself. If self is included, purity would be 1.0 for that point;
        if self is excluded, the true purity among actual k neighbors applies.

        Concretely: n=2 points per class, k=1. Two classes far apart but each
        class has exactly 2 identical points. The 1-NN of each point (excluding
        self) is its classmate -> purity=1.0 is still correct here.

        Better: n=3 per class arranged [A, A, B] with classes 0 and 1, k=1.
        Each 'A' point's nearest non-self neighbor is the other 'A' (same class)
        but if self is included it would return the self (trivially same class too).
        Instead we check the COUNT of neighbors per query: result should use
        exactly k=1 *other* points, never the query itself.

        We verify with a deterministic hand-crafted case:
          Points: p0=[0,0] (type 0), p1=[1,0] (type 1), p2=[2,0] (type 1)
          k=1, no self: p0's NN is p1 (type 1) -> purity_for_p0 = 0
                        p1's NN is p2 (type 1) -> purity_for_p1 = 1
                        p2's NN is p1 (type 1) -> purity_for_p2 = 1
          Overall = (0+1+1)/3 ≈ 0.667
          If self included: p1's NN is itself (type 1) -> still 1;
            p0's NN is still p1 (self is type 0, so p0 excluded anyway);
            same result. Rely on p1: its NN MUST be p2, not itself.
        """
        mod = _require_module()
        # p2 sits at 1.5 (not 2.0) so p1's nearest NON-self neighbour is
        # unambiguously p2 (dist 0.5) rather than an exact p0/p2 tie at dist 1.0
        # that sklearn would break toward the lower index. This keeps the
        # self-exclusion intent while making the expected value tie-independent.
        Z = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0]])
        types = np.array([0, 1, 1])

        result = mod.knn_purity(Z, types, k=1)

        # p0 -> NN=p1 (type 1, wrong): purity 0
        # p1 -> NN=p2 if self excluded (type 1, right): purity 1
        # p2 -> NN=p1 (type 1, right): purity 1
        expected_overall = (0 + 1 + 1) / 3
        assert "overall" in result
        assert abs(result["overall"] - expected_overall) < 1e-6, (
            f"Self-exclusion check: expected overall={expected_overall:.6f}, "
            f"got {result['overall']:.6f}. "
            "If you got 1.0, self was included as its own neighbor."
        )

    def test_knn_purity_returns_per_type_keys(self):
        """Additional: result dict contains a key for each type label present."""
        mod = _require_module()
        Z, types = _make_blobs(50, centers=[[0.0], [5.0], [10.0]], std=0.5, seed=30)
        result = mod.knn_purity(Z, types, k=5)

        for t in np.unique(types):
            assert t in result, (
                f"knn_purity result missing per-type key for type {t}. "
                f"Keys present: {list(result.keys())}"
            )

    def test_knn_purity_returns_overall_key(self):
        """Additional: result dict always contains an 'overall' key."""
        mod = _require_module()
        Z = np.random.default_rng(40).standard_normal((30, 4))
        types = np.array([0] * 15 + [1] * 15)
        result = mod.knn_purity(Z, types, k=5)

        assert "overall" in result, (
            f"knn_purity must return an 'overall' key. Got keys: {list(result.keys())}"
        )
        assert isinstance(result["overall"], float), (
            f"'overall' value must be a float, got {type(result['overall'])}"
        )

    def test_knn_purity_k_larger_than_n(self):
        """Edge case: k >= n should not crash; use all available non-self neighbors."""
        mod = _require_module()
        Z = np.array([[0.0], [1.0], [2.0], [3.0]])
        types = np.array([0, 0, 1, 1])

        # k=10 >> n=4; function should clamp or handle gracefully
        try:
            result = mod.knn_purity(Z, types, k=10)
        except Exception as exc:
            pytest.fail(
                f"knn_purity raised exception with k > n: {exc}"
            )

        assert "overall" in result
        assert 0.0 <= result["overall"] <= 1.0, (
            f"Purity out of [0,1] range: {result['overall']}"
        )

    def test_knn_purity_single_type(self):
        """Edge case: all points have the same type -> per-type purity = 1.0."""
        mod = _require_module()
        rng = np.random.default_rng(50)
        Z = rng.standard_normal((30, 4))
        types = np.zeros(30, dtype=int)

        result = mod.knn_purity(Z, types, k=5)

        assert result["overall"] == pytest.approx(1.0, abs=1e-6), (
            f"Single type: all neighbors share type -> purity=1.0, got {result['overall']}"
        )
        # Per-type key for label 0 should also be 1.0
        assert 0 in result
        assert result[0] == pytest.approx(1.0, abs=1e-6), (
            f"Single type per-type key: expected 1.0, got {result[0]}"
        )
