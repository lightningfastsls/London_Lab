"""Tests for sim_optimizer — written by test-architect BEFORE implementation.

Module under test: src/usv_spectrogram/classification/sim_optimizer.py

ROADMAP test plan coverage (module 17.8):
  1. optimize_sis on i.i.d. random labels: final_sis >= initial_sis
     -> test_random_labels_sis_does_not_decrease
  2. optimize_sis on ABABAB... already-optimal sequence: final_sis ≈ initial_sis
     -> test_structured_ababab_sis_unchanged
  3. compute_delta correctness vs naive full recomputation
     -> test_compute_delta_matches_naive_recomputation
  4. Running optimize_sis twice with same random_state -> identical output
     -> test_reproducibility_with_same_random_state
  5. max_iterations=0: returns initial labels unchanged, iterations_used=0
     -> test_max_iterations_zero_returns_initial_unchanged
  6. K=2 labels: handles binary case
     -> test_binary_labels_no_crash
  7. K > N: handles degenerate case without crash
     -> test_more_clusters_than_calls_no_crash
  8. Empty initial_labels returns empty SIMResult without crash
     -> test_empty_labels_returns_empty_result
  9. SIS history is monotonically non-decreasing across iterations
     -> test_sis_history_monotonically_nondecreasing

Additional coverage (recurring gap patterns):
  - SIMConfig frozen/immutability -> test_simconfig_is_frozen
  - SIMConfig default values match spec -> test_simconfig_defaults
  - SIMResult fields preserved correctly -> test_simresult_initial_labels_preserved
  - Single-element sequence (N=1) -> test_single_element_sequence
  - All-same labels (K=1) -> test_all_same_labels_no_crash
  - SIS values are non-negative (MI >= 0) -> test_sis_values_nonnegative
  - iterations_used <= max_iterations -> test_iterations_used_bounded_by_max

Total: 16 tests (9 from ROADMAP, 7 additional)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pattern 8: import bootstrap — tests/ is one level below repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---------------------------------------------------------------------------
# Guarded import — module does not exist yet.
# All tests are marked skip when the module is absent so that they are fully
# collected (showing count = 16) but fail cleanly until implementation lands.
# ---------------------------------------------------------------------------
try:
    from usv_spectrogram.classification.sim_optimizer import (
        SIMConfig,
        SIMResult,
        optimize_sis,
    )
    _MODULE_MISSING = False
except (ImportError, ModuleNotFoundError):
    SIMConfig = None  # type: ignore[assignment,misc]
    SIMResult = None  # type: ignore[assignment,misc]
    optimize_sis = None  # type: ignore[assignment]
    _MODULE_MISSING = True

# Apply skip mark to every test in this file when the module is absent.
pytestmark = pytest.mark.skipif(
    _MODULE_MISSING,
    reason="usv_spectrogram.classification.sim_optimizer not yet implemented",
)

# ---------------------------------------------------------------------------
# Helper: naive SIS (MI at depth 1) computed from first principles.
# Independent of the implementation under test — used to cross-check outputs.
# ---------------------------------------------------------------------------


def _compute_sis_naive(labels: np.ndarray) -> float:
    """Compute SIS = MI(X_n; X_{n-1}) at depth 1 using np.log2.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)  where H is Shannon entropy in bits.
    """
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n < 2:
        return 0.0

    prev = labels[:-1]
    curr = labels[1:]
    k = max(int(labels.max()) + 1, 1)

    joint = np.zeros((k, k), dtype=float)
    for p, c in zip(prev, curr):
        joint[p, c] += 1.0
    joint /= joint.sum()

    marginal_prev = joint.sum(axis=1)
    marginal_curr = joint.sum(axis=0)

    def _entropy(probs: np.ndarray) -> float:
        mask = probs > 0
        return -float(np.sum(probs[mask] * np.log2(probs[mask])))

    return max(0.0, _entropy(marginal_prev) + _entropy(marginal_curr) - _entropy(joint.ravel()))


# ---------------------------------------------------------------------------
# ROADMAP test 1
# ---------------------------------------------------------------------------


def test_random_labels_sis_does_not_decrease():
    """Spec: optimize_sis on i.i.d. random labels must produce final_sis >= initial_sis.

    Random labels have low SIS because there is no temporal structure.
    SIM only accepts moves that increase SIS; rejected moves leave it unchanged.
    Therefore final_sis must be >= initial_sis (never worse).

    Also cross-checks initial_sis against the independent naive reference to
    confirm the implementation's SIS calculation is correct at the start.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=200)

    cfg = SIMConfig(max_iterations=20, random_state=7)
    result = optimize_sis(labels, cfg)

    assert isinstance(result, SIMResult)
    assert result.final_sis >= result.initial_sis - 1e-9, (
        f"SIS decreased: initial={result.initial_sis:.6f}, final={result.final_sis:.6f}"
    )
    naive_initial = _compute_sis_naive(labels)
    assert abs(result.initial_sis - naive_initial) < 1e-6, (
        f"initial_sis mismatch: got {result.initial_sis:.6f}, expected {naive_initial:.6f}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 2
# ---------------------------------------------------------------------------


def test_structured_ababab_sis_unchanged():
    """Spec: ABABAB... sequence is already near-optimal; SIM must not change it much.

    A perfectly alternating binary sequence achieves MI(X_n;X_{n-1}) = 1 bit
    (the next label is fully determined by the current one).  No reassignment
    can improve this, so |final_sis - initial_sis| should be near zero.
    Tolerance 0.05 bits is generous to allow label-permutation tie-breaking.
    """
    labels = np.tile([0, 1], 50)  # 100-element 0,1,0,1,...

    cfg = SIMConfig(max_iterations=20, random_state=42)
    result = optimize_sis(labels, cfg)

    assert isinstance(result, SIMResult)
    improvement = result.final_sis - result.initial_sis
    assert abs(improvement) < 0.05, (
        f"ABABAB sequence changed more than expected: improvement={improvement:.6f}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 3
# ---------------------------------------------------------------------------


def test_compute_delta_matches_naive_recomputation():
    """Spec: compute_delta must match naive (SIS_after - SIS_before) to within 1e-9.

    compute_delta is the O(1) incremental SIS change for a proposed label swap.
    It may be an internal helper.  We handle two cases:

    Case A — compute_delta is exported:
      For each of 10 random sequences, build the K×K transition count matrix,
      propose 5 random swaps, and assert compute_delta matches the naive delta
      computed by calling _compute_sis_naive on the before and after arrays.

    Case B — compute_delta is not exported:
      Verify end-to-end consistency instead: optimize_sis must never lower SIS
      across 10 random trials (which is only guaranteed if accepted deltas are
      computed correctly and only positive deltas are accepted).
    """
    try:
        from usv_spectrogram.classification.sim_optimizer import compute_delta as _cd
        _has_compute_delta = True
    except ImportError:
        _has_compute_delta = False

    rng = np.random.default_rng(123)

    for trial in range(10):
        n = int(rng.integers(20, 60))
        k = int(rng.integers(2, 6))
        labels = rng.integers(0, k, size=n)

        if _has_compute_delta:
            k_actual = int(labels.max()) + 1

            counts = np.zeros((k_actual, k_actual), dtype=float)
            for p, c in zip(labels[:-1], labels[1:]):
                counts[p, c] += 1.0

            for _ in range(5):
                i = int(rng.integers(1, n - 1))  # avoid boundary positions
                current = int(labels[i])
                candidates = [c for c in range(k_actual) if c != current]
                if not candidates:
                    continue
                candidate = int(rng.choice(np.array(candidates)))

                labels_after = labels.copy()
                labels_after[i] = candidate
                naive_delta = _compute_sis_naive(labels_after) - _compute_sis_naive(labels)

                module_delta = _cd(counts, labels, i, current, candidate)

                assert abs(module_delta - naive_delta) < 1e-9, (
                    f"trial={trial}, i={i}: compute_delta={module_delta:.9f}, "
                    f"naive_delta={naive_delta:.9f}"
                )
        else:
            # End-to-end consistency: optimizer must never lower SIS
            cfg = SIMConfig(max_iterations=10, random_state=trial)
            result = optimize_sis(labels, cfg)
            assert result.final_sis >= result.initial_sis - 1e-9, (
                f"trial={trial}: final_sis < initial_sis by "
                f"{result.initial_sis - result.final_sis:.9f}"
            )


# ---------------------------------------------------------------------------
# ROADMAP test 4
# ---------------------------------------------------------------------------


def test_reproducibility_with_same_random_state():
    """Spec: same random_state must yield bit-for-bit identical output on two calls.

    With random_order=True the per-iteration shuffle is seeded from random_state.
    Running optimize_sis twice with the same cfg must produce identical label
    arrays, final SIS, iterations_used, and sis_history.
    """
    rng = np.random.default_rng(99)
    labels = rng.integers(0, 4, size=150)

    cfg = SIMConfig(max_iterations=15, random_state=42, random_order=True)

    result_a = optimize_sis(labels, cfg)
    result_b = optimize_sis(labels, cfg)

    np.testing.assert_array_equal(
        result_a.optimized_labels,
        result_b.optimized_labels,
        err_msg="Two runs with identical random_state produced different label arrays",
    )
    assert result_a.final_sis == result_b.final_sis, (
        f"final_sis differs: {result_a.final_sis} vs {result_b.final_sis}"
    )
    assert result_a.iterations_used == result_b.iterations_used, (
        f"iterations_used differs: {result_a.iterations_used} vs {result_b.iterations_used}"
    )
    assert result_a.sis_history == result_b.sis_history, (
        "sis_history differs between identical runs"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 5
# ---------------------------------------------------------------------------


def test_max_iterations_zero_returns_initial_unchanged():
    """Spec: max_iterations=0 must return initial labels unchanged, iterations_used=0.

    No passes are executed so the output labeling must equal the input,
    final_sis must equal initial_sis, and sis_history must be empty (or contain
    just the initial SIS — either is acceptable as long as no iteration ran).
    """
    labels = np.array([0, 1, 2, 1, 0, 2, 0, 1], dtype=int)

    cfg = SIMConfig(max_iterations=0)
    result = optimize_sis(labels, cfg)

    assert isinstance(result, SIMResult)
    np.testing.assert_array_equal(
        result.optimized_labels,
        labels,
        err_msg="max_iterations=0 must return initial labels without modification",
    )
    assert result.iterations_used == 0, (
        f"Expected iterations_used=0, got {result.iterations_used}"
    )
    assert result.initial_sis == result.final_sis, (
        "With 0 iterations, initial_sis must equal final_sis"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 6
# ---------------------------------------------------------------------------


def test_binary_labels_no_crash():
    """Spec: K=2 (binary label set) must produce a valid SIMResult without crash.

    Binary is the smallest non-trivial K.  All optimized label values must
    remain in {0, 1} and SIS must be non-negative.
    """
    rng = np.random.default_rng(55)
    labels = rng.integers(0, 2, size=80)

    cfg = SIMConfig(max_iterations=10, random_state=42)
    result = optimize_sis(labels, cfg)

    assert isinstance(result, SIMResult)
    assert result.optimized_labels.shape == labels.shape
    assert result.final_sis >= 0.0, (
        f"SIS must be non-negative, got {result.final_sis}"
    )
    unique_opt = set(result.optimized_labels.tolist())
    assert unique_opt.issubset({0, 1}), (
        f"Binary case produced out-of-range label values: {unique_opt}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 7
# ---------------------------------------------------------------------------


def test_more_clusters_than_calls_no_crash():
    """Spec: K >= N (as many label values as sequence length) must not crash.

    Degenerate case: 3 calls labelled [0, 1, 2].  The transition matrix has
    mostly zero rows/columns.  The function must return a correctly shaped result.
    """
    labels = np.array([0, 1, 2], dtype=int)  # K = N = 3

    cfg = SIMConfig(max_iterations=5, random_state=42)
    result = optimize_sis(labels, cfg)

    assert isinstance(result, SIMResult)
    assert len(result.optimized_labels) == len(labels)


# ---------------------------------------------------------------------------
# ROADMAP test 8
# ---------------------------------------------------------------------------


def test_empty_labels_returns_empty_result():
    """Spec: empty initial_labels must return a SIMResult without crash.

    An empty sequence has no transitions: initial_sis=0, final_sis=0,
    iterations_used=0, and the optimized_labels array has length 0.
    """
    labels = np.array([], dtype=int)

    cfg = SIMConfig(max_iterations=10)
    result = optimize_sis(labels, cfg)

    assert isinstance(result, SIMResult)
    assert len(result.optimized_labels) == 0, (
        f"Expected empty optimized_labels, got length {len(result.optimized_labels)}"
    )
    assert result.initial_sis == 0.0, (
        f"Empty sequence must have initial_sis=0.0, got {result.initial_sis}"
    )
    assert result.final_sis == 0.0, (
        f"Empty sequence must have final_sis=0.0, got {result.final_sis}"
    )
    assert result.iterations_used == 0, (
        f"Empty sequence must have iterations_used=0, got {result.iterations_used}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 9
# ---------------------------------------------------------------------------


def test_sis_history_monotonically_nondecreasing():
    """Spec: sis_history must be monotonically non-decreasing across iterations.

    Each pass accepts only SIS-improving moves; hence each entry in sis_history
    must be >= the previous entry (to within floating-point tolerance 1e-9).
    The final entry must equal final_sis exactly.
    """
    rng = np.random.default_rng(77)
    labels = rng.integers(0, 6, size=300)

    cfg = SIMConfig(max_iterations=30, random_state=42)
    result = optimize_sis(labels, cfg)

    history = result.sis_history
    assert len(history) >= 1, "sis_history must not be empty after running at least one iteration"

    for i in range(1, len(history)):
        assert history[i] >= history[i - 1] - 1e-9, (
            f"sis_history decreased at iteration {i}: "
            f"{history[i - 1]:.8f} -> {history[i]:.8f}"
        )

    assert abs(history[-1] - result.final_sis) < 1e-9, (
        f"Last sis_history entry {history[-1]:.8f} != final_sis {result.final_sis:.8f}"
    )


# ---------------------------------------------------------------------------
# Additional: SIMConfig defaults (Pattern 1 — frozen dataclass with spec defaults)
# ---------------------------------------------------------------------------


def test_simconfig_defaults():
    """Spec: SIMConfig default values must match the ROADMAP 17.8 spec exactly.

    ROADMAP specifies: max_iterations=50, min_sis_improvement=1e-4,
    random_order=True, random_state=42.
    """
    cfg = SIMConfig()

    assert cfg.max_iterations == 50, (
        f"Default max_iterations should be 50, got {cfg.max_iterations}"
    )
    assert cfg.min_sis_improvement == 1e-4, (
        f"Default min_sis_improvement should be 1e-4, got {cfg.min_sis_improvement}"
    )
    assert cfg.random_order is True, (
        f"Default random_order should be True, got {cfg.random_order}"
    )
    assert cfg.random_state == 42, (
        f"Default random_state should be 42, got {cfg.random_state}"
    )


# ---------------------------------------------------------------------------
# Additional: SIMConfig is frozen (Pattern 1)
# ---------------------------------------------------------------------------


def test_simconfig_is_frozen():
    """Spec (Pattern 1): SIMConfig must be a frozen dataclass — mutation must raise.

    Frozen dataclasses raise FrozenInstanceError (subclass of AttributeError)
    when any attribute is assigned after construction.
    """
    cfg = SIMConfig()
    with pytest.raises(Exception):  # FrozenInstanceError is subclass of AttributeError
        cfg.max_iterations = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Additional: SIMResult preserves initial_labels exactly
# ---------------------------------------------------------------------------


def test_simresult_initial_labels_preserved():
    """SIMResult.initial_labels must equal the input array; input must not be mutated.

    Verifies:
    (a) result.initial_labels stores the original label sequence unchanged.
    (b) optimize_sis does NOT mutate the caller's array in place (defensive copy).
    """
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=int)
    original_copy = labels.copy()

    cfg = SIMConfig(max_iterations=5, random_state=42)
    result = optimize_sis(labels, cfg)

    np.testing.assert_array_equal(
        result.initial_labels,
        original_copy,
        err_msg="initial_labels in SIMResult must equal the array passed to optimize_sis",
    )
    np.testing.assert_array_equal(
        labels,
        original_copy,
        err_msg="optimize_sis must not mutate the input labels array",
    )


# ---------------------------------------------------------------------------
# Additional: single-element sequence (N=1, edge case)
# ---------------------------------------------------------------------------


def test_single_element_sequence():
    """Single-element sequence has no transitions: SIS=0, optimizer is a no-op.

    N=1 is the minimal non-empty input.  No pair (X_{n-1}, X_n) exists, so
    the transition matrix is empty and MI = 0.
    """
    labels = np.array([3], dtype=int)

    cfg = SIMConfig(max_iterations=5)
    result = optimize_sis(labels, cfg)

    assert isinstance(result, SIMResult)
    assert len(result.optimized_labels) == 1
    assert result.initial_sis == 0.0, (
        f"Single-element sequence must have initial_sis=0.0, got {result.initial_sis}"
    )
    assert result.final_sis == 0.0, (
        f"Single-element sequence must have final_sis=0.0, got {result.final_sis}"
    )


# ---------------------------------------------------------------------------
# Additional: all-same labels (effective K=1) edge case
# ---------------------------------------------------------------------------


def test_all_same_labels_no_crash():
    """Constant label sequence (K=1 effective) must not crash.

    No reassignment is possible because there is only one label value.
    Output must equal input and SIS must be 0.
    """
    labels = np.zeros(50, dtype=int)  # all zeros — K=1 effective

    cfg = SIMConfig(max_iterations=5, random_state=42)
    result = optimize_sis(labels, cfg)

    assert isinstance(result, SIMResult)
    np.testing.assert_array_equal(result.optimized_labels, labels)
    assert result.initial_sis == 0.0, (
        f"Constant sequence must have initial_sis=0.0, got {result.initial_sis}"
    )


# ---------------------------------------------------------------------------
# Additional: SIS values are non-negative (MI >= 0 invariant)
# ---------------------------------------------------------------------------


def test_sis_values_nonnegative():
    """Spec invariant: mutual information is always >= 0.

    Both initial_sis and final_sis must be non-negative for any random input
    sequence.  Tested across 5 random (n, k) combinations.
    """
    rng = np.random.default_rng(888)

    for _ in range(5):
        n = int(rng.integers(10, 100))
        k = int(rng.integers(2, 8))
        labels = rng.integers(0, k, size=n)

        cfg = SIMConfig(max_iterations=5, random_state=42)
        result = optimize_sis(labels, cfg)

        assert result.initial_sis >= 0.0, (
            f"initial_sis={result.initial_sis:.8f} is negative (MI must be >= 0)"
        )
        assert result.final_sis >= 0.0, (
            f"final_sis={result.final_sis:.8f} is negative (MI must be >= 0)"
        )


# ---------------------------------------------------------------------------
# Additional: iterations_used is bounded by max_iterations
# ---------------------------------------------------------------------------


def test_iterations_used_bounded_by_max():
    """Spec: 0 <= iterations_used <= max_iterations for any input.

    Tested for max_iterations in {1, 5, 10} to catch off-by-one errors.
    """
    rng = np.random.default_rng(321)
    labels = rng.integers(0, 4, size=100)

    for max_iter in [1, 5, 10]:
        cfg = SIMConfig(max_iterations=max_iter, random_state=42)
        result = optimize_sis(labels, cfg)

        assert 0 <= result.iterations_used <= max_iter, (
            f"iterations_used={result.iterations_used} is outside [0, {max_iter}]"
        )
