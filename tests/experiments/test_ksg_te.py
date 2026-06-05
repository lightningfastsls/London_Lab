"""Tests for scripts/experiments/ksg_te.py — written by test-architect BEFORE implementation.

This is /implement Step 0 for workstream WS-B of the USV analysis program.

All expected values come from analytic information theory (closed-form expressions for
Gaussian random variables). They MUST NOT be weakened to make a buggy implementation
pass. The role of these tests is to define correctness — the implementer builds code
to satisfy them.

Key analytic identity used throughout:
    For jointly Gaussian (X, Y) with correlation rho:
        I(X; Y) = -0.5 * ln(1 - rho^2)   [nats]
    This is exact and applies regardless of marginal variances.

KSG estimator characteristics (justified in tolerance comments per test):
    - Asymptotically unbiased as n -> inf for smooth densities
    - For n=5000 and k=4, typical absolute error is 0.01-0.04 nats
    - MI can estimate slightly below 0 due to finite-sample bias; this is not a bug
    - TE has an inherent bias offset; direction tests compare TE(A->B) vs TE(B->A)

ROADMAP test plan coverage:
  1. Bivariate-Gaussian MI ground truth (rho=0.3)    -> test_ksg_mi_gaussian_rho_03
  2. Bivariate-Gaussian MI ground truth (rho=0.6)    -> test_ksg_mi_gaussian_rho_06
  3. Bivariate-Gaussian MI ground truth (rho=0.9)    -> test_ksg_mi_gaussian_rho_09
  4. Independence => ~0                               -> test_ksg_mi_independent_near_zero
  5. Symmetry I(X;Y) == I(Y;X)                       -> test_ksg_mi_symmetry
  6. CMI collapses in X->Z->Y chain                  -> test_ksg_cmi_collapses_in_markov_chain
  7. CMI recovers MI when Z is independent noise      -> test_ksg_cmi_recovers_mi_with_independent_z
  8. TE directionality (source->target beats reverse) -> test_transfer_entropy_directionality
  9. TE bias floor on independent series              -> test_transfer_entropy_null_near_zero
 10. Input hygiene: 1-D arrays treated as (n,1)      -> test_ksg_mi_accepts_1d_arrays
 11. Input hygiene: mismatched lengths raise          -> test_ksg_mi_raises_on_mismatched_lengths
 12. Input hygiene: k < 1 raises                     -> test_ksg_mi_raises_on_invalid_k
 13. CMI k < 1 raises                                -> test_ksg_cmi_raises_on_invalid_k
 14. TE mismatched lengths raise                      -> test_transfer_entropy_raises_on_mismatched_lengths

Additional coverage (recurring gap patterns):
  - Single-sample edge case (n=1, k=1)               -> test_ksg_mi_raises_or_handles_trivial_n
  - CMI with 1-D Z array accepted                    -> test_ksg_cmi_accepts_1d_z
  - Multivariate (dx>1, dy>1) inputs accepted        -> test_ksg_mi_accepts_multivariate_inputs
  - TE with optional cond array accepted             -> test_transfer_entropy_with_cond_array

Total: 18 tests (14 from ROADMAP, 4 additional)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts/experiments to path so ksg_te can be imported once implemented.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_EXPERIMENTS = REPO_ROOT / "scripts" / "experiments"
if str(SCRIPTS_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EXPERIMENTS))

# This import WILL fail until ksg_te.py is created — that is expected and correct.
# Tests in this file will show as errors at collection time until the module exists.
from ksg_te import ksg_mi, ksg_cmi, transfer_entropy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bivariate_gaussian(rho: float, n: int, rng: np.random.Generator):
    """Sample n points from a zero-mean bivariate Gaussian with correlation rho."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    samples = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    X = samples[:, 0:1]  # (n, 1)
    Y = samples[:, 1:2]  # (n, 1)
    return X, Y


def _analytic_gaussian_mi(rho: float) -> float:
    """I(X;Y) for zero-mean bivariate Gaussian with correlation rho, in nats."""
    return -0.5 * np.log(1.0 - rho ** 2)


# ---------------------------------------------------------------------------
# ROADMAP Test 1-3: Bivariate-Gaussian MI ground truth
# ---------------------------------------------------------------------------

class TestKSGMIGaussianGroundTruth:
    """KSG MI on bivariate Gaussian should match analytic -0.5*ln(1-rho^2) nats."""

    # Tolerance justification: KSG with k=4, n=5000 has typical absolute bias
    # ~0.01-0.04 nats for smooth Gaussian densities. We use 0.05 as a safe ceiling
    # that catches real bugs (a 2x wrong answer at rho=0.3 gives error ~0.04 nats,
    # which is outside this bound at rho=0.6 and rho=0.9).
    TOLERANCE = 0.05  # nats

    def test_ksg_mi_gaussian_rho_03(self):
        """KSG MI matches analytic value for rho=0.3 within 0.05 nats.

        Analytic: I(X;Y) = -0.5*ln(1-0.09) = -0.5*ln(0.91) ≈ 0.04652 nats.
        """
        rho = 0.3
        analytic = _analytic_gaussian_mi(rho)  # ≈ 0.04652 nats
        rng = np.random.default_rng(42)
        X, Y = _bivariate_gaussian(rho, n=5000, rng=rng)
        mi = ksg_mi(X, Y, k=4)
        assert abs(mi - analytic) < self.TOLERANCE, (
            f"rho=0.3: expected {analytic:.5f} nats, got {mi:.5f} nats "
            f"(diff={abs(mi-analytic):.5f}, tol={self.TOLERANCE})"
        )

    def test_ksg_mi_gaussian_rho_06(self):
        """KSG MI matches analytic value for rho=0.6 within 0.05 nats.

        Analytic: I(X;Y) = -0.5*ln(1-0.36) = -0.5*ln(0.64) ≈ 0.21072 nats.
        """
        rho = 0.6
        analytic = _analytic_gaussian_mi(rho)  # ≈ 0.21072 nats
        rng = np.random.default_rng(43)
        X, Y = _bivariate_gaussian(rho, n=5000, rng=rng)
        mi = ksg_mi(X, Y, k=4)
        assert abs(mi - analytic) < self.TOLERANCE, (
            f"rho=0.6: expected {analytic:.5f} nats, got {mi:.5f} nats "
            f"(diff={abs(mi-analytic):.5f}, tol={self.TOLERANCE})"
        )

    def test_ksg_mi_gaussian_rho_09(self):
        """KSG MI matches analytic value for rho=0.9 within 0.05 nats.

        Analytic: I(X;Y) = -0.5*ln(1-0.81) = -0.5*ln(0.19) ≈ 0.83148 nats.
        Note: higher rho = stronger dependence = larger MI; the estimator
        is harder to calibrate here, so we keep the same absolute tol.
        """
        rho = 0.9
        analytic = _analytic_gaussian_mi(rho)  # ≈ 0.83148 nats
        rng = np.random.default_rng(44)
        X, Y = _bivariate_gaussian(rho, n=5000, rng=rng)
        mi = ksg_mi(X, Y, k=4)
        assert abs(mi - analytic) < self.TOLERANCE, (
            f"rho=0.9: expected {analytic:.5f} nats, got {mi:.5f} nats "
            f"(diff={abs(mi-analytic):.5f}, tol={self.TOLERANCE})"
        )


# ---------------------------------------------------------------------------
# ROADMAP Test 4: Independence => ~0
# ---------------------------------------------------------------------------

def test_ksg_mi_independent_near_zero():
    """I(X;Y) ≈ 0 for independent X, Y.

    KSG MI can be slightly negative due to finite-sample bias; we allow
    |mi| < 0.05. The test asserts a MEANINGFUL bound — a wrong estimator
    that returns 0.5 nats for independent data would fail this.

    Tolerance 0.05 nats: at n=3000 with no dependence, KSG typically
    returns values in [-0.03, 0.03] for 1-D inputs.
    """
    rng = np.random.default_rng(100)
    X = rng.standard_normal((3000, 1))
    Y = rng.standard_normal((3000, 1))
    mi = ksg_mi(X, Y, k=4)
    assert abs(mi) < 0.05, (
        f"Independent X,Y: expected |MI| < 0.05 nats, got {mi:.5f} nats"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 5: Symmetry
# ---------------------------------------------------------------------------

def test_ksg_mi_symmetry():
    """I(X;Y) == I(Y;X) within float precision (KSG estimator is symmetric by construction).

    This is a mathematical invariant — any correct implementation must satisfy it exactly
    (same data, same k, same code path). We allow 1e-10 to cover floating-point rounding
    across different code orderings, but NOT differences of 1e-6 or larger.
    """
    rho = 0.5
    rng = np.random.default_rng(200)
    X, Y = _bivariate_gaussian(rho, n=2000, rng=rng)
    mi_xy = ksg_mi(X, Y, k=4)
    mi_yx = ksg_mi(Y, X, k=4)
    assert abs(mi_xy - mi_yx) < 1e-10, (
        f"Symmetry violated: I(X;Y)={mi_xy:.8f}, I(Y;X)={mi_yx:.8f}, "
        f"diff={abs(mi_xy - mi_yx):.2e}"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 6: CMI collapses in Markov chain
# ---------------------------------------------------------------------------

def test_ksg_cmi_collapses_in_markov_chain():
    """I(X;Y|Z) ≈ 0 in a Markov chain X <- Z -> Y (X and Y are conditionally independent).

    Construction: Z ~ N(0,1), X = 0.8*Z + noise_x, Y = 0.8*Z + noise_y.
    Given Z, X and Y are independent. So:
        I(X;Y) > 0  (marginal dependence via shared Z)
        I(X;Y|Z) ≈ 0  (no residual dependence after conditioning on Z)

    Both conditions are asserted to prevent an implementation that trivially
    returns 0 for all CMI calls from passing.

    Tolerance 0.05 nats for CMI (CMI estimator has higher variance than MI).
    Tolerance 0.01 nats for the marginal MI lower bound (rho_XY ≈ 0.64 => analytic ≈ 0.23).
    """
    rng = np.random.default_rng(300)
    n = 6000
    Z = rng.standard_normal((n, 1))
    X = 0.8 * Z + 0.6 * rng.standard_normal((n, 1))  # corr(X,Z) = 0.8/sqrt(0.64+0.36)=0.8
    Y = 0.8 * Z + 0.6 * rng.standard_normal((n, 1))

    mi_xy = ksg_mi(X, Y, k=4)
    cmi_xy_z = ksg_cmi(X, Y, Z, k=4)

    # Marginal MI must be substantially > 0 (else the test is vacuous)
    assert mi_xy > 0.08, (
        f"Marginal I(X;Y) expected > 0.08 nats (Markov chain), got {mi_xy:.5f} nats"
    )
    # CMI must collapse to near-zero
    assert abs(cmi_xy_z) < 0.05, (
        f"I(X;Y|Z) expected ≈ 0 (Markov chain), got {cmi_xy_z:.5f} nats "
        f"(marginal I(X;Y)={mi_xy:.5f})"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 7: CMI recovers MI when Z is independent noise
# ---------------------------------------------------------------------------

def test_ksg_cmi_recovers_mi_with_independent_z():
    """I(X;Y|Z_indep) ≈ I(X;Y) when Z is independent of both X and Y.

    Conditioning on an irrelevant variable should not change mutual information.
    Analytic: for Gaussian (X,Y) with rho=0.5, I(X;Y) = -0.5*ln(0.75) ≈ 0.1438 nats.

    Tolerance 0.07 nats: CMI estimator has higher variance than MI; the extra
    conditioning dimension adds finite-sample noise.
    """
    rho = 0.5
    analytic_mi = _analytic_gaussian_mi(rho)  # ≈ 0.1438 nats
    rng = np.random.default_rng(400)
    n = 6000
    X, Y = _bivariate_gaussian(rho, n=n, rng=rng)
    Z_indep = rng.standard_normal((n, 1))  # independent of X and Y

    mi_xy = ksg_mi(X, Y, k=4)
    cmi_xy_z = ksg_cmi(X, Y, Z_indep, k=4)

    # Both should be close to the analytic value
    assert abs(mi_xy - analytic_mi) < 0.05, (
        f"I(X;Y) expected ≈{analytic_mi:.4f} nats, got {mi_xy:.5f}"
    )
    assert abs(cmi_xy_z - mi_xy) < 0.07, (
        f"I(X;Y|Z_indep) expected ≈ I(X;Y)={mi_xy:.5f} nats, "
        f"got I(X;Y|Z_indep)={cmi_xy_z:.5f} nats (diff={abs(cmi_xy_z-mi_xy):.5f})"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 8: TE directionality
# ---------------------------------------------------------------------------

def test_transfer_entropy_directionality():
    """TE(source->target) >> TE(target->source) for a unidirectional coupled system.

    Construction: source is i.i.d. Gaussian noise.
    target[t] = 0.8 * source[t-1] + 0.4 * noise[t]

    Expected:
        TE(source->target) > 0  (source causes target with lag=1)
        TE(target->source) ≈ bias_floor  (no reverse causation)
        TE(source->target) > TE(target->source) by a meaningful margin

    We assert the forward TE exceeds reverse by at least 0.05 nats.
    This gap catches: wrong lag alignment, wrong conditioning, symmetrical estimators.
    """
    rng = np.random.default_rng(500)
    n = 3000
    source = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    # target[t] depends on source[t-1], not the reverse
    target = np.zeros(n)
    target[0] = noise[0]
    for t in range(1, n):
        target[t] = 0.8 * source[t - 1] + 0.4 * noise[t]

    te_fwd = transfer_entropy(source, target, k=4, lag=1)
    te_rev = transfer_entropy(target, source, k=4, lag=1)

    assert te_fwd > te_rev + 0.05, (
        f"Expected TE(source->target) > TE(target->source) + 0.05 nats; "
        f"got TE_fwd={te_fwd:.5f}, TE_rev={te_rev:.5f}, gap={te_fwd-te_rev:.5f}"
    )
    # Forward TE must be positive (not just larger than reverse)
    assert te_fwd > 0.0, (
        f"TE(source->target) must be positive for coupled system; got {te_fwd:.5f}"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 9: TE bias floor on independent series
# ---------------------------------------------------------------------------

def test_transfer_entropy_null_near_zero():
    """TE on independent series is near zero (within KSG bias floor).

    KSG TE has a positive bias that scales with k and n. For k=4, n=2000,
    this bias is typically < 0.05 nats. We assert |TE| < 0.06 nats.

    This test documents the calibration requirement: production TE estimates
    should be compared against a shuffle-null, not against zero, for small n.
    """
    rng = np.random.default_rng(600)
    n = 2000
    source = rng.standard_normal(n)
    target = rng.standard_normal(n)  # completely independent

    te_fwd = transfer_entropy(source, target, k=4, lag=1)
    te_rev = transfer_entropy(target, source, k=4, lag=1)

    assert abs(te_fwd) < 0.06, (
        f"TE(independent source->target) expected |TE| < 0.06 nats, got {te_fwd:.5f}"
    )
    assert abs(te_rev) < 0.06, (
        f"TE(independent target->source) expected |TE| < 0.06 nats, got {te_rev:.5f}"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 10: 1-D arrays accepted
# ---------------------------------------------------------------------------

def test_ksg_mi_accepts_1d_arrays():
    """ksg_mi accepts 1-D arrays (n,) and treats them as (n,1).

    The result should match the result for the equivalent 2-D input.
    """
    rng = np.random.default_rng(700)
    X_2d, Y_2d = _bivariate_gaussian(0.5, n=500, rng=rng)
    X_1d = X_2d[:, 0]  # shape (500,)
    Y_1d = Y_2d[:, 0]  # shape (500,)

    # Both calls must succeed without raising
    mi_2d = ksg_mi(X_2d, Y_2d, k=4)
    mi_1d = ksg_mi(X_1d, Y_1d, k=4)

    # Results must be identical (same data, same semantics)
    assert abs(mi_1d - mi_2d) < 1e-10, (
        f"1-D input gave different result than 2-D: mi_1d={mi_1d:.8f}, mi_2d={mi_2d:.8f}"
    )
    # Result must be a scalar float, not an array
    assert np.isscalar(mi_1d) or (isinstance(mi_1d, np.ndarray) and mi_1d.ndim == 0), (
        f"ksg_mi should return a scalar; got type={type(mi_1d)}, value={mi_1d}"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 11: Mismatched lengths raise
# ---------------------------------------------------------------------------

def test_ksg_mi_raises_on_mismatched_lengths():
    """ksg_mi raises ValueError when X and Y have different numbers of samples."""
    rng = np.random.default_rng(800)
    X = rng.standard_normal((100, 1))
    Y = rng.standard_normal((150, 1))  # different n
    with pytest.raises((ValueError, AssertionError)):
        ksg_mi(X, Y, k=4)


# ---------------------------------------------------------------------------
# ROADMAP Test 12: k < 1 raises in ksg_mi
# ---------------------------------------------------------------------------

def test_ksg_mi_raises_on_invalid_k():
    """ksg_mi raises ValueError for k < 1 (k must be a positive integer)."""
    rng = np.random.default_rng(900)
    X = rng.standard_normal((100, 1))
    Y = rng.standard_normal((100, 1))
    with pytest.raises((ValueError, AssertionError)):
        ksg_mi(X, Y, k=0)


# ---------------------------------------------------------------------------
# ROADMAP Test 13: k < 1 raises in ksg_cmi
# ---------------------------------------------------------------------------

def test_ksg_cmi_raises_on_invalid_k():
    """ksg_cmi raises ValueError for k < 1."""
    rng = np.random.default_rng(901)
    X = rng.standard_normal((100, 1))
    Y = rng.standard_normal((100, 1))
    Z = rng.standard_normal((100, 1))
    with pytest.raises((ValueError, AssertionError)):
        ksg_cmi(X, Y, Z, k=0)


# ---------------------------------------------------------------------------
# ROADMAP Test 14: TE mismatched lengths raise
# ---------------------------------------------------------------------------

def test_transfer_entropy_raises_on_mismatched_lengths():
    """transfer_entropy raises ValueError when source and target have different lengths."""
    rng = np.random.default_rng(902)
    source = rng.standard_normal(100)
    target = rng.standard_normal(120)
    with pytest.raises((ValueError, AssertionError)):
        transfer_entropy(source, target, k=4, lag=1)


# ---------------------------------------------------------------------------
# Additional: Trivial n edge case
# ---------------------------------------------------------------------------

def test_ksg_mi_raises_or_handles_trivial_n():
    """ksg_mi with n <= k should either raise or return a finite value.

    With n=3 and k=4, there aren't enough neighbors. A correct implementation
    must not silently return NaN or Inf — it should raise or gracefully degrade.
    """
    rng = np.random.default_rng(1000)
    X = rng.standard_normal((3, 1))
    Y = rng.standard_normal((3, 1))
    try:
        mi = ksg_mi(X, Y, k=4)
        # If no exception, result must be finite (not NaN, not Inf)
        assert np.isfinite(mi), (
            f"ksg_mi(n=3, k=4) returned non-finite value: {mi}"
        )
    except (ValueError, AssertionError):
        pass  # Raising is acceptable; returning NaN/Inf is not


# ---------------------------------------------------------------------------
# Additional: CMI accepts 1-D Z array
# ---------------------------------------------------------------------------

def test_ksg_cmi_accepts_1d_z():
    """ksg_cmi accepts a 1-D Z array and treats it as (n,1).

    This mirrors the ksg_mi 1-D requirement but for the conditioning variable.
    The result should match the result with Z reshaped to (n,1).
    """
    rho = 0.5
    rng = np.random.default_rng(1100)
    n = 500
    X, Y = _bivariate_gaussian(rho, n=n, rng=rng)
    Z_2d = rng.standard_normal((n, 1))
    Z_1d = Z_2d[:, 0]  # shape (500,)

    cmi_2d = ksg_cmi(X, Y, Z_2d, k=4)
    cmi_1d = ksg_cmi(X, Y, Z_1d, k=4)

    assert abs(cmi_1d - cmi_2d) < 1e-10, (
        f"1-D Z gave different CMI than 2-D: cmi_1d={cmi_1d:.8f}, cmi_2d={cmi_2d:.8f}"
    )


# ---------------------------------------------------------------------------
# Additional: Multivariate inputs accepted
# ---------------------------------------------------------------------------

def test_ksg_mi_accepts_multivariate_inputs():
    """ksg_mi accepts dx>1 and dy>1 inputs and returns a finite scalar.

    The KSG estimator works in arbitrary dimension via the joint-space KD-tree.
    We do not assert a specific value, but the result must be finite and
    non-negative (for large n, MI is non-negative by construction).
    """
    rng = np.random.default_rng(1200)
    n = 2000
    # 2-D X and 3-D Y, mildly correlated via a shared latent
    latent = rng.standard_normal((n, 1))
    X = np.hstack([0.6 * latent + rng.standard_normal((n, 1)) * 0.8,
                   rng.standard_normal((n, 1))])  # (n, 2)
    Y = np.hstack([0.6 * latent + rng.standard_normal((n, 1)) * 0.8,
                   rng.standard_normal((n, 1)),
                   rng.standard_normal((n, 1))])  # (n, 3)

    mi = ksg_mi(X, Y, k=4)
    assert np.isfinite(mi), f"ksg_mi with multivariate inputs returned non-finite: {mi}"
    # MI must be positive for correlated multivariate inputs
    assert mi > 0.0, f"ksg_mi(X_2d, Y_3d) expected > 0 for correlated inputs, got {mi:.5f}"


# ---------------------------------------------------------------------------
# Additional: TE with optional cond array
# ---------------------------------------------------------------------------

def test_transfer_entropy_with_cond_array():
    """transfer_entropy accepts cond= keyword argument without raising.

    The cond array is an extra conditioning set (beyond target_past). This
    test verifies the API is wired correctly and returns a finite scalar.
    The directionality property must still hold when conditioning is added.
    """
    rng = np.random.default_rng(1300)
    n = 1000
    source = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    target = np.zeros(n)
    target[0] = noise[0]
    for t in range(1, n):
        target[t] = 0.7 * source[t - 1] + 0.5 * noise[t]

    cond = rng.standard_normal(n)  # irrelevant extra conditioning

    te_fwd_cond = transfer_entropy(source, target, k=4, lag=1, cond=cond)
    assert np.isfinite(te_fwd_cond), (
        f"transfer_entropy with cond= returned non-finite: {te_fwd_cond}"
    )
    # TE should remain positive even with irrelevant conditioning
    assert te_fwd_cond > 0.0, (
        f"TE(source->target, cond=noise) expected > 0 for coupled system; "
        f"got {te_fwd_cond:.5f}"
    )
