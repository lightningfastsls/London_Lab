"""Tests for train_shape_vae_v3_deriv — written by test-architect BEFORE implementation.

Context: This is Pathway A derivative-loss contour-VAE. The derivative penalty is computed
on a kHz-aware soft-argmax ridge (not raw pixel gradients), making it pitch-invariant.
Two prior attempts failed:
  - M10: pixel-gradient "derivative" is NOT pitch-invariant (identical-shape ridge shifted
    in freq gives different derivative because lit pixels move). These tests GUARD that bug.
  - M8: derivative on pre-registered ridge had nothing to do (pitch/position already removed).

ROADMAP test plan coverage:
  1. soft_argmax_frequency shape (B,1,H,W) -> (B,W)       -> test_soft_argmax_shape_4d
  2. soft_argmax_frequency shape (B,H,W) -> (B,W)          -> test_soft_argmax_shape_3d
  3. soft_argmax_frequency localization (one-hot column)   -> test_soft_argmax_localizes_to_bright_row
  4. soft_argmax_frequency differentiable                  -> test_soft_argmax_is_differentiable
  5. soft_argmax_frequency monotone temperature            -> test_soft_argmax_temperature_monotone
  6. ridge_first_derivative shape                          -> test_ridge_first_derivative_shape
  7. ridge_first_derivative linear ramp is constant        -> test_ridge_first_derivative_linear_ramp
  8. ridge_first_derivative flat track is zero             -> test_ridge_first_derivative_flat_track
  9. ridge_second_derivative shape                         -> test_ridge_second_derivative_shape
 10. ridge_second_derivative linear ramp is ~0             -> test_ridge_second_derivative_linear_ramp
 11. derivative_loss keys and scalar shape                 -> test_derivative_loss_keys_and_scalar_shape
 12. derivative_loss zero when matched                     -> test_derivative_loss_zero_when_matched
 13. derivative_loss masking: masked columns ignored       -> test_derivative_loss_masking_excludes_invalid
 14. derivative_loss lambda_c=0 -> curv==0                 -> test_derivative_loss_zero_curv_when_lambda_c_zero
 15. PITCH-SHIFT INVARIANCE (THE HEADLINE TEST)            -> test_pitch_shift_invariance_of_dFdt
 16. total_shape_vae_loss keys and scalar shape            -> test_total_loss_keys_and_scalar_shape
 17. total_shape_vae_loss reduces to baseline ELBO         -> test_total_loss_reduces_to_baseline_elbo
 18. ShapeVAEDerivConfig defaults                          -> test_config_defaults
 19. ShapeVAEDerivConfig invalid recon raises              -> test_config_invalid_recon_raises
 20. ShapeVAEDerivConfig negative lambda_d raises          -> test_config_negative_lambda_d_raises
 21. load_ridge_cache valid npz                            -> test_load_ridge_cache_valid
 22. load_ridge_cache mismatched N raises ValueError       -> test_load_ridge_cache_mismatch_raises

Additional coverage (recurring gap patterns):
  - Empty / single-row frequency edge case                -> test_soft_argmax_single_row_freq
  - Single batch item (B=1)                               -> test_ridge_first_derivative_single_batch
  - derivative_loss non-negative deriv                    -> test_derivative_loss_nonnegative
  - load_ridge_cache optional d2_true present             -> test_load_ridge_cache_with_optional_d2
  - total_loss formula: correct weighting of each term    -> test_total_loss_formula_weighted_sum

Total: 27 tests (22 from ROADMAP, 5 additional)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path bootstrap — make both scripts/ and scripts/experiments/ importable
# once the implementation modules exist.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_EXPERIMENTS_ROOT = _SCRIPTS_ROOT / "experiments"
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SCRIPTS_ROOT, _EXPERIMENTS_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# The module under test — will cause ImportError until implemented.
# All tests below will be collected but fail at import time (expected red state).
# ---------------------------------------------------------------------------
import train_shape_vae_v3_deriv as _mod  # noqa: E402  — Will fail until implemented

soft_argmax_frequency = _mod.soft_argmax_frequency
ridge_first_derivative = _mod.ridge_first_derivative
ridge_second_derivative = _mod.ridge_second_derivative
derivative_loss = _mod.derivative_loss
total_shape_vae_loss = _mod.total_shape_vae_loss
ShapeVAEDerivConfig = _mod.ShapeVAEDerivConfig
load_ridge_cache = _mod.load_ridge_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_near_onehot_img(B: int, H: int, W: int, bright_rows: list[int]) -> torch.Tensor:
    """Create a (B, 1, H, W) image where column t has value 1.0 at row bright_rows[t],
    1e-4 everywhere else. All B items share the same ridge pattern."""
    img = torch.full((B, 1, H, W), 1e-4)
    for t, r in enumerate(bright_rows):
        img[:, 0, r, t] = 1.0
    return img


def _make_linear_ramp(B: int, W: int, a: float = 10.0, b: float = 2.0) -> torch.Tensor:
    """Return (B, W) tensor F[t] = a + b*t."""
    t = torch.arange(W, dtype=torch.float32)
    F_ = a + b * t  # (W,)
    return F_.unsqueeze(0).expand(B, -1)  # (B, W)


# ============================================================
# 1-5: soft_argmax_frequency
# ============================================================

class TestSoftArgmaxFrequency:

    def test_soft_argmax_shape_4d(self):
        """Spec: (B,1,H,W) img + (H,) freq_khz -> (B,W) output."""
        B, H, W = 2, 16, 8
        img = torch.rand(B, 1, H, W)
        freq_khz = torch.linspace(20.0, 120.0, H)
        out = soft_argmax_frequency(img, freq_khz, tau=0.05)
        assert out.shape == (B, W), f"Expected ({B},{W}), got {out.shape}"

    def test_soft_argmax_shape_3d(self):
        """Spec: also accepts (B,H,W) without channel dim."""
        B, H, W = 2, 16, 8
        img = torch.rand(B, H, W)
        freq_khz = torch.linspace(20.0, 120.0, H)
        out = soft_argmax_frequency(img, freq_khz, tau=0.05)
        assert out.shape == (B, W), f"Expected ({B},{W}), got {out.shape}"

    def test_soft_argmax_localizes_to_bright_row(self):
        """Spec: near-one-hot column at row k -> output kHz ≈ freq_khz[k] within 1e-2.

        At tau=0.01 the softmax is very peaked, so the expected-frequency
        weighted sum collapses to the bright row's kHz value.
        """
        H, W = 16, 8
        freq_khz = torch.linspace(20.0, 120.0, H)
        bright_row = 7  # arbitrary interior row
        expected_kHz = float(freq_khz[bright_row])

        # Build a single-sample image: column 0 is near-one-hot at row 7,
        # other columns are uniform (we only check column 0).
        img = torch.full((1, 1, H, W), 1e-6)
        img[0, 0, bright_row, 0] = 1.0

        out = soft_argmax_frequency(img, freq_khz, tau=0.01)
        actual_kHz = float(out[0, 0])
        assert abs(actual_kHz - expected_kHz) < 1e-2, (
            f"Expected freq ≈ {expected_kHz:.4f} kHz for bright_row={bright_row}, "
            f"got {actual_kHz:.4f} kHz"
        )

    def test_soft_argmax_is_differentiable(self):
        """Spec: gradients flow through soft_argmax_frequency w.r.t. img."""
        H, W = 16, 8
        img = torch.rand(2, 1, H, W, requires_grad=True)
        freq_khz = torch.linspace(20.0, 120.0, H)
        out = soft_argmax_frequency(img, freq_khz, tau=0.05)
        out.sum().backward()
        assert img.grad is not None, "img.grad should be populated after backward()"
        assert torch.isfinite(img.grad).all(), "img.grad contains non-finite values"

    def test_soft_argmax_temperature_monotone(self):
        """Spec: smaller tau -> result closer to hard-argmax freq than large tau.

        Build a near-one-hot column with a secondary weaker peak at a
        different row. Small tau collapses to the dominant peak; large tau
        smears toward the secondary peak. The small-tau result must be closer
        to freq_khz[bright_row] than the large-tau result.
        """
        H, W = 32, 4
        bright_row = 10
        weak_row = 25
        freq_khz = torch.linspace(20.0, 150.0, H)
        expected_kHz = float(freq_khz[bright_row])

        img = torch.full((1, 1, H, W), 1e-6)
        img[0, 0, bright_row, 0] = 1.0
        img[0, 0, weak_row, 0] = 0.3  # secondary peak

        out_small = soft_argmax_frequency(img, freq_khz, tau=0.01)
        out_large = soft_argmax_frequency(img, freq_khz, tau=5.0)

        err_small = abs(float(out_small[0, 0]) - expected_kHz)
        err_large = abs(float(out_large[0, 0]) - expected_kHz)
        assert err_small < err_large, (
            f"Small-tau error ({err_small:.4f}) should be less than large-tau error ({err_large:.4f})"
        )

    def test_soft_argmax_single_row_freq(self):
        """Edge: H=1 (degenerate case) -> output is all freq_khz[0], shape still (B,W)."""
        H, W = 1, 6
        img = torch.rand(2, 1, H, W)
        freq_khz = torch.tensor([70.0])
        out = soft_argmax_frequency(img, freq_khz, tau=0.05)
        assert out.shape == (2, W)
        # With only one row, softmax is 1.0 everywhere, so output == freq_khz[0]
        assert torch.allclose(out, torch.full_like(out, 70.0), atol=1e-4), (
            "Single-row soft_argmax should equal freq_khz[0] for all columns"
        )


# ============================================================
# 6-10: ridge_first_derivative and ridge_second_derivative
# ============================================================

class TestRidgeFirstDerivative:

    def test_ridge_first_derivative_shape(self):
        """Spec: (B, W) -> (B, W-1)."""
        B, W = 3, 10
        F_ = torch.rand(B, W)
        dF = ridge_first_derivative(F_)
        assert dF.shape == (B, W - 1), f"Expected ({B},{W-1}), got {dF.shape}"

    def test_ridge_first_derivative_linear_ramp(self):
        """Spec: F[t] = a + b*t -> dF/dt = b (constant) within 1e-5."""
        B, W = 3, 10
        a, b = 10.0, 2.5
        F_ = _make_linear_ramp(B, W, a=a, b=b)
        dF = ridge_first_derivative(F_)
        expected = torch.full((B, W - 1), b)
        assert torch.allclose(dF, expected, atol=1e-5), (
            f"Linear ramp dF/dt should be constant {b}; got range "
            f"[{dF.min():.6f}, {dF.max():.6f}]"
        )

    def test_ridge_first_derivative_flat_track(self):
        """Spec: constant F -> dF/dt == 0 everywhere."""
        B, W = 2, 8
        F_ = torch.full((B, W), 70.0)
        dF = ridge_first_derivative(F_)
        assert torch.allclose(dF, torch.zeros(B, W - 1), atol=1e-6), (
            "Flat track should have zero derivative"
        )

    def test_ridge_first_derivative_single_batch(self):
        """Edge: B=1 — shape and values still correct."""
        F_ = _make_linear_ramp(1, 5, a=0.0, b=3.0)
        dF = ridge_first_derivative(F_)
        assert dF.shape == (1, 4)
        assert torch.allclose(dF, torch.full((1, 4), 3.0), atol=1e-5)


class TestRidgeSecondDerivative:

    def test_ridge_second_derivative_shape(self):
        """Spec: (B, W) -> (B, W-2)."""
        B, W = 4, 12
        F_ = torch.rand(B, W)
        d2F = ridge_second_derivative(F_)
        assert d2F.shape == (B, W - 2), f"Expected ({B},{W-2}), got {d2F.shape}"

    def test_ridge_second_derivative_linear_ramp(self):
        """Spec: linear F -> second derivative ≈ 0 (up to floating-point noise)."""
        B, W = 2, 10
        F_ = _make_linear_ramp(B, W, a=5.0, b=3.0)
        d2F = ridge_second_derivative(F_)
        assert torch.allclose(d2F, torch.zeros(B, W - 2), atol=1e-4), (
            f"Second derivative of linear ramp should be ~0; max abs = {d2F.abs().max():.2e}"
        )


# ============================================================
# 11-15 + additional: derivative_loss
# ============================================================

class TestDerivativeLoss:

    def _build_matched_loss_inputs(self):
        """Build (img, freq_khz, dFdt_true) where the soft-argmax ridge of img
        exactly matches a known F(t), so the loss should be ~0."""
        B, H, W = 2, 16, 12
        freq_khz = torch.linspace(20.0, 120.0, H)
        # Place bright row at linearly varying positions to define a ridge
        bright_rows = [4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3]  # 12 columns
        assert len(bright_rows) == W
        img = _make_near_onehot_img(B, H, W, bright_rows)
        # The soft-argmax ridge at tau=0.01 should closely track freq_khz[bright_rows]
        F_true = torch.tensor(
            [float(freq_khz[r]) for r in bright_rows], dtype=torch.float32
        ).unsqueeze(0).expand(B, -1)  # (B, W)
        dFdt_true = F_true[:, 1:] - F_true[:, :-1]  # (B, W-1)
        valid_mask = torch.ones(B, W - 1, dtype=torch.bool)
        return img, freq_khz, dFdt_true, valid_mask

    def test_derivative_loss_keys_and_scalar_shape(self):
        """Spec: returns dict with keys 'deriv', 'curv', 'total'; all are 0-dim tensors."""
        img, freq_khz, dFdt_true, valid_mask = self._build_matched_loss_inputs()
        result = derivative_loss(img, freq_khz, dFdt_true, valid_mask=valid_mask)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        for key in ("deriv", "curv", "total"):
            assert key in result, f"Missing key '{key}' in derivative_loss output"
            assert result[key].ndim == 0, (
                f"Key '{key}' should be a 0-dim (scalar) tensor; got shape {result[key].shape}"
            )

    def test_derivative_loss_nonnegative(self):
        """Spec: 'deriv' component is a mean-squared quantity, so >= 0."""
        img, freq_khz, dFdt_true, valid_mask = self._build_matched_loss_inputs()
        result = derivative_loss(img, freq_khz, dFdt_true, valid_mask=valid_mask)
        assert float(result["deriv"]) >= 0.0, (
            f"'deriv' must be >= 0 (it's an MSE), got {result['deriv']}"
        )

    def test_derivative_loss_zero_when_matched(self):
        """Spec: when img's soft-argmax ridge matches dFdt_true, deriv ≈ 0 (within 1e-2).

        We use tau=0.01 (sharply peaked) so the soft-argmax closely tracks the
        bright row's freq, and we supply dFdt_true computed from those exact
        expected frequencies.
        """
        img, freq_khz, dFdt_true, valid_mask = self._build_matched_loss_inputs()
        result = derivative_loss(
            img, freq_khz, dFdt_true,
            valid_mask=valid_mask, tau=0.01, lambda_d=1.0, lambda_c=0.0
        )
        assert float(result["deriv"]) < 1e-2, (
            f"'deriv' should be ~0 when img ridge matches dFdt_true; got {result['deriv']:.4f}"
        )

    def test_derivative_loss_masking_excludes_invalid(self):
        """Spec: changing dFdt_true ONLY at masked (invalid) columns does NOT change 'deriv'.

        valid_mask=0 means that column should be ignored. Any dFdt_true value at
        those columns is irrelevant — the loss must be identical.
        """
        img, freq_khz, dFdt_true_base, _ = self._build_matched_loss_inputs()
        B, W = dFdt_true_base.shape

        # Mask: first 3 columns are invalid, rest valid
        valid_mask = torch.ones(B, W, dtype=torch.bool)
        valid_mask[:, :3] = False  # mask out first 3 derivative columns

        # Two dFdt_true that differ ONLY at the masked positions
        dFdt_true_a = dFdt_true_base.clone()
        dFdt_true_b = dFdt_true_base.clone()
        dFdt_true_b[:, :3] = 999.0  # wildly different, but masked

        result_a = derivative_loss(img, freq_khz, dFdt_true_a, valid_mask=valid_mask, tau=0.01)
        result_b = derivative_loss(img, freq_khz, dFdt_true_b, valid_mask=valid_mask, tau=0.01)

        assert abs(float(result_a["deriv"]) - float(result_b["deriv"])) < 1e-6, (
            f"Masking should render invalid-column changes irrelevant; "
            f"deriv_a={result_a['deriv']:.6f}, deriv_b={result_b['deriv']:.6f}"
        )

    def test_derivative_loss_zero_curv_when_lambda_c_zero(self):
        """Spec: lambda_c=0 -> 'curv' == 0.0."""
        img, freq_khz, dFdt_true, valid_mask = self._build_matched_loss_inputs()
        result = derivative_loss(
            img, freq_khz, dFdt_true, valid_mask=valid_mask, lambda_c=0.0
        )
        assert float(result["curv"]) == 0.0, (
            f"lambda_c=0 should produce curv=0.0; got {result['curv']}"
        )

    def test_derivative_loss_total_formula(self):
        """Spec: total = lambda_d * deriv + lambda_c * curv."""
        B, H, W = 2, 16, 12
        freq_khz = torch.linspace(20.0, 120.0, H)
        img = torch.rand(B, 1, H, W).clamp(0.0, 1.0)
        dFdt_true = torch.zeros(B, W - 1)
        valid_mask = torch.ones(B, W - 1, dtype=torch.bool)
        d2_true = torch.zeros(B, W - 2)
        valid_mask2 = torch.ones(B, W - 2, dtype=torch.bool)

        ld, lc = 2.5, 0.7
        result = derivative_loss(
            img, freq_khz, dFdt_true,
            valid_mask=valid_mask,
            tau=0.05,
            lambda_d=ld,
            lambda_c=lc,
            d2_true=d2_true,
            valid_mask2=valid_mask2,
        )
        expected_total = ld * float(result["deriv"]) + lc * float(result["curv"])
        # float32 sum at this magnitude (~1e3) has 1 ULP ~1e-4, so an absolute
        # 1e-5 is tighter than representable. The identity is unchanged — only
        # the tolerance is made float32-aware (relative). Approved 2026-05-27.
        assert math.isclose(float(result["total"]), expected_total, rel_tol=1e-5), (
            f"total != lambda_d*deriv + lambda_c*curv; "
            f"total={result['total']:.6f}, expected={expected_total:.6f}"
        )


# ============================================================
# THE HEADLINE TEST: pitch-shift invariance
# ============================================================

class TestPitchShiftInvariance:

    def test_pitch_shift_invariance_of_dFdt(self):
        """THE CRITICAL GUARD AGAINST M10's PIXEL-GRADIENT MISTAKE.

        M10 used raw pixel gradients as 'the derivative'. Shifting a ridge up/down
        in frequency moves all the lit pixels, producing a large derivative term for
        a geometrically identical shape. soft_argmax_frequency + ridge_first_derivative
        must be INVARIANT to a pure vertical (pitch) shift.

        Construction:
          - chevron: rows go 3,5,7,9,11,9,7,5,3,5,7,9 (12 columns), inside H=20 rows
          - unshifted ridge -> F1 -> dF1
          - shifted ridge (rows += delta=4) -> F2 -> dF2
          - ASSERT dF2 ≈ dF1 within tol 1e-2

        The key property: F2 = F1 + const (pitch offset), so F2[t]-F2[t-1] = F1[t]-F1[t-1].
        The derivative is invariant to additive constant — a property soft_argmax provides
        but raw pixel gradients DO NOT.
        """
        H, W = 20, 12
        freq_khz = torch.linspace(20.0, 110.0, H)
        delta = 4  # shift all rows DOWN by 4 (higher frequency) — must stay in bounds

        # Chevron pattern — rows go up then down, well-contained within H=20
        bright_rows_base = [3, 5, 7, 9, 11, 9, 7, 5, 3, 5, 7, 9]
        assert len(bright_rows_base) == W
        assert max(bright_rows_base) + delta < H, "Shift goes out of bounds"
        bright_rows_shifted = [r + delta for r in bright_rows_base]

        B = 1
        tau = 0.01  # very peaked so soft-argmax closely tracks bright row

        img_base = _make_near_onehot_img(B, H, W, bright_rows_base)
        img_shifted = _make_near_onehot_img(B, H, W, bright_rows_shifted)

        F1 = soft_argmax_frequency(img_base, freq_khz, tau=tau)     # (1, W)
        F2 = soft_argmax_frequency(img_shifted, freq_khz, tau=tau)  # (1, W)

        dF1 = ridge_first_derivative(F1)  # (1, W-1)
        dF2 = ridge_first_derivative(F2)  # (1, W-1)

        # The SLOPE must match: dF2 should equal dF1 within numerical tolerance.
        # (The absolute values F2 = F1 + const, but differences cancel the constant.)
        max_err = float((dF2 - dF1).abs().max())
        assert max_err < 1e-2, (
            f"dF/dt should be pitch-invariant: max |dF2-dF1| = {max_err:.4e} "
            f"(M10 bug: raw pixel gradients would give large error here)"
        )

        # Also verify the ABSOLUTE ridge DID shift (so we know the shift was real)
        mean_F1 = float(F1.mean())
        mean_F2 = float(F2.mean())
        freq_step = float((freq_khz[1] - freq_khz[0]))
        expected_shift_kHz = delta * freq_step
        actual_shift_kHz = mean_F2 - mean_F1
        assert abs(actual_shift_kHz - expected_shift_kHz) < 1.0, (
            f"Ridge should be shifted by ~{expected_shift_kHz:.2f} kHz; "
            f"got {actual_shift_kHz:.2f} kHz"
        )


# ============================================================
# 16-17: total_shape_vae_loss
# ============================================================

class TestTotalShapeVaeLoss:

    def _make_tiny_batch(self, B: int = 2, H: int = 16, W: int = 8):
        """Build a tiny batch of (x_recon, x, mu, logvar) in [0,1]."""
        torch.manual_seed(0)
        x = torch.rand(B, 1, H, W)
        x_recon = torch.rand(B, 1, H, W)  # NOT clamped — let the loss handle it
        mu = torch.randn(B, 4)
        logvar = torch.full((B, 4), -1.0)  # safe logvar
        freq_khz = torch.linspace(20.0, 120.0, H)
        dFdt_true = torch.zeros(B, W - 1)
        valid_mask = torch.ones(B, W - 1, dtype=torch.bool)
        return x_recon, x, mu, logvar, freq_khz, dFdt_true, valid_mask

    def test_total_loss_keys_and_scalar_shape(self):
        """Spec: returns dict with keys 'total','recon','kl','deriv','curv'; all scalar."""
        x_recon, x, mu, logvar, freq_khz, dFdt_true, valid_mask = self._make_tiny_batch()
        cfg = ShapeVAEDerivConfig(image_size=16, latent_dim=4, base_channels=8)
        result = total_shape_vae_loss(
            x_recon, x, mu, logvar, freq_khz, dFdt_true, valid_mask, cfg
        )
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        for key in ("total", "recon", "kl", "deriv", "curv"):
            assert key in result, f"Missing key '{key}'"
            assert result[key].ndim == 0, (
                f"'{key}' should be scalar; got shape {result[key].shape}"
            )

    def test_total_loss_reduces_to_baseline_elbo(self):
        """Spec: with lambda_d=0, lambda_c=0, mask_recon=False, recon='bce',
        ('recon','kl') and 'total' must match image_vae_loss(x_recon,x,mu,logvar,beta=cfg.beta)
        within 1e-4.

        image_vae_loss returns (loss, recon, kl). We compare element by element.
        Note: image_vae_loss sanitizes x_recon (nan_to_num + clamp) internally;
        total_shape_vae_loss with the same recon='bce' path should do the same.
        """
        # Import baseline from train_contour_vae_v2 (which lives in scripts/)
        from train_contour_vae_v2 import image_vae_loss

        B, H, W = 2, 16, 8
        torch.manual_seed(42)
        x = torch.rand(B, 1, H, W)
        # x_recon already in [0,1] to avoid sanitization path differences
        x_recon = torch.rand(B, 1, H, W)
        mu = torch.randn(B, 4)
        logvar = torch.full((B, 4), -1.0)
        freq_khz = torch.linspace(20.0, 120.0, H)
        dFdt_true = torch.zeros(B, W - 1)
        valid_mask = torch.ones(B, W - 1, dtype=torch.bool)

        cfg = ShapeVAEDerivConfig(
            image_size=16,
            latent_dim=4,
            base_channels=8,
            beta=0.1,
            lambda_d=0.0,
            lambda_c=0.0,
            recon="bce",
            mask_recon=False,
        )

        result = total_shape_vae_loss(
            x_recon, x, mu, logvar, freq_khz, dFdt_true, valid_mask, cfg
        )

        baseline_loss, baseline_recon, baseline_kl = image_vae_loss(
            x_recon, x, mu, logvar, beta=cfg.beta
        )

        assert abs(float(result["recon"]) - float(baseline_recon)) < 1e-4, (
            f"recon mismatch vs image_vae_loss: "
            f"new={result['recon']:.6f}, baseline={baseline_recon:.6f}"
        )
        assert abs(float(result["kl"]) - float(baseline_kl)) < 1e-4, (
            f"kl mismatch vs image_vae_loss: "
            f"new={result['kl']:.6f}, baseline={baseline_kl:.6f}"
        )
        assert abs(float(result["total"]) - float(baseline_loss)) < 1e-4, (
            f"total mismatch vs image_vae_loss ELBO: "
            f"new={result['total']:.6f}, baseline={baseline_loss:.6f}"
        )

    def test_total_loss_formula_weighted_sum(self):
        """Spec: total = recon + cfg.beta*kl + cfg.lambda_d*deriv + cfg.lambda_c*curv."""
        x_recon, x, mu, logvar, freq_khz, dFdt_true, valid_mask = self._make_tiny_batch()
        d2_true = torch.zeros(x.shape[0], x.shape[-1] - 2)
        valid_mask2 = torch.ones(x.shape[0], x.shape[-1] - 2, dtype=torch.bool)

        cfg = ShapeVAEDerivConfig(
            image_size=16, latent_dim=4, base_channels=8,
            beta=0.3, lambda_d=2.0, lambda_c=0.5
        )
        result = total_shape_vae_loss(
            x_recon, x, mu, logvar, freq_khz, dFdt_true, valid_mask, cfg,
            d2_true=d2_true, valid_mask2=valid_mask2
        )
        expected_total = (
            float(result["recon"])
            + cfg.beta * float(result["kl"])
            + cfg.lambda_d * float(result["deriv"])
            + cfg.lambda_c * float(result["curv"])
        )
        # float32 sum at magnitude ~3.7e3 -> 1 ULP ~4e-4; an absolute 1e-5 is
        # tighter than representable. Identity unchanged; tolerance is float32-
        # aware (relative). Approved 2026-05-27.
        assert math.isclose(float(result["total"]), expected_total, rel_tol=1e-5), (
            f"total formula wrong: total={result['total']:.6f}, computed={expected_total:.6f}"
        )


# ============================================================
# 18-20: ShapeVAEDerivConfig
# ============================================================

class TestShapeVAEDerivConfig:

    def test_config_defaults(self):
        """Spec: exact field defaults as listed in the design.

        beta==0.1 (NOT 1.0) is a critical guard: the dead-end used beta=1.0,
        which over-regularizes contour-VAE training. Low beta is by design.
        """
        cfg = ShapeVAEDerivConfig()
        assert cfg.image_size == 256, f"image_size default should be 256, got {cfg.image_size}"
        assert cfg.latent_dim == 32, f"latent_dim default should be 32, got {cfg.latent_dim}"
        assert cfg.base_channels == 32, f"base_channels default should be 32, got {cfg.base_channels}"
        assert cfg.beta == 0.1, (
            f"beta default MUST be 0.1 (NOT 1.0 — the dead-end used 1.0); got {cfg.beta}"
        )
        assert cfg.lambda_d == 1.0, f"lambda_d default should be 1.0, got {cfg.lambda_d}"
        assert cfg.lambda_c == 0.0, f"lambda_c default should be 0.0, got {cfg.lambda_c}"
        assert cfg.tau == 0.05, f"tau default should be 0.05, got {cfg.tau}"
        assert cfg.recon == "bce", f"recon default should be 'bce', got '{cfg.recon}'"
        assert cfg.mask_recon is False, f"mask_recon default should be False, got {cfg.mask_recon}"

    def test_config_invalid_recon_raises(self):
        """Spec: recon not in allowed set raises ValueError."""
        with pytest.raises(ValueError, match=r"(?i)recon"):
            ShapeVAEDerivConfig(recon="foo")

    def test_config_negative_lambda_d_raises(self):
        """Spec: negative lambda_d raises ValueError (loss weight must be >= 0)."""
        with pytest.raises(ValueError):
            ShapeVAEDerivConfig(lambda_d=-0.5)

    def test_config_is_frozen(self):
        """Spec: frozen dataclass — mutation raises FrozenInstanceError or TypeError."""
        cfg = ShapeVAEDerivConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.beta = 2.0  # type: ignore[misc]


# ============================================================
# 21-22 + additional: load_ridge_cache
# ============================================================

class TestLoadRidgeCache:

    def test_load_ridge_cache_valid(self, tmp_path):
        """Spec: valid npz with dFdt_true (N,W-1) + valid_mask (N,W-1) -> dict, keys present,
        shapes returned correctly."""
        N, W = 50, 16
        dFdt = np.random.randn(N, W - 1).astype(np.float32)
        vmask = np.ones((N, W - 1), dtype=bool)
        npz_path = tmp_path / "ridge_cache.npz"
        np.savez(str(npz_path), dFdt_true=dFdt, valid_mask=vmask)

        result = load_ridge_cache(npz_path)

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "dFdt_true" in result, "Missing key 'dFdt_true'"
        assert "valid_mask" in result, "Missing key 'valid_mask'"
        assert result["dFdt_true"].shape == (N, W - 1), (
            f"Expected dFdt_true shape ({N},{W-1}), got {result['dFdt_true'].shape}"
        )
        assert result["valid_mask"].shape == (N, W - 1), (
            f"Expected valid_mask shape ({N},{W-1}), got {result['valid_mask'].shape}"
        )

    def test_load_ridge_cache_with_optional_d2(self, tmp_path):
        """Optional d2_true and valid_mask2 arrays are loaded when present."""
        N, W = 30, 12
        dFdt = np.random.randn(N, W - 1).astype(np.float32)
        vmask = np.ones((N, W - 1), dtype=bool)
        d2 = np.random.randn(N, W - 2).astype(np.float32)
        vmask2 = np.ones((N, W - 2), dtype=bool)
        npz_path = tmp_path / "ridge_cache_full.npz"
        np.savez(str(npz_path), dFdt_true=dFdt, valid_mask=vmask, d2_true=d2, valid_mask2=vmask2)

        result = load_ridge_cache(npz_path)
        assert "d2_true" in result, "Missing key 'd2_true' (present in npz)"
        assert "valid_mask2" in result, "Missing key 'valid_mask2' (present in npz)"
        assert result["d2_true"].shape == (N, W - 2)

    def test_load_ridge_cache_mismatch_raises(self, tmp_path):
        """Spec: arrays with different first-dim N raise ValueError."""
        N, W = 50, 12
        dFdt = np.random.randn(N, W - 1).astype(np.float32)
        # valid_mask has DIFFERENT N — this is the mismatch that must be caught
        vmask = np.ones((N + 5, W - 1), dtype=bool)
        npz_path = tmp_path / "ridge_cache_bad.npz"
        np.savez(str(npz_path), dFdt_true=dFdt, valid_mask=vmask)

        with pytest.raises(ValueError):
            load_ridge_cache(npz_path)


# ============================================================
# Integration: the realistic training path (model forward -> band crop ->
# total_shape_vae_loss). Guards BLOCKER-1 (recon shape mismatch), which the
# isolated total_shape_vae_loss tests missed because they used matched shapes.
# ============================================================

class TestRunEpochForwardPath:

    def _setup(self):
        """Tiny end-to-end setup mirroring _run_epoch: full 16x16 input/recon,
        derivative term on the band crop with a band-length freq axis."""
        B = 2
        image_size, f_in, t_in = 16, 10, 12
        cfg = ShapeVAEDerivConfig(
            image_size=image_size, latent_dim=4, base_channels=8, lambda_d=1.0
        )
        model = _mod.ImageVAE(cfg.image_vae_config())
        padding = _mod.PaddingSpec.for_shape(f_in, t_in, image_size)
        x = torch.rand(B, 1, image_size, image_size)  # full padded input in [0,1]
        freq_khz = torch.linspace(20.0, 120.0, f_in)  # band axis (length f_in)
        dFdt_true = torch.zeros(B, t_in - 1)
        valid_mask = torch.ones(B, t_in - 1, dtype=torch.bool)
        return B, cfg, model, padding, x, freq_khz, dFdt_true, valid_mask, f_in, t_in

    def test_run_epoch_path_no_crash_and_finite(self):
        """The exact composition _run_epoch uses must run and give finite loss.

        FULL recon (x_recon vs x, both 16x16) + derivative on the band crop
        (deriv_img = extract_band_region(x_recon) = (B,1,10,12)) with the
        band-length freq axis. This is the path BLOCKER-1 broke.
        """
        B, cfg, model, padding, x, freq_khz, dFdt_true, valid_mask, f_in, t_in = self._setup()
        x_recon, mu, logvar = model(x)
        assert x_recon.shape == (B, 1, 16, 16)
        recon_band = _mod.extract_band_region(x_recon, padding)
        assert recon_band.shape == (B, 1, f_in, t_in)

        losses = total_shape_vae_loss(
            x_recon, x, mu, logvar, freq_khz, dFdt_true, valid_mask, cfg,
            deriv_img=recon_band,
        )
        for key in ("total", "recon", "kl", "deriv", "curv"):
            assert key in losses and losses[key].ndim == 0
            assert torch.isfinite(losses[key]).all(), f"{key} is not finite"

    def test_run_epoch_path_backward_flows(self):
        """The total loss must backprop into the model (the derivative term is
        only useful if its gradient reaches the encoder/decoder)."""
        B, cfg, model, padding, x, freq_khz, dFdt_true, valid_mask, f_in, t_in = self._setup()
        x_recon, mu, logvar = model(x)
        recon_band = _mod.extract_band_region(x_recon, padding)
        losses = total_shape_vae_loss(
            x_recon, x, mu, logvar, freq_khz, dFdt_true, valid_mask, cfg,
            deriv_img=recon_band,
        )
        losses["total"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "no gradients reached the model parameters"
        assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"

    def test_recon_target_shape_mismatch_raises(self):
        """REGRESSION GUARD for BLOCKER-1: passing a band-cropped recon as
        x_recon against a full-image target must fail loudly (not broadcast)."""
        B, cfg, model, padding, x, freq_khz, dFdt_true, valid_mask, f_in, t_in = self._setup()
        x_recon, mu, logvar = model(x)
        recon_band = _mod.extract_band_region(x_recon, padding)  # (B,1,10,12)
        # The old bug: recon_band as x_recon, full x as target -> BCE shape clash.
        with pytest.raises((ValueError, RuntimeError)):
            total_shape_vae_loss(
                recon_band, x, mu, logvar, freq_khz, dFdt_true, valid_mask, cfg,
            )
