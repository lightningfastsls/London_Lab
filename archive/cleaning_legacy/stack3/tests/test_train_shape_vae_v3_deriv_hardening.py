"""Adversarial hardening tests for train_shape_vae_v3_deriv + extract_ridge_targets_v3.

These tests guard real-data edge cases that WILL occur in the 69,293-patch rig
corpus but are NOT covered by the test-architect's 32-test baseline:

    1. Degenerate / empty ridge (C06 all-zero / all-NaN case) in derivative_targets
    2. Short active run in derivative_targets (only interior bins valid)
    3. All-masked derivative_loss (valid_mask all False -> no div-by-zero, result == 0)
    4. soft_argmax_frequency with very small tau (numerical stability)
    5. soft_argmax_frequency with a uniform (flat) column (expected freq == mean)
    6. load_ridge_cache with a MISSING required key (only dFdt_true present, no valid_mask)
    7. load_ridge_cache with ONLY valid_mask present (no dFdt_true)
    8. derivative_targets with NaN-in-interior (verify no NaN leaks, masked bins zeroed)
    9. extract_band_region geometry: correct offsets, correct shape, autograd preserved
   10. derivative_targets with valid_mask2 three-point window on interior-NaN run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path bootstrap — mirrors the existing test file exactly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_EXPERIMENTS_ROOT = _SCRIPTS_ROOT / "experiments"
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SCRIPTS_ROOT, _EXPERIMENTS_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_shape_vae_v3_deriv as _mod  # noqa: E402
import extract_ridge_targets_v3 as _ext  # noqa: E402

soft_argmax_frequency = _mod.soft_argmax_frequency
derivative_loss       = _mod.derivative_loss
load_ridge_cache      = _mod.load_ridge_cache
extract_band_region   = _mod.extract_band_region
PaddingSpec           = _mod.PaddingSpec

derivative_targets = _ext.derivative_targets


# ===========================================================================
# 1 + 2: derivative_targets — degenerate / empty ridge (C06 case)
# ===========================================================================


class TestDerivativeTargetsDegenerateRidge:

    def test_all_nan_fm_gives_all_zero_and_all_false(self):
        """C06 case: track_ridge found no active column -> fm is all NaN.

        derivative_targets must return dFdt_true all-zero, valid_mask all-False,
        and no NaN anywhere in the output arrays.

        Guards the documented cluster-C06 degenerate patches (12 all-zero patches
        whose ridge is fully silent).
        """
        T = 20
        fm = np.full(T, np.nan, dtype=np.float64)

        dFdt, vmask, d2, vmask2 = derivative_targets(fm)

        assert dFdt.shape == (T - 1,), f"dFdt shape wrong: {dFdt.shape}"
        assert vmask.shape == (T - 1,), f"vmask shape wrong: {vmask.shape}"
        assert d2.shape == (T - 2,), f"d2 shape wrong: {d2.shape}"
        assert vmask2.shape == (T - 2,), f"vmask2 shape wrong: {vmask2.shape}"

        assert not np.any(vmask),  "all-NaN fm must give all-False valid_mask"
        assert not np.any(vmask2), "all-NaN fm must give all-False valid_mask2"
        assert not np.any(np.isnan(dFdt)),  "NaN leaked into dFdt_true"
        assert not np.any(np.isnan(d2)),    "NaN leaked into d2_true"
        assert np.all(dFdt == 0.0),  "all-NaN fm -> dFdt must be all zeros"
        assert np.all(d2 == 0.0),    "all-NaN fm -> d2 must be all zeros"

    def test_short_active_run_validity_boundaries(self):
        """Only the bins whose BOTH endpoints are finite should be valid.

        fm: NaN NaN 30 35 40 NaN NaN  (7 columns, active run is indices 2-4)

        First-difference bins (indices into fm[1:]-fm[:-1]):
          0 (NaN,NaN) -> invalid
          1 (NaN,30)  -> invalid
          2 (30,35)   -> VALID
          3 (35,40)   -> VALID
          4 (40,NaN)  -> invalid
          5 (NaN,NaN) -> invalid

        So exactly bins 2 and 3 should be True.
        """
        fm = np.array([np.nan, np.nan, 30.0, 35.0, 40.0, np.nan, np.nan])

        dFdt, vmask, d2, vmask2 = derivative_targets(fm)

        expected_vmask = np.array([False, False, True, True, False, False])
        np.testing.assert_array_equal(
            vmask, expected_vmask,
            err_msg=f"valid_mask mismatch:\n  got      {vmask}\n  expected {expected_vmask}",
        )
        # Values at valid bins: 35-30=5, 40-35=5
        assert dFdt[2] == pytest.approx(5.0), f"dFdt[2] should be 5.0, got {dFdt[2]}"
        assert dFdt[3] == pytest.approx(5.0), f"dFdt[3] should be 5.0, got {dFdt[3]}"
        # Values at invalid bins must be 0 (not carrying NaN-contaminated values)
        for i in [0, 1, 4, 5]:
            assert dFdt[i] == 0.0, f"masked dFdt[{i}] must be 0.0, got {dFdt[i]}"
        # No NaN in outputs
        assert not np.any(np.isnan(dFdt)), "NaN in dFdt"
        assert not np.any(np.isnan(d2)),   "NaN in d2"


# ===========================================================================
# 3: derivative_loss — all-masked valid_mask
# ===========================================================================


class TestDerivativeLossAllMasked:

    def test_all_false_mask_gives_finite_zero_loss(self):
        """valid_mask all False -> _masked_mse uses denom=1 (clamped), returns 0.

        This guards the div-by-zero path in _masked_mse when no column is active.
        A real-data batch containing only C06-type patches could produce this
        entirely all-False mask slice.
        """
        B, H, W = 2, 16, 12
        torch.manual_seed(7)
        img = torch.rand(B, 1, H, W)
        freq_khz = torch.linspace(20.0, 120.0, H)
        dFdt_true = torch.randn(B, W - 1)
        valid_mask = torch.zeros(B, W - 1, dtype=torch.bool)  # ALL False

        result = derivative_loss(img, freq_khz, dFdt_true, valid_mask=valid_mask)

        assert torch.isfinite(result["deriv"]), (
            f"deriv is not finite with all-False mask: {result['deriv']}"
        )
        assert float(result["deriv"]) == pytest.approx(0.0), (
            f"all-False mask -> deriv should be 0.0, got {result['deriv']}"
        )
        assert torch.isfinite(result["total"]), "total is not finite with all-False mask"

    def test_all_false_mask2_gives_finite_zero_curv(self):
        """valid_mask2 all False -> curvature component must also be finite and zero."""
        B, H, W = 2, 16, 10
        torch.manual_seed(11)
        img = torch.rand(B, 1, H, W)
        freq_khz = torch.linspace(20.0, 120.0, H)
        dFdt_true = torch.randn(B, W - 1)
        valid_mask = torch.ones(B, W - 1, dtype=torch.bool)
        d2_true = torch.randn(B, W - 2)
        valid_mask2 = torch.zeros(B, W - 2, dtype=torch.bool)  # ALL False

        result = derivative_loss(
            img, freq_khz, dFdt_true,
            valid_mask=valid_mask,
            lambda_c=1.0,
            d2_true=d2_true,
            valid_mask2=valid_mask2,
        )

        assert torch.isfinite(result["curv"]), (
            f"curv is not finite with all-False mask2: {result['curv']}"
        )
        assert float(result["curv"]) == pytest.approx(0.0), (
            f"all-False mask2 -> curv should be 0.0, got {result['curv']}"
        )


# ===========================================================================
# 4 + 5: soft_argmax_frequency — numerical extremes
# ===========================================================================


class TestSoftArgmaxNumericalExtremes:

    def test_very_small_tau_no_nan_or_inf(self):
        """Extremely small tau must not produce NaN or Inf.

        With tau=1e-4 and an image in [0, 1], the softmax exponentials
        exp(img/tau) can overflow float32 in principle, but PyTorch softmax
        is numerically stabilised (subtracts max before exp). Verify here.
        """
        B, H, W = 2, 32, 10
        torch.manual_seed(42)
        img = torch.rand(B, 1, H, W)  # already in [0, 1]
        freq_khz = torch.linspace(20.0, 150.0, H)

        out = soft_argmax_frequency(img, freq_khz, tau=1e-4)

        assert out.shape == (B, W)
        assert torch.isfinite(out).all(), (
            f"soft_argmax_frequency with tau=1e-4 produced non-finite values: "
            f"max={out.max()}, min={out.min()}, nan_count={out.isnan().sum()}"
        )

    def test_uniform_column_gives_mean_frequency(self):
        """A flat (uniform) column has no dominant row.

        With a constant activation across all rows, softmax assigns equal weight
        to every row, so the expected frequency equals the mean of freq_khz.
        This tests that no spurious localization occurs for toneless noise.
        """
        B, H, W = 1, 16, 6
        freq_khz = torch.linspace(20.0, 120.0, H)
        expected_mean = float(freq_khz.mean())

        # All columns perfectly uniform — every value identical
        img = torch.ones(B, 1, H, W)

        out = soft_argmax_frequency(img, freq_khz, tau=0.05)

        assert out.shape == (B, W)
        # Every column should yield exactly the mean frequency
        assert torch.allclose(out, torch.full_like(out, expected_mean), atol=1e-3), (
            f"Uniform image: expected all columns ≈ {expected_mean:.3f} kHz, "
            f"got range [{out.min():.3f}, {out.max():.3f}]"
        )

    def test_large_tau_also_finite(self):
        """Large tau (near-uniform softmax) must also be finite — guards the
        opposite extreme from the small-tau test."""
        B, H, W = 2, 16, 8
        torch.manual_seed(3)
        img = torch.rand(B, 1, H, W)
        freq_khz = torch.linspace(20.0, 120.0, H)

        out = soft_argmax_frequency(img, freq_khz, tau=1000.0)

        assert torch.isfinite(out).all(), "soft_argmax with large tau produced non-finite"


# ===========================================================================
# 6 + 7: load_ridge_cache — missing required keys
# ===========================================================================


class TestLoadRidgeCacheMissingKeys:

    def test_only_dFdt_true_present_raises(self, tmp_path):
        """An npz with dFdt_true but NO valid_mask must raise ValueError.

        The existing tests cover: valid case and mismatched-N. They do NOT cover
        a partially populated cache where valid_mask was simply not written.
        """
        N, W = 20, 10
        dFdt = np.random.randn(N, W - 1).astype(np.float32)
        npz_path = tmp_path / "cache_no_vmask.npz"
        np.savez(str(npz_path), dFdt_true=dFdt)  # valid_mask is absent

        with pytest.raises(ValueError, match="valid_mask"):
            load_ridge_cache(npz_path)

    def test_only_valid_mask_present_raises(self, tmp_path):
        """An npz with valid_mask but NO dFdt_true must raise ValueError."""
        N, W = 20, 10
        vmask = np.ones((N, W - 1), dtype=bool)
        npz_path = tmp_path / "cache_no_dFdt.npz"
        np.savez(str(npz_path), valid_mask=vmask)  # dFdt_true is absent

        with pytest.raises(ValueError, match="dFdt_true"):
            load_ridge_cache(npz_path)

    def test_empty_npz_raises(self, tmp_path):
        """Completely empty npz (no arrays at all) must raise ValueError."""
        npz_path = tmp_path / "cache_empty.npz"
        np.savez(str(npz_path))  # no arrays

        with pytest.raises(ValueError):
            load_ridge_cache(npz_path)


# ===========================================================================
# 8: derivative_targets — NaN in the interior (track_ridge can produce this)
# ===========================================================================


class TestDerivativeTargetsInteriorNaN:

    def test_interior_nan_no_nan_leaks(self):
        """fm finite at the ends but with a NaN gap in the middle.

        track_ridge can produce interior NaN when a column falls below the
        silence threshold in a run that is otherwise active. Verify:
          - No NaN appears anywhere in dFdt, d2, vmask, vmask2
          - The bins touching the gap are marked invalid (vmask False)
          - The bins NOT touching the gap retain their finite derivatives

        fm = [10, 20, 30, NaN, 50, 60, 70]  (7 columns)

        First differences (6 bins):
          bin 0: (10,20) -> both finite -> valid, dFdt=10
          bin 1: (20,30) -> both finite -> valid, dFdt=10
          bin 2: (30,NaN) -> invalid -> 0
          bin 3: (NaN,50) -> invalid -> 0
          bin 4: (50,60) -> both finite -> valid, dFdt=10
          bin 5: (60,70) -> both finite -> valid, dFdt=10
        """
        fm = np.array([10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0])

        dFdt, vmask, d2, vmask2 = derivative_targets(fm)

        # --- no NaN in any output ---
        assert not np.any(np.isnan(dFdt)),  "NaN leaked into dFdt"
        assert not np.any(np.isnan(d2)),    "NaN leaked into d2"
        assert not np.any(np.isnan(vmask.astype(float)))
        assert not np.any(np.isnan(vmask2.astype(float)))

        # --- first-difference validity ---
        expected_vmask = np.array([True, True, False, False, True, True])
        np.testing.assert_array_equal(
            vmask, expected_vmask,
            err_msg=f"vmask mismatch:\n  got      {vmask}\n  expected {expected_vmask}",
        )

        # --- valid bins carry correct derivatives ---
        assert dFdt[0] == pytest.approx(10.0), f"dFdt[0] should be 10, got {dFdt[0]}"
        assert dFdt[1] == pytest.approx(10.0), f"dFdt[1] should be 10, got {dFdt[1]}"
        assert dFdt[4] == pytest.approx(10.0), f"dFdt[4] should be 10, got {dFdt[4]}"
        assert dFdt[5] == pytest.approx(10.0), f"dFdt[5] should be 10, got {dFdt[5]}"

        # --- invalid bins are zeroed ---
        assert dFdt[2] == 0.0, f"masked dFdt[2] must be 0, got {dFdt[2]}"
        assert dFdt[3] == 0.0, f"masked dFdt[3] must be 0, got {dFdt[3]}"

    def test_interior_nan_second_derivative_validity(self):
        """The three-point curvature window spanning the NaN gap must be False.

        fm = [10, 20, 30, NaN, 50, 60, 70]

        Second-difference windows (5 bins, indices into d2[i] = fm[i+2]-2*fm[i+1]+fm[i]):
          window 0: (10,20,30)   -> all finite -> valid
          window 1: (20,30,NaN)  -> NaN present -> invalid
          window 2: (30,NaN,50)  -> NaN present -> invalid
          window 3: (NaN,50,60)  -> NaN present -> invalid
          window 4: (50,60,70)   -> all finite -> valid
        """
        fm = np.array([10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0])

        _, _, d2, vmask2 = derivative_targets(fm)

        expected_vmask2 = np.array([True, False, False, False, True])
        np.testing.assert_array_equal(
            vmask2, expected_vmask2,
            err_msg=f"vmask2 mismatch:\n  got      {vmask2}\n  expected {expected_vmask2}",
        )

        # Valid window 0: 30 - 2*20 + 10 = 0 (linear ramp -> zero curvature)
        assert d2[0] == pytest.approx(0.0, abs=1e-5), f"d2[0] should be 0 (linear), got {d2[0]}"
        # Valid window 4: 70 - 2*60 + 50 = 0 (linear ramp -> zero curvature)
        assert d2[4] == pytest.approx(0.0, abs=1e-5), f"d2[4] should be 0 (linear), got {d2[4]}"
        # Invalid windows must be 0
        for i in [1, 2, 3]:
            assert d2[i] == 0.0, f"masked d2[{i}] must be 0, got {d2[i]}"
        # No NaN
        assert not np.any(np.isnan(d2)), "NaN in d2 with interior-NaN fm"


# ===========================================================================
# 9: extract_band_region — geometry, correct offsets, autograd propagation
# ===========================================================================


class TestExtractBandRegion:

    def test_correct_shape_and_offset(self):
        """extract_band_region must return the exact (B,1,f_in,t_in) sub-block
        at the correct (pad_f_top, pad_t_left) offset.

        Build a padded tensor where every spatial position holds a unique value
        (row * image_size + col). The crop must contain exactly the values for
        row indices [pad_f_top, pad_f_top+f_in) and all columns (pad_t_left=0
        for canonical contour patches which are wider than they are tall).
        """
        B = 2
        f_in, t_in, image_size = 40, 50, 64  # small but realistic aspect ratio

        padding = PaddingSpec.for_shape(f_in, t_in, image_size)

        # Build a tensor where position [b, 0, r, c] = float(r * image_size + c)
        # so we can verify the slice offset exactly.
        rows = torch.arange(image_size, dtype=torch.float32).view(1, 1, -1, 1)
        cols = torch.arange(image_size, dtype=torch.float32).view(1, 1, 1, -1)
        img = (rows * image_size + cols).expand(B, 1, image_size, image_size)

        cropped = extract_band_region(img, padding)

        assert cropped.shape == (B, 1, f_in, t_in), (
            f"extract_band_region shape mismatch: expected ({B},1,{f_in},{t_in}), "
            f"got {tuple(cropped.shape)}"
        )

        # Verify the top-left corner of the crop
        f0 = padding.pad_f_top
        t0 = padding.pad_t_left
        expected_topleft = float(f0 * image_size + t0)
        actual_topleft = float(cropped[0, 0, 0, 0])
        assert actual_topleft == pytest.approx(expected_topleft), (
            f"Top-left value mismatch: crop[0,0,0,0]={actual_topleft}, "
            f"expected {expected_topleft} (pad_f_top={f0}, pad_t_left={t0})"
        )

        # Verify the full crop matches the expected slice from the full tensor
        expected_crop = img[:, :, f0: f0 + f_in, t0: t0 + t_in]
        assert torch.equal(cropped, expected_crop), (
            "extract_band_region crop does not match manual slicing"
        )

    def test_autograd_propagates_through_crop(self):
        """Torch slicing preserves the autograd graph.

        A gradient through the cropped region must flow back to the padded
        tensor's grad — the derivative term's entire gradient path depends on
        this. Verify requires_grad propagates and .backward() works.
        """
        B = 1
        f_in, t_in, image_size = 30, 40, 64
        padding = PaddingSpec.for_shape(f_in, t_in, image_size)

        img = torch.rand(B, 1, image_size, image_size, requires_grad=True)
        cropped = extract_band_region(img, padding)

        # The crop is part of the computation graph
        assert cropped.requires_grad, "extract_band_region result must require grad"

        # Backward through the crop
        loss = cropped.sum()
        loss.backward()

        assert img.grad is not None, "No gradient flowed back to padded tensor"

        # Only the crop region gets gradient (1.0); padding positions get 0.0
        f0 = padding.pad_f_top
        t0 = padding.pad_t_left

        crop_grad = img.grad[0, 0, f0: f0 + f_in, t0: t0 + t_in]
        assert torch.all(crop_grad == 1.0), (
            "Gradient in crop region should be all 1.0 from sum()"
        )

        # Padding rows above the crop should have zero gradient
        if f0 > 0:
            pad_grad = img.grad[0, 0, :f0, :]
            assert torch.all(pad_grad == 0.0), (
                "Padding rows above crop region should have zero gradient"
            )

    def test_square_input_passthrough(self):
        """When f_in == t_in == image_size, no padding is needed and the crop
        should be the entire tensor (or a full-size slice with zero offsets)."""
        B = 2
        image_size = 16
        f_in, t_in = image_size, image_size
        padding = PaddingSpec.for_shape(f_in, t_in, image_size)

        img = torch.rand(B, 1, image_size, image_size)
        cropped = extract_band_region(img, padding)

        assert cropped.shape == (B, 1, f_in, t_in)
        # The crop must equal the full tensor (no padding needed)
        assert torch.equal(cropped, img[:, :, padding.pad_f_top: padding.pad_f_top + f_in,
                                        padding.pad_t_left: padding.pad_t_left + t_in])
