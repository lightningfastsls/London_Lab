"""Tests for scripts/experiments/train_shape_vae_v3_hybrid.py

These are PRE-IMPLEMENTATION SPEC tests written by test-architect BEFORE the
module exists. They define what correctness looks like. DO NOT weaken assertions
to make tests pass — fix the implementation instead.

ROADMAP test plan coverage:
  1. ShapeVAEv3Config defaults and field presence       -> test_config_default_values
  2. ShapeVAEv3Config validation: latent_dim <= 0       -> test_config_invalid_latent_dim
  3. ShapeVAEv3Config validation: beta < 0              -> test_config_invalid_beta
  4. ShapeVAEv3Config validation: lambda_nt < 0         -> test_config_invalid_lambda_nt
  5. ShapeVAEv3Config validation: lambda_recon < 0      -> test_config_invalid_lambda_recon
  6. ShapeVAEv3Config validation: lambda_lc < 0         -> test_config_invalid_lambda_lc
  7. ShapeVAEv3Config validation: lambda_deriv < 0      -> test_config_invalid_lambda_deriv
  8. ShapeVAEv3Config validation: max_df_hz < 0         -> test_config_invalid_max_df_hz
  9. ShapeVAEv3Config validation: max_dt_frames < 0     -> test_config_invalid_max_dt_frames
 10. ShapeVAEv3Config validation: recon_anneal_epochs<1 -> test_config_invalid_recon_anneal_epochs
 11. ShapeVAEv3Config validation: image_size not pow2   -> test_config_invalid_image_size_not_pow2
 12. ShapeVAEv3Config validation: image_size < 16       -> test_config_invalid_image_size_too_small
 13. soft_argmax_ridge output shape                     -> test_soft_argmax_ridge_output_shape
 14. soft_argmax_ridge differentiable                   -> test_soft_argmax_ridge_gradient_flows
 15. soft_argmax_ridge one-hot correctness              -> test_soft_argmax_ridge_one_hot_correctness
 16. soft_argmax_ridge monotonic response               -> test_soft_argmax_ridge_monotonic
 17. soft_argmax_ridge values in [fmin, fmax]           -> test_soft_argmax_ridge_values_in_range
 18. nt_xent scalar, finite, nonneg                     -> test_nt_xent_finite_nonneg
 19. nt_xent aligned < shuffled                         -> test_nt_xent_aligned_lower_than_shuffled
 20. nt_xent gradient flows                             -> test_nt_xent_gradient_flows
 21. latent_consistency exact zero for equal inputs     -> test_latent_consistency_zero_identical
 22. latent_consistency positive for different inputs   -> test_latent_consistency_positive
 23. latent_consistency gradient flows                  -> test_latent_consistency_gradient_flows
 24. derivative_loss zero when decoded == true          -> test_derivative_loss_zero_identical
 25. derivative_loss positive/finite when different     -> test_derivative_loss_positive_when_different
 26. derivative_loss mask all-false                     -> test_derivative_loss_mask_all_masked
 27. derivative_loss gradient flows                     -> test_derivative_loss_gradient_flows
 28. augment_pitch_time_shift output shape              -> test_augment_output_shape
 29. augment in-band guarantee (battery)                -> test_augment_in_band_guarantee
 30. augment full-band ridge gets df=0                  -> test_augment_full_band_ridge_no_shift
 31. augment df=0,dt=0 returns unchanged                -> test_augment_zero_shift_unchanged
 32. augment vertical shift is non-wrapping             -> test_augment_vertical_nonwrapping
 33. annealed_weights epoch 0                           -> test_annealed_weights_epoch_zero
 34. annealed_weights epoch >= start+epochs             -> test_annealed_weights_full_at_end
 35. annealed_weights midpoint                          -> test_annealed_weights_midpoint
 36. annealed_weights nt/lc never change                -> test_annealed_weights_nt_lc_constant
 37. annealed_weights recon monotone                    -> test_annealed_weights_monotone
 38. hybrid_loss total is finite                        -> test_hybrid_loss_finite
 39. hybrid_loss components dict keys                   -> test_hybrid_loss_components_keys
 40. hybrid_loss term isolation (one nonzero weight)    -> test_hybrid_loss_term_isolation
 41. hybrid_loss all-zero weights -> total 0            -> test_hybrid_loss_all_zero_weights

Additional coverage (recurring gap patterns):
  - 3-dim input to soft_argmax_ridge [B,H,W]            -> test_soft_argmax_ridge_3dim_input
  - Config round-trip (frozen dataclass immutability)   -> test_config_frozen
  - nt_xent batch size 1 edge case                      -> test_nt_xent_batch_size_1
  - latent_consistency single item                      -> test_latent_consistency_single_item
  - derivative_loss with partial mask                   -> test_derivative_loss_partial_mask
  - augment with generator for determinism              -> test_augment_deterministic_with_seed
  - augment single sample                               -> test_augment_single_sample

Total: 48 tests (41 from ROADMAP plan, 7 additional gap patterns)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Repo root on sys.path so `scripts.experiments.*` resolves
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Corpus constants (src/ must be on sys.path — mirrors conftest.py setup)
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from usv_spectrogram.corpus import USV_FREQ_MIN_HZ, USV_FREQ_MAX_HZ  # noqa: E402

# ---------------------------------------------------------------------------
# Import the target module — if it doesn't exist these tests FAIL (correct TDD
# behavior; do NOT catch and skip).
# ---------------------------------------------------------------------------
from scripts.experiments.train_shape_vae_v3_hybrid import (  # noqa: E402
    ShapeVAEv3Config,
    annealed_weights,
    augment_pitch_time_shift,
    derivative_loss,
    hybrid_loss,
    latent_consistency,
    nt_xent,
    soft_argmax_ridge,
)

# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------
_SEED = 42


def _seed_all() -> torch.Generator:
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    g = torch.Generator()
    g.manual_seed(_SEED)
    return g


# ===========================================================================
# 1. ShapeVAEv3Config
# ===========================================================================


class TestShapeVAEv3Config:
    """Verifies ShapeVAEv3Config field defaults, types, and validation guards."""

    def test_config_default_values(self) -> None:
        """All documented defaults must be present and equal to spec values."""
        cfg = ShapeVAEv3Config()
        assert cfg.latent_dim == 32
        assert cfg.image_size == 256
        assert cfg.lambda_nt == 1.0
        assert cfg.lambda_recon == 0.05
        assert cfg.beta == 0.05
        assert cfg.lambda_lc == 1.0
        assert cfg.lambda_deriv == 0.1
        assert cfg.nt_temperature == 0.2
        assert cfg.max_df_hz == 15_000.0
        assert cfg.max_dt_frames == 20
        # time_warp_range must be a 2-tuple of floats
        assert cfg.time_warp_range == (0.9, 1.1)
        assert cfg.recon_anneal_start == 10
        assert cfg.recon_anneal_epochs == 20
        assert cfg.softargmax_temp == 1.0

    def test_config_frozen(self) -> None:
        """ShapeVAEv3Config is a frozen dataclass — attribute assignment raises."""
        cfg = ShapeVAEv3Config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.latent_dim = 64  # type: ignore[misc]

    def test_config_invalid_latent_dim(self) -> None:
        """latent_dim <= 0 must raise ValueError."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(latent_dim=0)
        with pytest.raises(ValueError):
            ShapeVAEv3Config(latent_dim=-1)

    def test_config_invalid_beta(self) -> None:
        """beta < 0 must raise ValueError (0.0 is valid — allows KL-free mode)."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(beta=-0.001)

    def test_config_invalid_lambda_nt(self) -> None:
        """lambda_nt < 0 must raise ValueError."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(lambda_nt=-1.0)

    def test_config_invalid_lambda_recon(self) -> None:
        """lambda_recon < 0 must raise ValueError."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(lambda_recon=-0.01)

    def test_config_invalid_lambda_lc(self) -> None:
        """lambda_lc < 0 must raise ValueError."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(lambda_lc=-1.0)

    def test_config_invalid_lambda_deriv(self) -> None:
        """lambda_deriv < 0 must raise ValueError."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(lambda_deriv=-0.1)

    def test_config_invalid_max_df_hz(self) -> None:
        """max_df_hz < 0 must raise ValueError (0 is valid: no pitch shift)."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(max_df_hz=-100.0)

    def test_config_invalid_max_dt_frames(self) -> None:
        """max_dt_frames < 0 must raise ValueError (0 is valid: no time shift)."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(max_dt_frames=-1)

    def test_config_invalid_recon_anneal_epochs(self) -> None:
        """recon_anneal_epochs < 1 must raise ValueError (zero-length ramp is meaningless)."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(recon_anneal_epochs=0)
        with pytest.raises(ValueError):
            ShapeVAEv3Config(recon_anneal_epochs=-5)

    def test_config_invalid_image_size_not_pow2(self) -> None:
        """image_size that is not a power of 2 must raise ValueError."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(image_size=100)
        with pytest.raises(ValueError):
            ShapeVAEv3Config(image_size=48)

    def test_config_invalid_image_size_too_small(self) -> None:
        """image_size < 16 must raise ValueError (4 stride-2 convs require >=16)."""
        with pytest.raises(ValueError):
            ShapeVAEv3Config(image_size=8)


# ===========================================================================
# 2. soft_argmax_ridge
# ===========================================================================


class TestSoftArgmaxRidge:
    """Verifies soft_argmax_ridge shape, differentiability, and numeric correctness."""

    def _make_freqs(self, H: int) -> torch.Tensor:
        """Linearly spaced frequency grid over the corpus USV band."""
        return torch.linspace(float(USV_FREQ_MIN_HZ), float(USV_FREQ_MAX_HZ), H)

    def test_soft_argmax_ridge_output_shape(self) -> None:
        """Output must be (B, W) for [B,1,H,W] input."""
        B, H, W = 4, 16, 32
        freqs = self._make_freqs(H)
        mag = torch.rand(B, 1, H, W)
        out = soft_argmax_ridge(mag, freqs, temp=1.0)
        assert out.shape == (B, W), f"Expected shape ({B}, {W}), got {out.shape}"

    def test_soft_argmax_ridge_3dim_input(self) -> None:
        """[B,H,W] input (no channel dim) must also return (B, W)."""
        B, H, W = 3, 16, 24
        freqs = self._make_freqs(H)
        mag = torch.rand(B, H, W)
        out = soft_argmax_ridge(mag, freqs, temp=1.0)
        assert out.shape == (B, W), f"Expected shape ({B}, {W}), got {out.shape}"

    def test_soft_argmax_ridge_gradient_flows(self) -> None:
        """Gradient must propagate back through soft_argmax_ridge."""
        B, H, W = 2, 16, 8
        freqs = self._make_freqs(H)
        mag = torch.rand(B, 1, H, W, requires_grad=True)
        out = soft_argmax_ridge(mag, freqs, temp=1.0)
        out.sum().backward()
        assert mag.grad is not None, "magnitude.grad is None — no gradient flow"
        assert torch.isfinite(mag.grad).all(), "Non-finite gradients detected"

    def test_soft_argmax_ridge_one_hot_correctness(self) -> None:
        """One-hot column (all mass on row k) with low temp must return ~freqs[k].

        Using temp=0.01 and a 16-bin grid (bin width = 100k/15 ≈ 6667 Hz).
        With nearly all softmax mass on row k, expected freq ≈ freqs[k].
        Tolerance: 0.5 * bin_width ≈ 3333 Hz (half a bin) is generous.
        """
        B, H, W = 2, 16, 4
        freqs = self._make_freqs(H)
        bin_width_hz = (freqs[-1] - freqs[0]) / (H - 1)
        tol = float(bin_width_hz) * 0.6  # within 60% of one bin

        # Put all energy on row 3 for batch item 0
        k = 3
        mag = torch.zeros(B, 1, H, W)
        mag[0, 0, k, :] = 100.0  # large value → near-one-hot after softmax

        out = soft_argmax_ridge(mag, freqs, temp=0.01)
        expected = float(freqs[k])
        got = float(out[0, 0])
        assert abs(got - expected) < tol, (
            f"One-hot at row {k}: expected ~{expected:.0f} Hz, got {got:.0f} Hz "
            f"(tol={tol:.0f} Hz)"
        )

    def test_soft_argmax_ridge_monotonic(self) -> None:
        """Shifting mass from a low-frequency row to a high-frequency row must
        increase the returned expected frequency."""
        B, H, W = 1, 16, 1
        freqs = self._make_freqs(H)

        # Low-frequency biased
        mag_lo = torch.zeros(B, 1, H, W)
        mag_lo[0, 0, 2, 0] = 50.0  # near bottom
        out_lo = soft_argmax_ridge(mag_lo, freqs, temp=0.1)

        # High-frequency biased
        mag_hi = torch.zeros(B, 1, H, W)
        mag_hi[0, 0, H - 3, 0] = 50.0  # near top
        out_hi = soft_argmax_ridge(mag_hi, freqs, temp=0.1)

        assert float(out_hi[0, 0]) > float(out_lo[0, 0]), (
            f"Expected out_hi ({float(out_hi[0,0]):.0f}) > out_lo ({float(out_lo[0,0]):.0f})"
        )

    def test_soft_argmax_ridge_values_in_range(self) -> None:
        """All output values must lie within [freqs.min(), freqs.max()]."""
        _seed_all()
        B, H, W = 8, 32, 64
        freqs = self._make_freqs(H)
        mag = torch.rand(B, 1, H, W)
        out = soft_argmax_ridge(mag, freqs, temp=1.0)
        fmin, fmax = float(freqs.min()), float(freqs.max())
        assert (out >= fmin - 1.0).all(), "Output below freqs.min()"
        assert (out <= fmax + 1.0).all(), "Output above freqs.max()"


# ===========================================================================
# 3. nt_xent
# ===========================================================================


class TestNtXent:
    """Verifies NT-Xent loss properties: finiteness, correctness, gradients."""

    def test_nt_xent_finite_nonneg(self) -> None:
        """NT-Xent must return a finite scalar >= 0 for valid inputs."""
        _seed_all()
        B, D = 8, 32
        z1 = F.normalize(torch.randn(B, D), dim=1)
        z2 = F.normalize(torch.randn(B, D), dim=1)
        loss = nt_xent(z1, z2, temp=0.2)
        assert loss.ndim == 0, "nt_xent must return a scalar"
        assert torch.isfinite(loss), f"nt_xent returned non-finite: {loss}"
        assert float(loss) >= 0.0, f"nt_xent returned negative: {float(loss)}"

    def test_nt_xent_aligned_lower_than_shuffled(self) -> None:
        """Perfectly aligned positive pairs must yield lower loss than shuffled pairs.

        Construction: z2 = z1 + tiny noise (near-perfect positives).
        Shuffled: z2_shuf = z1 permuted so no sample matches its pair.
        """
        _seed_all()
        B, D = 16, 64
        # Well-separated unit vectors: one-hot style embeddings with noise
        z1 = torch.zeros(B, D)
        for i in range(B):
            z1[i, i % D] = 1.0
        z1 = F.normalize(z1 + 0.01 * torch.randn(B, D), dim=1)
        z2_aligned = F.normalize(z1 + 1e-4 * torch.randn(B, D), dim=1)

        # Shuffle so all positive pairs are mismatched
        perm = torch.roll(torch.arange(B), shifts=1)  # cyclic shift — always mismatches
        z2_shuffled = z1[perm]

        loss_aligned = nt_xent(z1, z2_aligned, temp=0.2)
        loss_shuffled = nt_xent(z1, z2_shuffled, temp=0.2)
        assert float(loss_aligned) < float(loss_shuffled), (
            f"Aligned loss ({float(loss_aligned):.4f}) should be < "
            f"shuffled loss ({float(loss_shuffled):.4f})"
        )

    def test_nt_xent_gradient_flows(self) -> None:
        """Gradients must propagate back through nt_xent to both z1 and z2."""
        _seed_all()
        B, D = 8, 16
        z1 = torch.randn(B, D, requires_grad=True)
        z2 = torch.randn(B, D, requires_grad=True)
        loss = nt_xent(z1, z2, temp=0.2)
        loss.backward()
        assert z1.grad is not None, "z1.grad is None"
        assert z2.grad is not None, "z2.grad is None"
        assert torch.isfinite(z1.grad).all(), "Non-finite gradient in z1"
        assert torch.isfinite(z2.grad).all(), "Non-finite gradient in z2"

    def test_nt_xent_batch_size_1(self) -> None:
        """NT-Xent with a single pair (B=1) must not raise and return a finite scalar.

        B=1 is a boundary condition: the 2*N=2 similarity matrix has diagonals
        masked, leaving only the off-diagonal as the positive pair.
        """
        z1 = F.normalize(torch.randn(1, 16), dim=1)
        z2 = F.normalize(torch.randn(1, 16), dim=1)
        loss = nt_xent(z1, z2, temp=0.2)
        assert torch.isfinite(loss), f"B=1 produced non-finite loss: {loss}"


# ===========================================================================
# 4. latent_consistency
# ===========================================================================


class TestLatentConsistency:
    """Verifies latent_consistency: zero for identical, positive for different,
    gradient flows."""

    def test_latent_consistency_zero_identical(self) -> None:
        """latent_consistency(z, z) must be exactly 0.0 (within fp tolerance).

        mean(‖z - z‖²) = mean(0) = 0. No tolerance is needed — this is exact.
        """
        _seed_all()
        B, D = 8, 32
        z = torch.randn(B, D)
        loss = latent_consistency(z, z)
        assert torch.isfinite(loss), "Result is not finite"
        assert float(loss) == pytest.approx(0.0, abs=1e-6), (
            f"Expected 0.0 for identical inputs, got {float(loss)}"
        )

    def test_latent_consistency_positive(self) -> None:
        """latent_consistency must be > 0 for different inputs."""
        _seed_all()
        B, D = 8, 32
        z = torch.randn(B, D)
        z_aug = torch.randn(B, D)
        loss = latent_consistency(z, z_aug)
        assert float(loss) > 0.0, "Expected positive loss for different inputs"
        assert torch.isfinite(loss), "Non-finite result"

    def test_latent_consistency_gradient_flows(self) -> None:
        """Gradient must propagate back to both z and z_aug."""
        _seed_all()
        B, D = 4, 16
        z = torch.randn(B, D, requires_grad=True)
        z_aug = torch.randn(B, D, requires_grad=True)
        loss = latent_consistency(z, z_aug)
        loss.backward()
        assert z.grad is not None, "z.grad is None"
        assert z_aug.grad is not None, "z_aug.grad is None"
        assert torch.isfinite(z.grad).all()
        assert torch.isfinite(z_aug.grad).all()

    def test_latent_consistency_single_item(self) -> None:
        """Single-item batch must not raise and return the correct scalar loss."""
        z = torch.tensor([[1.0, 2.0, 3.0]])
        z_aug = torch.tensor([[4.0, 5.0, 6.0]])
        # hand computed: ‖(1-4, 2-5, 3-6)‖² = 9+9+9 = 27; mean over 1 sample = 27.0
        loss = latent_consistency(z, z_aug)
        assert float(loss) == pytest.approx(27.0, rel=1e-5), (
            f"Expected 27.0, got {float(loss)}"
        )

    def test_latent_consistency_known_value(self) -> None:
        """Hand-computed expected value for a 2-item batch.

        z   = [[0,0], [3,4]],  z_aug = [[1,1], [0,0]]
        item 0: ‖(0-1,0-1)‖² = 1+1 = 2
        item 1: ‖(3-0,4-0)‖² = 9+16 = 25
        mean = (2 + 25) / 2 = 13.5
        """
        z = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        z_aug = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        loss = latent_consistency(z, z_aug)
        assert float(loss) == pytest.approx(13.5, rel=1e-5), (
            f"Expected 13.5, got {float(loss)}"
        )


# ===========================================================================
# 5. derivative_loss
# ===========================================================================


class TestDerivativeLoss:
    """Verifies derivative_loss: zero when decoded==true, finite when not,
    mask handling, gradient flow."""

    def test_derivative_loss_zero_identical(self) -> None:
        """derivative_loss must return 0.0 when decoded == true."""
        _seed_all()
        B, W = 4, 32
        ridge = torch.randn(B, W)
        loss = derivative_loss(ridge, ridge, valid_mask=None)
        assert float(loss) == pytest.approx(0.0, abs=1e-6), (
            f"Expected 0.0 for identical inputs, got {float(loss)}"
        )

    def test_derivative_loss_positive_when_different(self) -> None:
        """derivative_loss must be positive and finite for different decoded/true."""
        _seed_all()
        B, W = 4, 32
        ridge_decoded = torch.randn(B, W)
        ridge_true = torch.randn(B, W)
        loss = derivative_loss(ridge_decoded, ridge_true, valid_mask=None)
        assert float(loss) > 0.0, "Expected positive loss for different inputs"
        assert torch.isfinite(loss), "Non-finite loss"

    def test_derivative_loss_mask_all_masked(self) -> None:
        """With all mask columns False, the loss must be 0 or a defined safe value.

        The spec says 'masking out all columns -> 0 or NaN-safe 0'. We test
        that the result is finite and equals 0.0 (NaN-safe zero policy).
        """
        B, W = 4, 16
        ridge_decoded = torch.randn(B, W)
        ridge_true = torch.randn(B, W)
        # mask length aligns with diff output: W-1 columns
        mask = torch.zeros(B, W - 1, dtype=torch.bool)
        loss = derivative_loss(ridge_decoded, ridge_true, valid_mask=mask)
        assert torch.isfinite(loss), f"All-masked loss is non-finite: {loss}"
        assert float(loss) == pytest.approx(0.0, abs=1e-6), (
            f"All-masked loss should be 0.0, got {float(loss)}"
        )

    def test_derivative_loss_partial_mask(self) -> None:
        """Partial mask: loss computed only on unmasked columns must differ from
        no-mask loss when the zeroed columns have non-zero derivative error."""
        _seed_all()
        B, W = 2, 16
        # Make decoded and true differ only in the first half of derivatives
        ridge_decoded = torch.zeros(B, W)
        ridge_true = torch.zeros(B, W)
        # Introduce derivative error only in columns 0..W//2-1
        ridge_decoded[:, : W // 2] = torch.randn(B, W // 2) * 5.0

        # Mask that includes only the second half (where no error exists)
        mask = torch.zeros(B, W - 1, dtype=torch.bool)
        mask[:, W // 2 :] = True  # second half — no derivative error here

        loss_masked = derivative_loss(ridge_decoded, ridge_true, valid_mask=mask)
        loss_unmasked = derivative_loss(ridge_decoded, ridge_true, valid_mask=None)

        # Masked loss should be close to 0 (no error in unmasked region)
        assert float(loss_masked) == pytest.approx(0.0, abs=1e-4), (
            f"Expected ~0 in masked (error-free) region, got {float(loss_masked):.6f}"
        )
        # Unmasked should be positive (captures the error region)
        assert float(loss_unmasked) > 1e-4, "Unmasked loss should be nonzero"

    def test_derivative_loss_gradient_flows(self) -> None:
        """Gradient must propagate back through derivative_loss to ridge_decoded."""
        _seed_all()
        B, W = 4, 16
        ridge_decoded = torch.randn(B, W, requires_grad=True)
        ridge_true = torch.randn(B, W)
        loss = derivative_loss(ridge_decoded, ridge_true, valid_mask=None)
        loss.backward()
        assert ridge_decoded.grad is not None, "ridge_decoded.grad is None"
        assert torch.isfinite(ridge_decoded.grad).all(), "Non-finite gradients"


# ===========================================================================
# 6. augment_pitch_time_shift
# ===========================================================================


class TestAugmentPitchTimeShift:
    """Verifies augment_pitch_time_shift output shape, in-band contract,
    non-wrapping vertical shift, and identity when max_df_hz=0, max_dt_frames=0."""

    # Use USV band from corpus constants (do not redeclare)
    BAND_LO: float = float(USV_FREQ_MIN_HZ)
    BAND_HI: float = float(USV_FREQ_MAX_HZ)
    # 100-bin frequency grid for test images: freq_per_bin = 100000/99 ≈ 1010 Hz
    H, W = 100, 64
    FREQ_PER_BIN = (BAND_HI - BAND_LO) / (H - 1)  # ≈ 1010.1 Hz

    def _make_batch(self, B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x, ridge_lo, ridge_hi) with ridges well inside the band."""
        _seed_all()
        x = torch.rand(B, 1, self.H, self.W)
        # Ridges from 40 kHz to 80 kHz — plenty of headroom on both sides
        ridge_lo = torch.full((B,), 40_000.0)
        ridge_hi = torch.full((B,), 80_000.0)
        return x, ridge_lo, ridge_hi

    def _make_cfg(self, *, max_df_hz: float = 5_000.0, max_dt_frames: int = 5) -> ShapeVAEv3Config:
        return ShapeVAEv3Config(
            max_df_hz=max_df_hz,
            max_dt_frames=max_dt_frames,
        )

    def test_augment_output_shape(self) -> None:
        """Augmented tensor must have the same shape as the input."""
        B = 4
        x, ridge_lo, ridge_hi = self._make_batch(B)
        cfg = self._make_cfg()
        x_aug, df_bins, dt_frames = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=self.FREQ_PER_BIN,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
        )
        assert x_aug.shape == x.shape, (
            f"Output shape {x_aug.shape} != input shape {x.shape}"
        )
        assert df_bins.shape == (B,), f"df_bins shape {df_bins.shape} != ({B},)"
        assert dt_frames.shape == (B,), f"dt_frames shape {dt_frames.shape} != ({B},)"

    def test_augment_in_band_guarantee(self) -> None:
        """Battery test: realized pitch shift must never push the ridge outside the band.

        Runs 20 random (ridge_lo, ridge_hi) within [25 kHz, 115 kHz] and verifies
        the in-band constraint holds for all of them.
        """
        B = 20
        cfg = self._make_cfg(max_df_hz=15_000.0, max_dt_frames=10)
        tol = 1.0  # 1 Hz numerical tolerance

        rng = np.random.default_rng(seed=7)
        ridge_lo_vals = rng.uniform(25_000.0, 60_000.0, B)
        ridge_hi_vals = ridge_lo_vals + rng.uniform(5_000.0, 40_000.0, B)
        # Clamp to fit in band
        ridge_hi_vals = np.minimum(ridge_hi_vals, 115_000.0)

        x = torch.rand(B, 1, self.H, self.W)
        ridge_lo = torch.tensor(ridge_lo_vals, dtype=torch.float32)
        ridge_hi = torch.tensor(ridge_hi_vals, dtype=torch.float32)

        g = _seed_all()
        _, df_bins, _ = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=self.FREQ_PER_BIN,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
            generator=g,
        )

        df_hz = df_bins.float() * self.FREQ_PER_BIN
        shifted_lo = ridge_lo + df_hz
        shifted_hi = ridge_hi + df_hz

        assert (shifted_lo >= self.BAND_LO - tol).all(), (
            "In-band lower constraint violated: "
            f"min shifted_lo = {float(shifted_lo.min()):.1f} Hz, band_lo = {self.BAND_LO}"
        )
        assert (shifted_hi <= self.BAND_HI + tol).all(), (
            "In-band upper constraint violated: "
            f"max shifted_hi = {float(shifted_hi.max()):.1f} Hz, band_hi = {self.BAND_HI}"
        )

    def test_augment_full_band_ridge_no_shift(self) -> None:
        """A ridge spanning the full band (lo≈band_lo, hi≈band_hi) must get df_bins==0.

        No room to shift in either direction without violating the in-band contract.
        """
        B = 4
        # Ridge pins to [band_lo, band_hi] exactly
        ridge_lo = torch.full((B,), self.BAND_LO)
        ridge_hi = torch.full((B,), self.BAND_HI)
        x = torch.rand(B, 1, self.H, self.W)
        cfg = self._make_cfg(max_df_hz=5_000.0)

        g = _seed_all()
        _, df_bins, _ = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=self.FREQ_PER_BIN,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
            generator=g,
        )
        assert (df_bins == 0).all(), (
            f"Full-band ridge must produce df_bins=0; got {df_bins.tolist()}"
        )

    def test_augment_zero_shift_unchanged(self) -> None:
        """max_df_hz=0, max_dt_frames=0 must return x unchanged (allclose)."""
        _seed_all()
        B = 4
        x, ridge_lo, ridge_hi = self._make_batch(B)
        cfg = self._make_cfg(max_df_hz=0.0, max_dt_frames=0)

        x_aug, df_bins, dt_frames = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=self.FREQ_PER_BIN,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
        )
        assert torch.allclose(x_aug, x), "Zero-shift config must leave x unchanged"
        assert (df_bins == 0).all(), f"Expected df_bins all 0, got {df_bins.tolist()}"
        assert (dt_frames == 0).all(), f"Expected dt_frames all 0, got {dt_frames.tolist()}"

    def test_augment_vertical_nonwrapping(self) -> None:
        """Vertical pitch shift must be non-wrapping (pad+slice, not roll).

        A bright pixel placed at the top row with a large upward shift (positive df)
        must NOT reappear at the bottom row.
        """
        B = 1
        x = torch.zeros(B, 1, self.H, self.W)
        # Bright pixel at top row (row 0 = highest freq in the stored image —
        # or row H-1, depending on orientation). We test BOTH rows to be safe.
        # The spec says vertical roll; we check no reappearance at the opposite end.
        x[0, 0, 0, :] = 1.0  # bright top row

        # Force a large downward shift in image space (df_bins = -large,
        # meaning we shift image content down, i.e. toward row 0) by pushing
        # ridge_hi to band_hi so only negative df (image shift UP) is allowed.
        # Simpler: force max_df_hz large but ridge pinned such that only one
        # direction is feasible. We use ridge_lo=band_lo+small to allow only
        # positive df (shift content toward higher rows), which will move the
        # bright top row off the canvas.
        ridge_lo = torch.tensor([self.BAND_LO + 1_000.0])
        ridge_hi = torch.tensor([self.BAND_LO + 20_000.0])
        cfg = ShapeVAEv3Config(max_df_hz=50_000.0, max_dt_frames=0)

        # Run many times to ensure a non-zero shift occurs
        found_nonzero_shift = False
        bottom_reappeared = False
        for seed in range(30):
            g = torch.Generator()
            g.manual_seed(seed)
            x_aug, df_bins, _ = augment_pitch_time_shift(
                x, ridge_lo, ridge_hi, cfg,
                freq_per_bin_hz=self.FREQ_PER_BIN,
                band_lo_hz=self.BAND_LO,
                band_hi_hz=self.BAND_HI,
                generator=g,
            )
            if int(df_bins[0]) != 0:
                found_nonzero_shift = True
                # Bottom row must NOT contain the bright pixel that was at top
                if float(x_aug[0, 0, -1, :].max()) > 0.5:
                    bottom_reappeared = True
                    break

        assert found_nonzero_shift, (
            "No non-zero shift occurred in 30 seeds — test setup issue"
        )
        assert not bottom_reappeared, (
            "Bright pixel reappeared at bottom row — vertical shift is wrapping, "
            "must be non-wrapping (pad+slice)"
        )

    def test_augment_deterministic_with_seed(self) -> None:
        """Same generator seed must produce identical shifts."""
        B = 8
        x, ridge_lo, ridge_hi = self._make_batch(B)
        cfg = self._make_cfg(max_df_hz=8_000.0, max_dt_frames=5)

        g1 = torch.Generator()
        g1.manual_seed(999)
        _, df1, dt1 = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=self.FREQ_PER_BIN,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
            generator=g1,
        )

        g2 = torch.Generator()
        g2.manual_seed(999)
        _, df2, dt2 = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=self.FREQ_PER_BIN,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
            generator=g2,
        )

        assert (df1 == df2).all(), "df_bins differ across same-seed runs"
        assert (dt1 == dt2).all(), "dt_frames differ across same-seed runs"

    def test_augment_single_sample(self) -> None:
        """augment_pitch_time_shift must work with batch size 1."""
        B = 1
        x = torch.rand(B, 1, self.H, self.W)
        ridge_lo = torch.tensor([50_000.0])
        ridge_hi = torch.tensor([70_000.0])
        cfg = self._make_cfg(max_df_hz=5_000.0, max_dt_frames=3)
        x_aug, df_bins, dt_frames = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=self.FREQ_PER_BIN,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
        )
        assert x_aug.shape == (1, 1, self.H, self.W)
        assert df_bins.shape == (1,)
        assert dt_frames.shape == (1,)


# ===========================================================================
# 7. annealed_weights
# ===========================================================================


class TestAnnealedWeights:
    """Verifies the staged anneal schedule for all 5 loss weights."""

    def _default_cfg(self) -> ShapeVAEv3Config:
        return ShapeVAEv3Config(
            lambda_nt=1.0,
            lambda_recon=0.05,
            beta=0.05,
            lambda_lc=1.0,
            lambda_deriv=0.1,
            recon_anneal_start=10,
            recon_anneal_epochs=20,
        )

    def test_annealed_weights_epoch_zero(self) -> None:
        """At epoch 0 (before anneal starts): recon, beta, deriv must be 0;
        nt and lc must equal their cfg values."""
        cfg = self._default_cfg()
        w = annealed_weights(cfg, epoch=0)
        assert set(w.keys()) >= {"lambda_nt", "lambda_recon", "beta", "lambda_lc", "lambda_deriv"}, (
            f"Missing keys in annealed_weights output: {w.keys()}"
        )
        assert w["lambda_recon"] == pytest.approx(0.0, abs=1e-9), (
            f"lambda_recon at epoch 0 must be 0, got {w['lambda_recon']}"
        )
        assert w["beta"] == pytest.approx(0.0, abs=1e-9), (
            f"beta at epoch 0 must be 0, got {w['beta']}"
        )
        assert w["lambda_deriv"] == pytest.approx(0.0, abs=1e-9), (
            f"lambda_deriv at epoch 0 must be 0, got {w['lambda_deriv']}"
        )
        assert w["lambda_nt"] == pytest.approx(cfg.lambda_nt, rel=1e-6), (
            f"lambda_nt at epoch 0 must equal cfg.lambda_nt={cfg.lambda_nt}"
        )
        assert w["lambda_lc"] == pytest.approx(cfg.lambda_lc, rel=1e-6), (
            f"lambda_lc at epoch 0 must equal cfg.lambda_lc={cfg.lambda_lc}"
        )

    def test_annealed_weights_full_at_end(self) -> None:
        """At epoch >= recon_anneal_start + recon_anneal_epochs, all weights
        must equal their cfg values."""
        cfg = self._default_cfg()
        end_epoch = cfg.recon_anneal_start + cfg.recon_anneal_epochs  # epoch 30

        for epoch in [end_epoch, end_epoch + 1, end_epoch + 50]:
            w = annealed_weights(cfg, epoch=epoch)
            assert w["lambda_recon"] == pytest.approx(cfg.lambda_recon, rel=1e-6), (
                f"lambda_recon not full at epoch {epoch}: {w['lambda_recon']}"
            )
            assert w["beta"] == pytest.approx(cfg.beta, rel=1e-6), (
                f"beta not full at epoch {epoch}: {w['beta']}"
            )
            assert w["lambda_deriv"] == pytest.approx(cfg.lambda_deriv, rel=1e-6), (
                f"lambda_deriv not full at epoch {epoch}: {w['lambda_deriv']}"
            )
            assert w["lambda_nt"] == pytest.approx(cfg.lambda_nt, rel=1e-6)
            assert w["lambda_lc"] == pytest.approx(cfg.lambda_lc, rel=1e-6)

    def test_annealed_weights_midpoint(self) -> None:
        """At the midpoint epoch (start + epochs//2), lambda_recon ≈ 0.5 * cfg.lambda_recon.

        The anneal is linear from 0 at start to full at start+epochs, so the
        midpoint (halfway through the ramp) must be 0.5 * full value.
        """
        cfg = self._default_cfg()
        # midpoint epoch: anneal_start + anneal_epochs // 2 = 10 + 10 = 20
        midpoint = cfg.recon_anneal_start + cfg.recon_anneal_epochs // 2
        w = annealed_weights(cfg, epoch=midpoint)
        expected_recon = 0.5 * cfg.lambda_recon
        assert w["lambda_recon"] == pytest.approx(expected_recon, rel=0.05), (
            f"At midpoint epoch {midpoint}: expected lambda_recon ≈ {expected_recon:.4f}, "
            f"got {w['lambda_recon']:.4f}"
        )
        expected_beta = 0.5 * cfg.beta
        assert w["beta"] == pytest.approx(expected_beta, rel=0.05), (
            f"At midpoint epoch {midpoint}: expected beta ≈ {expected_beta:.4f}, "
            f"got {w['beta']:.4f}"
        )

    def test_annealed_weights_nt_lc_constant(self) -> None:
        """lambda_nt and lambda_lc must be constant across all epochs."""
        cfg = self._default_cfg()
        nt_vals = []
        lc_vals = []
        for epoch in range(0, cfg.recon_anneal_start + cfg.recon_anneal_epochs + 10):
            w = annealed_weights(cfg, epoch=epoch)
            nt_vals.append(w["lambda_nt"])
            lc_vals.append(w["lambda_lc"])

        assert all(v == pytest.approx(cfg.lambda_nt, rel=1e-6) for v in nt_vals), (
            "lambda_nt changes across epochs — must be constant"
        )
        assert all(v == pytest.approx(cfg.lambda_lc, rel=1e-6) for v in lc_vals), (
            "lambda_lc changes across epochs — must be constant"
        )

    def test_annealed_weights_monotone(self) -> None:
        """lambda_recon, beta, and lambda_deriv must be monotonically non-decreasing
        across epochs during and after the anneal window."""
        cfg = self._default_cfg()
        epochs = list(range(0, cfg.recon_anneal_start + cfg.recon_anneal_epochs + 5))
        weights_over_time = [annealed_weights(cfg, epoch=e) for e in epochs]

        for key in ("lambda_recon", "beta", "lambda_deriv"):
            vals = [w[key] for w in weights_over_time]
            for i in range(1, len(vals)):
                assert vals[i] >= vals[i - 1] - 1e-9, (
                    f"{key} decreased from epoch {epochs[i-1]} ({vals[i-1]:.6f}) "
                    f"to epoch {epochs[i]} ({vals[i]:.6f})"
                )


# ===========================================================================
# 8. hybrid_loss
# ===========================================================================


def _make_tiny_model_out(
    B: int = 4,
    latent_dim: int = 8,
    H: int = 16,
    W: int = 16,
) -> dict[str, torch.Tensor]:
    """Build a synthetic model_out dict matching the hybrid_loss API.

    x_recon: sigmoid output in [0,1]  shape (B,1,H,W)
    mu:      shape (B, latent_dim)
    logvar:  shape (B, latent_dim)
    z:       reparameterized latent  shape (B, latent_dim)
    z_aug:   latent of augmented view shape (B, latent_dim)
    """
    _seed_all()
    return {
        "x_recon": torch.sigmoid(torch.randn(B, 1, H, W)),
        "mu": torch.randn(B, latent_dim),
        "logvar": torch.clamp(torch.randn(B, latent_dim), -5.0, 5.0),
        "z": torch.randn(B, latent_dim),
        "z_aug": torch.randn(B, latent_dim),
    }


class TestHybridLoss:
    """Verifies hybrid_loss assembly: finiteness, keys, term isolation, zero weights."""

    # Small image size (16x16) so ImageVAE is cheap on CPU.
    IMG = 16
    H = 16
    W = 16
    LATENT = 8
    B = 4

    def _make_inputs(self) -> tuple[Any, ...]:
        """Return (model_out, x, x_aug, ridge_true, valid_mask, freqs_hz, cfg)."""
        _seed_all()
        model_out = _make_tiny_model_out(self.B, self.LATENT, self.H, self.W)
        x = torch.rand(self.B, 1, self.H, self.W)
        x_aug = torch.rand(self.B, 1, self.H, self.W)
        ridge_true = torch.rand(self.B, self.W) * 80_000.0 + 30_000.0
        # valid_mask aligns with diff output: W-1 columns
        valid_mask = torch.ones(self.B, self.W - 1, dtype=torch.bool)
        # freqs_hz: H frequencies for soft_argmax_ridge
        freqs_hz = torch.linspace(float(USV_FREQ_MIN_HZ), float(USV_FREQ_MAX_HZ), self.H)
        cfg = ShapeVAEv3Config(
            latent_dim=self.LATENT,
            image_size=self.IMG,
        )
        return model_out, x, x_aug, ridge_true, valid_mask, freqs_hz, cfg

    def _full_weights(self, cfg: ShapeVAEv3Config) -> dict[str, float]:
        return {
            "lambda_nt": cfg.lambda_nt,
            "lambda_recon": cfg.lambda_recon,
            "beta": cfg.beta,
            "lambda_lc": cfg.lambda_lc,
            "lambda_deriv": cfg.lambda_deriv,
        }

    def test_hybrid_loss_finite(self) -> None:
        """hybrid_loss must return a finite total for valid synthetic inputs."""
        model_out, x, x_aug, ridge_true, mask, freqs_hz, cfg = self._make_inputs()
        weights = self._full_weights(cfg)
        total, _ = hybrid_loss(model_out, x, x_aug, ridge_true, mask, freqs_hz, weights, cfg)
        assert total.ndim == 0, "total must be a scalar"
        assert torch.isfinite(total), f"hybrid_loss total is non-finite: {total}"

    def test_hybrid_loss_components_keys(self) -> None:
        """Components dict must contain all 5 term keys."""
        required_keys = {"lambda_nt", "lambda_recon", "beta", "lambda_lc", "lambda_deriv"}
        model_out, x, x_aug, ridge_true, mask, freqs_hz, cfg = self._make_inputs()
        weights = self._full_weights(cfg)
        _, components = hybrid_loss(model_out, x, x_aug, ridge_true, mask, freqs_hz, weights, cfg)
        assert required_keys.issubset(components.keys()), (
            f"Missing keys: {required_keys - set(components.keys())}"
        )

    def test_hybrid_loss_term_isolation(self) -> None:
        """With only one nonzero weight, total == that_weight * unweighted_component.

        We test this for each of the 5 terms individually.
        For each term key k: set weights[k]=1.0, all others=0.0, verify
        total ≈ components[k].
        """
        model_out, x, x_aug, ridge_true, mask, freqs_hz, cfg = self._make_inputs()

        term_keys = ["lambda_nt", "lambda_recon", "beta", "lambda_lc", "lambda_deriv"]
        for key in term_keys:
            weights = {k: 0.0 for k in term_keys}
            weights[key] = 1.0
            total, components = hybrid_loss(
                model_out, x, x_aug, ridge_true, mask, freqs_hz, weights, cfg
            )
            assert torch.isfinite(total), f"Non-finite total for term {key}"
            expected = float(components[key])
            got = float(total)
            assert got == pytest.approx(expected, rel=1e-4, abs=1e-8), (
                f"Term isolation failed for {key}: "
                f"total={got:.6f}, unweighted_component={expected:.6f}"
            )

    def test_hybrid_loss_all_zero_weights(self) -> None:
        """All-zero weights must produce total = 0.0."""
        model_out, x, x_aug, ridge_true, mask, freqs_hz, cfg = self._make_inputs()
        weights = {
            "lambda_nt": 0.0,
            "lambda_recon": 0.0,
            "beta": 0.0,
            "lambda_lc": 0.0,
            "lambda_deriv": 0.0,
        }
        total, _ = hybrid_loss(model_out, x, x_aug, ridge_true, mask, freqs_hz, weights, cfg)
        assert float(total) == pytest.approx(0.0, abs=1e-8), (
            f"All-zero weights must produce total=0.0, got {float(total)}"
        )

    def test_hybrid_loss_gradients_flow_to_model_out(self) -> None:
        """Gradients must flow back from hybrid_loss through x_recon, z, and z_aug."""
        _seed_all()
        B, LATENT, H, W = self.B, self.LATENT, self.H, self.W

        x_recon = torch.sigmoid(torch.randn(B, 1, H, W, requires_grad=True))
        mu = torch.randn(B, LATENT, requires_grad=True)
        logvar = torch.clamp(torch.randn(B, LATENT), -5.0, 5.0).requires_grad_(True)
        z = torch.randn(B, LATENT, requires_grad=True)
        z_aug = torch.randn(B, LATENT, requires_grad=True)

        model_out = {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_aug": z_aug,
        }
        x = torch.rand(B, 1, H, W)
        x_aug = torch.rand(B, 1, H, W)
        ridge_true = torch.rand(B, W) * 80_000.0 + 30_000.0
        mask = torch.ones(B, W - 1, dtype=torch.bool)
        freqs_hz = torch.linspace(float(USV_FREQ_MIN_HZ), float(USV_FREQ_MAX_HZ), H)

        cfg = ShapeVAEv3Config(latent_dim=LATENT, image_size=H)
        weights = self._full_weights(cfg)

        total, _ = hybrid_loss(model_out, x, x_aug, ridge_true, mask, freqs_hz, weights, cfg)
        total.backward()

        # At least the contrastive terms should push gradients to z and z_aug
        assert z.grad is not None, "No gradient to z"
        assert z_aug.grad is not None, "No gradient to z_aug"
        assert torch.isfinite(z.grad).all(), "Non-finite gradient in z"
        assert torch.isfinite(z_aug.grad).all(), "Non-finite gradient in z_aug"
