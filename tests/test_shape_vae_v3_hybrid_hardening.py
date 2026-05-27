"""Adversarial hardening tests for scripts/experiments/train_shape_vae_v3_hybrid.py

These tests are SUPPLEMENTARY to the locked spec (test_shape_vae_v3_hybrid.py).
They focus on edge cases, boundary conditions, and numeric-stability traps that
the 50 spec tests do not exercise.

Do NOT modify the spec file. This file is append-only.
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
# Repo root on sys.path — mirrors the shim in the spec test exactly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from usv_spectrogram.corpus import USV_FREQ_MIN_HZ, USV_FREQ_MAX_HZ  # noqa: E402

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
_SEED = 1337


def _seed_all() -> torch.Generator:
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    g = torch.Generator()
    g.manual_seed(_SEED)
    return g


# ===========================================================================
# soft_argmax_ridge — adversarial inputs
# ===========================================================================


class TestSoftArgmaxRidgeAdversarial:
    """Edge cases not in the spec: all-zero input, huge magnitudes, near-tie at
    low temp, single frequency bin, and negative magnitudes."""

    def _make_freqs(self, H: int) -> torch.Tensor:
        return torch.linspace(float(USV_FREQ_MIN_HZ), float(USV_FREQ_MAX_HZ), H)

    def test_all_zero_magnitude_returns_freqs_mean(self) -> None:
        """All-zero magnitude column: softmax over zeros is uniform, so the
        expected frequency must equal freqs.mean() (within 1 Hz tolerance).

        The spec only tests random positive inputs in-range. This tests the
        degenerate case where the batch has no signal (e.g. zero-padded rows).
        """
        B, H, W = 3, 32, 16
        freqs = self._make_freqs(H)
        mag = torch.zeros(B, 1, H, W)
        out = soft_argmax_ridge(mag, freqs, temp=1.0)

        assert torch.isfinite(out).all(), "All-zero input produced NaN/Inf output"
        expected_mean = float(freqs.mean())
        for b in range(B):
            for w in range(W):
                val = float(out[b, w])
                assert abs(val - expected_mean) < 1.0, (
                    f"All-zero column: expected freqs.mean()={expected_mean:.1f} Hz, "
                    f"got {val:.1f} Hz at batch={b}, col={w}"
                )

    def test_very_large_magnitude_no_overflow(self) -> None:
        """Large magnitude values (e.g. 1e6) must not produce NaN/Inf.

        torch.softmax is max-stable (subtracts max before exp), so this should
        be safe. Verifies the implementation actually uses F.softmax.
        """
        _seed_all()
        B, H, W = 4, 16, 8
        freqs = self._make_freqs(H)
        # All large values pointing at one row
        mag = torch.zeros(B, 1, H, W)
        mag[:, :, H // 2, :] = 1e6
        out = soft_argmax_ridge(mag, freqs, temp=1.0)

        assert torch.isfinite(out).all(), (
            "Very large magnitude values produced NaN/Inf — softmax is not max-stable"
        )
        # All mass near row H//2
        expected_freq = float(freqs[H // 2])
        assert (out - expected_freq).abs().max() < 100.0, (
            f"Large-magnitude peak at row {H//2} ({expected_freq:.0f} Hz): "
            f"output far from peak, max deviation={float((out - expected_freq).abs().max()):.1f} Hz"
        )

    def test_low_temp_near_tie_two_adjacent_bins(self) -> None:
        """With temp=1e-3 and a near-tie between two adjacent rows, the result
        must lie between the two bin frequencies and be finite.

        At very low temperature the softmax becomes near-argmax. A perfect tie
        should land at the midpoint; slight asymmetry biases it toward the
        higher-energy bin but always within the two adjacent frequencies.
        """
        H, W = 32, 4
        freqs = self._make_freqs(H)
        k = 12  # bin to tie

        B = 1
        mag = torch.zeros(B, 1, H, W)
        # Equal mass on bins k and k+1 (a perfect tie)
        mag[0, 0, k, :] = 10.0
        mag[0, 0, k + 1, :] = 10.0

        out = soft_argmax_ridge(mag, freqs, temp=1e-3)

        assert torch.isfinite(out).all(), "Near-tie at low temp produced NaN/Inf"
        lo = float(freqs[k])
        hi = float(freqs[k + 1])
        for w in range(W):
            val = float(out[0, w])
            assert lo - 1.0 <= val <= hi + 1.0, (
                f"Near-tie at low temp: result {val:.1f} Hz outside "
                f"[{lo:.1f}, {hi:.1f}] Hz"
            )

    def test_single_frequency_bin_returns_that_freq(self) -> None:
        """H=1 (single frequency bin): output must equal that one frequency for
        every column, regardless of magnitude or temp.

        When there is only one bin, softmax({x/temp}) = [1.0], so the expected
        frequency is always freqs[0].
        """
        B, W = 4, 8
        H = 1
        freqs = torch.tensor([55_000.0])  # arbitrary single freq
        mag = torch.rand(B, 1, H, W) * 100.0  # random values

        out = soft_argmax_ridge(mag, freqs, temp=1.0)

        assert out.shape == (B, W), f"Expected ({B}, {W}), got {out.shape}"
        assert torch.isfinite(out).all(), "Single-bin input produced NaN/Inf"
        expected = float(freqs[0])
        assert (out - expected).abs().max() < 0.1, (
            f"Single-bin: expected constant {expected:.1f} Hz, "
            f"max deviation={float((out - expected).abs().max()):.4f} Hz"
        )

    def test_negative_magnitude_values_stay_in_range(self) -> None:
        """Negative magnitude values: the function does not promise non-negativity
        in its inputs. softmax weights are always in [0, 1] (a simplex), so the
        convex combination remains within [freqs.min(), freqs.max()].

        This tests a realistic failure mode: upstream normalization can leave
        negative values in the log-scale power.
        """
        _seed_all()
        B, H, W = 4, 16, 12
        freqs = self._make_freqs(H)
        fmin, fmax = float(freqs.min()), float(freqs.max())

        mag = torch.randn(B, 1, H, W) * 50.0  # freely negative

        out = soft_argmax_ridge(mag, freqs, temp=1.0)

        assert torch.isfinite(out).all(), "Negative magnitude produced NaN/Inf"
        assert (out >= fmin - 1.0).all(), (
            f"Output {float(out.min()):.1f} Hz below freqs.min()={fmin:.1f} Hz "
            "with negative magnitude input"
        )
        assert (out <= fmax + 1.0).all(), (
            f"Output {float(out.max()):.1f} Hz above freqs.max()={fmax:.1f} Hz "
            "with negative magnitude input"
        )

    def test_wrong_ndim_raises_value_error(self) -> None:
        """A 2D tensor (H, W) should raise ValueError, not crash silently."""
        freqs = torch.linspace(float(USV_FREQ_MIN_HZ), float(USV_FREQ_MAX_HZ), 16)
        mag_2d = torch.rand(16, 8)  # missing batch dim
        with pytest.raises(ValueError):
            soft_argmax_ridge(mag_2d, freqs, temp=1.0)


# ===========================================================================
# derivative_loss — edge cases
# ===========================================================================


class TestDerivativeLossAdversarial:
    """Edge cases: wrong-length mask, single-True mask, constant-offset invariance."""

    def test_constant_offset_gives_zero_loss(self) -> None:
        """If decoded ridge = true ridge + constant, dF/dt is identical in both,
        so derivative_loss must be 0.

        This is the whole algebraic point: the derivative loss is invariant to
        absolute pitch offset. Tests the property that constant-offset pairs
        vanish.
        """
        _seed_all()
        B, W = 4, 32
        ridge_true = torch.randn(B, W) * 5000.0 + 50_000.0  # realistic USV freqs

        # Add a different constant offset to each batch item
        offset = torch.tensor([100.0, 500.0, -200.0, 1000.0]).unsqueeze(1)  # (B, 1)
        ridge_decoded = ridge_true + offset

        loss = derivative_loss(ridge_decoded, ridge_true, valid_mask=None)

        assert torch.isfinite(loss), "Constant-offset loss is non-finite"
        assert float(loss) == pytest.approx(0.0, abs=1e-4), (
            f"Constant offset between decoded and true should give derivative_loss=0, "
            f"got {float(loss):.6e}"
        )

    def test_single_true_column_in_mask(self) -> None:
        """valid_mask with exactly one True column: loss must be computed from
        that one diff element and be finite.

        This tests the denom+1e-8 guard under near-zero denominator (but not
        exactly zero as in the all-False case).
        """
        B, W = 4, 16
        ridge_decoded = torch.randn(B, W)
        ridge_true = torch.randn(B, W)

        mask = torch.zeros(B, W - 1, dtype=torch.bool)
        mask[:, 0] = True  # only the first diff column

        loss = derivative_loss(ridge_decoded, ridge_true, valid_mask=mask)

        assert torch.isfinite(loss), f"Single-True mask produced non-finite: {loss}"
        assert float(loss) >= 0.0, f"derivative_loss must be non-negative, got {float(loss)}"

        # Cross-check: manually compute the expected value
        d_dec = torch.diff(ridge_decoded, dim=-1)  # (B, W-1)
        d_true = torch.diff(ridge_true, dim=-1)
        sq = (d_dec - d_true) ** 2
        # Only mask[:, 0] is True → sum of sq[:, 0], divided by B (float mask sum = B)
        expected = float(sq[:, 0].sum() / B)
        assert float(loss) == pytest.approx(expected, rel=1e-4), (
            f"Single-True mask: expected {expected:.6f}, got {float(loss):.6f}"
        )

    def test_mask_wrong_length_column_count(self) -> None:
        """A valid_mask of shape (B, W) instead of (B, W-1) should raise or
        produce a broadcast error — it must NOT silently give wrong results.

        The spec says mask is '(B, W-1) boolean'. A (B, W) mask has an extra
        column and should not produce a quietly-wrong answer.
        """
        B, W = 2, 8
        ridge_decoded = torch.randn(B, W)
        ridge_true = torch.randn(B, W)
        # Wrong shape: W instead of W-1
        bad_mask = torch.ones(B, W, dtype=torch.bool)

        # The implementation multiplies sq (B, W-1) by m (B, W) — shapes don't
        # broadcast, so this must raise a RuntimeError.
        with pytest.raises((RuntimeError, ValueError)):
            derivative_loss(ridge_decoded, ridge_true, valid_mask=bad_mask)


# ===========================================================================
# latent_consistency — edge cases
# ===========================================================================


class TestLatentConsistencyAdversarial:
    """Large-magnitude latents (overflow check) and D=1 latent."""

    def test_large_magnitude_latents_finite(self) -> None:
        """Latents with magnitude 1e4 must not overflow: ‖z - z_aug‖² can be
        large but must remain finite (no Inf, no NaN).
        """
        B, D = 8, 32
        z = torch.full((B, D), 1e4)
        z_aug = torch.full((B, D), -1e4)
        loss = latent_consistency(z, z_aug)
        assert torch.isfinite(loss), (
            f"Large-magnitude latents produced non-finite consistency loss: {loss}"
        )
        # Expected: ‖z - z_aug‖² = (2e4)² * D = 4e8 * D per sample, but finite
        assert float(loss) > 0.0, "Expected positive loss for different latents"

    def test_d1_latent(self) -> None:
        """D=1 latent (scalar per sample): must return the correct MSE scalar."""
        z = torch.tensor([[2.0], [5.0]])       # (2, 1)
        z_aug = torch.tensor([[0.0], [3.0]])   # (2, 1)
        # item 0: (2-0)² = 4; item 1: (5-3)² = 4; mean = 4.0
        loss = latent_consistency(z, z_aug)
        assert torch.isfinite(loss), "D=1 latent produced non-finite result"
        assert float(loss) == pytest.approx(4.0, rel=1e-5), (
            f"D=1 latent: expected 4.0, got {float(loss)}"
        )


# ===========================================================================
# nt_xent — adversarial inputs
# ===========================================================================


class TestNtXentAdversarial:
    """Very small temperature (sharpness) and z1 == z2 edge case."""

    def test_very_small_temp_finite_nonneg(self) -> None:
        """NT-Xent with temp=1e-4 (very sharp) must remain finite and >= 0.

        With small temp the logits blow up; cross_entropy must still be
        numerically stable because softmax subtracts max internally.
        """
        _seed_all()
        B, D = 8, 32
        z1 = F.normalize(torch.randn(B, D), dim=1)
        z2 = F.normalize(torch.randn(B, D), dim=1)
        loss = nt_xent(z1, z2, temp=1e-4)
        assert torch.isfinite(loss), f"Very small temp produced non-finite NT-Xent: {loss}"
        assert float(loss) >= 0.0, f"NT-Xent must be >= 0, got {float(loss)}"

    def test_identical_z1_z2_finite(self) -> None:
        """When z1 == z2 (every sample is its own hardest negative off-diagonal),
        the loss must still be finite and non-negative.

        This happens during early training or when augmentation is a no-op.
        The diagonal is masked out, leaving the off-diagonal as hard negatives.
        """
        _seed_all()
        B, D = 8, 32
        z = F.normalize(torch.randn(B, D), dim=1)
        loss = nt_xent(z, z, temp=0.2)
        assert torch.isfinite(loss), (
            f"z1==z2 produced non-finite NT-Xent: {loss}"
        )
        assert float(loss) >= 0.0, f"NT-Xent must be >= 0 for z1==z2, got {float(loss)}"


# ===========================================================================
# augment_pitch_time_shift — stress tests
# ===========================================================================


class TestAugmentAdversarial:
    """Stress tests not covered by the spec: edge-of-band ridges, huge max_df_hz,
    large dt_frames vs image width, and sign constraint for ridge near band_hi."""

    BAND_LO: float = float(USV_FREQ_MIN_HZ)   # 20 000 Hz
    BAND_HI: float = float(USV_FREQ_MAX_HZ)   # 120 000 Hz
    H, W = 100, 64
    FREQ_PER_BIN: float = (float(USV_FREQ_MAX_HZ) - float(USV_FREQ_MIN_HZ)) / (100 - 1)

    def test_ridge_exactly_one_bin_from_each_edge_no_violation(self) -> None:
        """Ridge placed exactly 0.5 * freq_per_bin from each band edge:
        the per-sample clamp must allow at most df_bins=0 and never violate.

        This is the tightest possible ridge that still has theoretical headroom
        of half a bin on each side. The integer ceiling/floor rounding must not
        round it outside the band.
        """
        fpb = self.FREQ_PER_BIN
        # ridge_lo = band_lo + 0.5*fpb  →  only upshift possible is at most 0
        # ridge_hi = band_hi - 0.5*fpb  →  only downshift possible is at most 0
        ridge_lo_val = self.BAND_LO + 0.5 * fpb
        ridge_hi_val = self.BAND_HI - 0.5 * fpb

        B = 4
        x = torch.ones(B, 1, self.H, self.W)
        ridge_lo = torch.full((B,), ridge_lo_val)
        ridge_hi = torch.full((B,), ridge_hi_val)
        cfg = ShapeVAEv3Config(max_df_hz=5_000.0, max_dt_frames=0)

        g = torch.Generator()
        g.manual_seed(0)
        _, df_bins, _ = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=fpb,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
            generator=g,
        )

        tol = 1.0  # 1 Hz
        df_hz = df_bins.float() * fpb
        shifted_lo = ridge_lo + df_hz
        shifted_hi = ridge_hi + df_hz

        assert (shifted_lo >= self.BAND_LO - tol).all(), (
            f"Edge-of-band ridge: shifted_lo {float(shifted_lo.min()):.1f} < band_lo {self.BAND_LO}"
        )
        assert (shifted_hi <= self.BAND_HI + tol).all(), (
            f"Edge-of-band ridge: shifted_hi {float(shifted_hi.max()):.1f} > band_hi {self.BAND_HI}"
        )

    def test_huge_max_df_hz_in_band_guarantee(self) -> None:
        """max_df_hz=200_000 Hz (larger than the whole 100 kHz band) must STILL
        keep every realized shift in-band via per-sample clamping.

        A naive implementation that draws from [-max_df_bins, +max_df_bins]
        without the per-sample ceiling/floor clamp would violate this.

        Battery: 50 random ridges, checked exhaustively.
        """
        rng = np.random.default_rng(seed=99)
        B = 50
        fpb = self.FREQ_PER_BIN

        ridge_lo_vals = rng.uniform(self.BAND_LO + fpb, self.BAND_LO + 40_000.0, B)
        ridge_hi_vals = ridge_lo_vals + rng.uniform(5_000.0, 30_000.0, B)
        ridge_hi_vals = np.minimum(ridge_hi_vals, self.BAND_HI - fpb)

        x = torch.rand(B, 1, self.H, self.W)
        ridge_lo = torch.tensor(ridge_lo_vals, dtype=torch.float32)
        ridge_hi = torch.tensor(ridge_hi_vals, dtype=torch.float32)

        cfg = ShapeVAEv3Config(max_df_hz=200_000.0, max_dt_frames=0)

        g = torch.Generator()
        g.manual_seed(7)
        _, df_bins, _ = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=fpb,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
            generator=g,
        )

        tol = 1.0  # 1 Hz
        df_hz = df_bins.float() * fpb
        shifted_lo = ridge_lo + df_hz
        shifted_hi = ridge_hi + df_hz

        violations_lo = (shifted_lo < self.BAND_LO - tol).sum().item()
        violations_hi = (shifted_hi > self.BAND_HI + tol).sum().item()

        assert violations_lo == 0, (
            f"huge max_df_hz: {violations_lo} sample(s) violated lower band edge. "
            f"Min shifted_lo={float(shifted_lo.min()):.1f} Hz"
        )
        assert violations_hi == 0, (
            f"huge max_df_hz: {violations_hi} sample(s) violated upper band edge. "
            f"Max shifted_hi={float(shifted_hi.max()):.1f} Hz"
        )

    def test_max_dt_frames_exceeds_image_width_no_crash(self) -> None:
        """max_dt_frames > W must not crash; output shape must be preserved and
        be finite (content can be fully shifted off-canvas → all zeros, acceptable).
        """
        B = 4
        W_small = 8   # image width smaller than max_dt_frames
        H = self.H
        x = torch.rand(B, 1, H, W_small)
        ridge_lo = torch.full((B,), 50_000.0)
        ridge_hi = torch.full((B,), 70_000.0)

        cfg = ShapeVAEv3Config(max_df_hz=0.0, max_dt_frames=100)  # >> W_small

        g = torch.Generator()
        g.manual_seed(42)
        x_aug, df_bins, dt_frames = augment_pitch_time_shift(
            x, ridge_lo, ridge_hi, cfg,
            freq_per_bin_hz=self.FREQ_PER_BIN,
            band_lo_hz=self.BAND_LO,
            band_hi_hz=self.BAND_HI,
            generator=g,
        )

        assert x_aug.shape == (B, 1, H, W_small), (
            f"Shape changed: expected {(B, 1, H, W_small)}, got {x_aug.shape}"
        )
        assert torch.isfinite(x_aug).all(), (
            "Non-finite values in output when dt > image width"
        )
        # At least one sample should have been shifted beyond canvas (all zeros)
        assert (x_aug >= 0.0).all(), "Negative values in augmented output (should be 0-filled)"

    def test_ridge_near_band_hi_only_negative_df_allowed(self) -> None:
        """A ridge with hi at band_hi forces all df_bins <= 0 across many seeds.

        This tests the direction constraint: if the ridge ceiling touches the
        band ceiling, no positive pitch shift is feasible (it would push the
        ridge above band_hi).
        """
        B = 1
        fpb = self.FREQ_PER_BIN
        # Place ridge_hi exactly at band_hi — no room to shift up at all
        ridge_lo_val = self.BAND_HI - 20_000.0
        ridge_hi_val = self.BAND_HI  # ceiling

        x = torch.rand(B, 1, self.H, self.W)
        ridge_lo = torch.tensor([ridge_lo_val])
        ridge_hi = torch.tensor([ridge_hi_val])
        cfg = ShapeVAEv3Config(max_df_hz=10_000.0, max_dt_frames=0)

        positive_shift_seen = False
        for seed in range(50):
            g = torch.Generator()
            g.manual_seed(seed)
            _, df_bins, _ = augment_pitch_time_shift(
                x, ridge_lo, ridge_hi, cfg,
                freq_per_bin_hz=fpb,
                band_lo_hz=self.BAND_LO,
                band_hi_hz=self.BAND_HI,
                generator=g,
            )
            if int(df_bins[0]) > 0:
                positive_shift_seen = True
                break

        assert not positive_shift_seen, (
            "A positive df_bins was produced even though ridge_hi == band_hi "
            "— the in-band ceiling clamp is broken"
        )


# ===========================================================================
# annealed_weights — boundary and sign edge cases
# ===========================================================================


class TestAnnealedWeightsAdversarial:
    """recon_anneal_start=0, negative epoch, exact boundaries."""

    def _make_cfg(self, anneal_start: int = 0, anneal_epochs: int = 10) -> ShapeVAEv3Config:
        return ShapeVAEv3Config(
            recon_anneal_start=anneal_start,
            recon_anneal_epochs=anneal_epochs,
        )

    def test_anneal_start_zero_epoch_zero_frac_zero(self) -> None:
        """recon_anneal_start=0, epoch=0: frac = (0-0)/epochs = 0 → annealed
        weights must be 0 for recon/beta/deriv at epoch 0.
        """
        cfg = self._make_cfg(anneal_start=0, anneal_epochs=10)
        w = annealed_weights(cfg, epoch=0)
        assert w["lambda_recon"] == pytest.approx(0.0, abs=1e-9), (
            f"anneal_start=0, epoch=0: lambda_recon should be 0, got {w['lambda_recon']}"
        )
        assert w["beta"] == pytest.approx(0.0, abs=1e-9), (
            f"anneal_start=0, epoch=0: beta should be 0, got {w['beta']}"
        )
        assert w["lambda_deriv"] == pytest.approx(0.0, abs=1e-9), (
            f"anneal_start=0, epoch=0: lambda_deriv should be 0, got {w['lambda_deriv']}"
        )

    def test_anneal_start_zero_epoch_after_start_full(self) -> None:
        """recon_anneal_start=0: at epoch == recon_anneal_epochs, all weights
        must equal their full cfg values (frac clamped to 1.0).
        """
        cfg = self._make_cfg(anneal_start=0, anneal_epochs=10)
        w = annealed_weights(cfg, epoch=cfg.recon_anneal_epochs)
        assert w["lambda_recon"] == pytest.approx(cfg.lambda_recon, rel=1e-6), (
            f"anneal_start=0, epoch=anneal_epochs: lambda_recon not full: {w['lambda_recon']}"
        )
        assert w["beta"] == pytest.approx(cfg.beta, rel=1e-6)
        assert w["lambda_deriv"] == pytest.approx(cfg.lambda_deriv, rel=1e-6)

    def test_negative_epoch_clamps_to_zero_frac(self) -> None:
        """A negative epoch (e.g. epoch=-5) must be clamped to frac=0.0 —
        recon/beta/deriv must be 0, not negative.
        """
        cfg = ShapeVAEv3Config(recon_anneal_start=10, recon_anneal_epochs=20)
        # epoch=-5 → frac = (-5 - 10) / 20 = -0.75, clamped to 0.0
        w = annealed_weights(cfg, epoch=-5)
        assert w["lambda_recon"] == pytest.approx(0.0, abs=1e-9), (
            f"Negative epoch should give lambda_recon=0, got {w['lambda_recon']}"
        )
        assert w["beta"] == pytest.approx(0.0, abs=1e-9), (
            f"Negative epoch should give beta=0, got {w['beta']}"
        )
        assert w["lambda_deriv"] == pytest.approx(0.0, abs=1e-9), (
            f"Negative epoch should give lambda_deriv=0, got {w['lambda_deriv']}"
        )

    def test_epoch_exactly_at_anneal_start_frac_zero(self) -> None:
        """Epoch == recon_anneal_start: frac = (start - start) / epochs = 0,
        so annealed terms must all be 0.
        """
        cfg = ShapeVAEv3Config(recon_anneal_start=15, recon_anneal_epochs=20)
        w = annealed_weights(cfg, epoch=cfg.recon_anneal_start)
        assert w["lambda_recon"] == pytest.approx(0.0, abs=1e-9), (
            f"epoch==anneal_start: lambda_recon should be 0, got {w['lambda_recon']}"
        )
        assert w["beta"] == pytest.approx(0.0, abs=1e-9)
        assert w["lambda_deriv"] == pytest.approx(0.0, abs=1e-9)

    def test_epoch_exactly_at_start_plus_epochs_frac_one(self) -> None:
        """Epoch == recon_anneal_start + recon_anneal_epochs: frac = 1.0 exactly."""
        cfg = ShapeVAEv3Config(recon_anneal_start=15, recon_anneal_epochs=20)
        end_epoch = cfg.recon_anneal_start + cfg.recon_anneal_epochs
        w = annealed_weights(cfg, epoch=end_epoch)
        assert w["lambda_recon"] == pytest.approx(cfg.lambda_recon, rel=1e-6), (
            f"epoch==start+epochs: lambda_recon not full: {w['lambda_recon']}"
        )
        assert w["beta"] == pytest.approx(cfg.beta, rel=1e-6)
        assert w["lambda_deriv"] == pytest.approx(cfg.lambda_deriv, rel=1e-6)


# ===========================================================================
# hybrid_loss — all-False mask and extreme logvar
# ===========================================================================


class TestHybridLossAdversarial:
    """valid_mask all-False (derivative term must not NaN total) and extreme
    logvar at the clamp boundary."""

    IMG = 16
    H = 16
    W = 16
    LATENT = 8
    B = 4

    def _base_model_out(
        self,
        logvar_val: float = 0.0,
    ) -> dict:
        _seed_all()
        return {
            "x_recon": torch.sigmoid(torch.randn(self.B, 1, self.H, self.W)),
            "mu": torch.randn(self.B, self.LATENT),
            "logvar": torch.full((self.B, self.LATENT), logvar_val),
            "z": torch.randn(self.B, self.LATENT),
            "z_aug": torch.randn(self.B, self.LATENT),
        }

    def _full_weights(self, cfg: ShapeVAEv3Config) -> dict:
        return {
            "lambda_nt": cfg.lambda_nt,
            "lambda_recon": cfg.lambda_recon,
            "beta": cfg.beta,
            "lambda_lc": cfg.lambda_lc,
            "lambda_deriv": cfg.lambda_deriv,
        }

    def test_all_false_valid_mask_total_finite(self) -> None:
        """hybrid_loss with all-False valid_mask must return a finite total.

        The derivative term collapses to 0 (NaN-safe zero from denom+1e-8).
        The other four terms must carry the total — finite overall.
        """
        _seed_all()
        model_out = self._base_model_out()
        x = torch.rand(self.B, 1, self.H, self.W)
        x_aug = torch.rand(self.B, 1, self.H, self.W)
        ridge_true = torch.rand(self.B, self.W) * 80_000.0 + 30_000.0

        all_false_mask = torch.zeros(self.B, self.W - 1, dtype=torch.bool)
        freqs_hz = torch.linspace(float(USV_FREQ_MIN_HZ), float(USV_FREQ_MAX_HZ), self.H)
        cfg = ShapeVAEv3Config(latent_dim=self.LATENT, image_size=self.IMG)
        weights = self._full_weights(cfg)

        total, components = hybrid_loss(
            model_out, x, x_aug, ridge_true, all_false_mask, freqs_hz, weights, cfg
        )

        assert torch.isfinite(total), (
            f"hybrid_loss with all-False mask produced non-finite total: {total}"
        )
        # Derivative component must be 0 (NaN-safe)
        deriv_val = float(components["lambda_deriv"])
        assert math.isfinite(deriv_val), (
            f"derivative component is non-finite with all-False mask: {deriv_val}"
        )
        assert abs(deriv_val) < 1e-6, (
            f"derivative component should be ~0 with all-False mask, got {deriv_val}"
        )

    def test_extreme_logvar_clamp_boundary_kl_finite(self) -> None:
        """logvar = +10 (clamp boundary in image_vae_loss, simulated): KL term
        must remain finite.

        image_vae_loss uses nan_to_num+clamp on x_recon, but KL involves
        exp(logvar). At logvar=10, exp(10) ≈ 22026. KL = -0.5*(1 + 10 - mu² - e^10).
        This must be large but finite, so hybrid_loss total must be finite.
        """
        _seed_all()
        # logvar=10 is extreme but finite
        model_out = self._base_model_out(logvar_val=10.0)
        x = torch.rand(self.B, 1, self.H, self.W)
        x_aug = torch.rand(self.B, 1, self.H, self.W)
        ridge_true = torch.rand(self.B, self.W) * 80_000.0 + 30_000.0
        mask = torch.ones(self.B, self.W - 1, dtype=torch.bool)
        freqs_hz = torch.linspace(float(USV_FREQ_MIN_HZ), float(USV_FREQ_MAX_HZ), self.H)
        cfg = ShapeVAEv3Config(latent_dim=self.LATENT, image_size=self.IMG)
        weights = self._full_weights(cfg)

        total, components = hybrid_loss(
            model_out, x, x_aug, ridge_true, mask, freqs_hz, weights, cfg
        )

        assert torch.isfinite(total), (
            f"Extreme logvar=10 produced non-finite hybrid_loss total: {total}"
        )
        kl_val = float(components["beta"])
        assert math.isfinite(kl_val), (
            f"KL component non-finite at logvar=10: {kl_val}"
        )
        assert kl_val > 0.0, (
            f"KL at logvar=10 should be positive (strong penalty), got {kl_val}"
        )

    def test_none_valid_mask_vs_all_true_mask_equivalent(self) -> None:
        """hybrid_loss with valid_mask=None must give the same derivative
        component as with an all-True mask of the correct shape.

        Both code paths in derivative_loss must produce identical results.
        """
        _seed_all()
        model_out = self._base_model_out()
        x = torch.rand(self.B, 1, self.H, self.W)
        x_aug = torch.rand(self.B, 1, self.H, self.W)
        ridge_true = torch.rand(self.B, self.W) * 80_000.0 + 30_000.0
        freqs_hz = torch.linspace(float(USV_FREQ_MIN_HZ), float(USV_FREQ_MAX_HZ), self.H)
        cfg = ShapeVAEv3Config(latent_dim=self.LATENT, image_size=self.IMG)
        weights = self._full_weights(cfg)

        # Run with None mask
        _seed_all()
        model_out_copy = {k: v.clone() for k, v in model_out.items()}
        total_none, comp_none = hybrid_loss(
            model_out_copy, x.clone(), x_aug.clone(), ridge_true.clone(),
            None, freqs_hz, weights, cfg
        )

        # Run with all-True mask
        _seed_all()
        model_out_copy2 = {k: v.clone() for k, v in model_out.items()}
        all_true_mask = torch.ones(self.B, self.W - 1, dtype=torch.bool)
        total_true, comp_true = hybrid_loss(
            model_out_copy2, x.clone(), x_aug.clone(), ridge_true.clone(),
            all_true_mask, freqs_hz, weights, cfg
        )

        deriv_none = float(comp_none["lambda_deriv"])
        deriv_true = float(comp_true["lambda_deriv"])
        assert abs(deriv_none - deriv_true) < 1e-4, (
            f"derivative component: None mask gave {deriv_none:.6f}, "
            f"all-True mask gave {deriv_true:.6f} — should be equivalent"
        )
