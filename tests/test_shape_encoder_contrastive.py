"""Tests for scripts/experiments/train_shape_encoder_contrastive.py

Pre-implementation spec tests for Pathway-B contrastive shape encoder (SimCLR-style).
Written by test-architect BEFORE any implementation exists. All tests are expected to
fail with ImportError or ModuleNotFoundError until the module is created.

ROADMAP test plan coverage:
  1. nt_xent_loss returns scalar, finite, >0          -> test_nt_xent_loss_returns_finite_scalar
  2. nt_xent_loss symmetric                           -> test_nt_xent_loss_symmetric
  3. nt_xent_loss collapse case -> log(2B-1)          -> test_nt_xent_loss_collapse_equals_log2Bm1
  4. nt_xent_loss separable < collapse                -> test_nt_xent_loss_separable_less_than_collapse
  5. nt_xent_loss temperature monotonicity            -> test_nt_xent_loss_temperature_monotonicity
  6. nt_xent_loss gradient flows through z1           -> test_nt_xent_loss_gradient_nonzero
  7. augment shape preserved                          -> test_augment_shape_preserved
  8. augment zero input -> zero output                -> test_augment_zero_input_zero_output
  9. augment determinism (same seed -> identical)     -> test_augment_deterministic_with_seed
  10. augment in-band guarantee (no off-edge wrap)    -> test_augment_inband_no_wraparound
  11. augment pair differs for different seeds        -> test_augment_two_views_differ
  12. augment warp_lo=warp_hi=1, no shift -> identity -> test_augment_identity_no_op
  13. ContrastiveEncoder forward shape (prod size)    -> test_encoder_forward_shape_prod_size
  14. ContrastiveEncoder forward shape (small input)  -> test_encoder_forward_shape_small_input

Additional coverage (recurring gap patterns):
  - Single-batch edge case (B=1 in augment)          -> test_augment_single_sample_shape
  - Batch consistency (B>1 augment is per-sample)    -> test_augment_per_sample_independence
  - nt_xent_loss B=1 edge case                       -> test_nt_xent_loss_single_pair
  - Projection head dim separate from embed_dim      -> test_encoder_embed_and_proj_dims_independent
  - Encoder outputs are finite for all-zero input    -> test_encoder_finite_on_zero_input
  - Gradient flows through encoder projection head   -> test_encoder_projection_gradient_flows

Total: 20 tests (14 from ROADMAP, 6 additional)
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Module loading via importlib (scripts/experiments/ is not an installed pkg)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = REPO_ROOT / "scripts" / "experiments" / "train_shape_encoder_contrastive.py"

_MODULE: ModuleType | None = None
_IMPORT_ERROR: str | None = None

if _MODULE_PATH.exists():
    try:
        spec = importlib.util.spec_from_file_location(
            "train_shape_encoder_contrastive", _MODULE_PATH
        )
        _MODULE = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules["train_shape_encoder_contrastive"] = _MODULE
        spec.loader.exec_module(_MODULE)  # type: ignore[union-attr]
    except Exception as exc:
        _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = (
        f"Module file not found: {_MODULE_PATH}  "
        "(expected after implementation)"
    )


def _require_module() -> ModuleType:
    """Return the loaded module or raise a clear skip/error."""
    if _MODULE is None:
        pytest.skip(
            f"train_shape_encoder_contrastive not yet implemented: {_IMPORT_ERROR}"
        )
    return _MODULE  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helper: build near-orthonormal rows  (B distinct directions in R^D)
# ---------------------------------------------------------------------------

def _orthonormal_rows(B: int, D: int, *, seed: int = 0) -> torch.Tensor:
    """Return (B, D) tensor with approximately orthonormal rows (B <= D)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    z = torch.randn(B, D, generator=rng)
    # QR decomposition gives exact orthonormal rows when B <= D
    q, _ = torch.linalg.qr(z.T)
    return q.T[:B].contiguous()  # (B, D)


# ===========================================================================
# nt_xent_loss tests
# ===========================================================================


class TestNtXentLoss:
    """Tests for the nt_xent_loss(z1, z2, tau) function."""

    def test_nt_xent_loss_returns_finite_scalar(self):
        """Spec: returns a 0-dim finite positive tensor for generic input."""
        mod = _require_module()
        torch.manual_seed(42)
        z1 = torch.randn(8, 16)
        z2 = torch.randn(8, 16)
        loss = mod.nt_xent_loss(z1, z2, tau=0.2)

        assert loss.ndim == 0, f"Expected scalar (0-dim), got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"

    def test_nt_xent_loss_symmetric(self):
        """Spec: nt_xent_loss(z1,z2) approx equals nt_xent_loss(z2,z1)."""
        mod = _require_module()
        torch.manual_seed(7)
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)
        loss_fwd = mod.nt_xent_loss(z1, z2, tau=0.2)
        loss_bwd = mod.nt_xent_loss(z2, z1, tau=0.2)

        assert torch.allclose(loss_fwd, loss_bwd, atol=1e-5), (
            f"Asymmetry: nt_xent(z1,z2)={loss_fwd.item():.6f} "
            f"!= nt_xent(z2,z1)={loss_bwd.item():.6f}"
        )

    def test_nt_xent_loss_collapse_equals_log2Bm1(self):
        """Spec: collapse case z1=z2=all-ones -> loss ≈ log(2B-1).

        Derivation: after L2-normalisation all rows are identical (1/sqrt(D) each
        dimension). Therefore every dot product is 1.0 / tau. The diagonal is
        masked to -inf. Each row has 2B-1 remaining entries all equal to 1/tau.
        The softmax over those 2B-1 equal logits is uniform (1/(2B-1)). The
        cross-entropy target is exactly one of those 2B-1 entries, so
        CE = log(2B-1).
        """
        mod = _require_module()
        B, D = 6, 16
        # All-ones: after L2 normalise, all rows are identical => collapse
        z = torch.ones(B, D)
        loss = mod.nt_xent_loss(z, z.clone(), tau=0.2)

        expected = math.log(2 * B - 1)
        assert abs(loss.item() - expected) < 1e-3, (
            f"Collapse loss {loss.item():.6f} != log(2B-1)={expected:.6f} "
            f"(B={B})"
        )

    def test_nt_xent_loss_separable_less_than_collapse(self):
        """Spec: well-separated (near-orthonormal) rows -> loss < collapse loss."""
        mod = _require_module()
        B, D = 6, 32
        # Near-orthonormal rows: positives are maximally separated from negatives
        z = _orthonormal_rows(B, D, seed=3)

        loss_sep = mod.nt_xent_loss(z, z.clone(), tau=0.2)

        z_collapse = torch.ones(B, D)
        loss_collapse = mod.nt_xent_loss(z_collapse, z_collapse.clone(), tau=0.2)

        assert loss_sep.item() < loss_collapse.item(), (
            f"Separable loss {loss_sep.item():.4f} should be < "
            f"collapse loss {loss_collapse.item():.4f}"
        )

    def test_nt_xent_loss_temperature_monotonicity(self):
        """Spec: tau=0.1 yields smaller loss than tau=1.0 for aligned positives."""
        mod = _require_module()
        B, D = 6, 32
        # Orthonormal positives: low tau should sharpen the distribution and lower loss
        z = _orthonormal_rows(B, D, seed=5)
        loss_cold = mod.nt_xent_loss(z, z.clone(), tau=0.1)
        loss_warm = mod.nt_xent_loss(z, z.clone(), tau=1.0)

        assert loss_cold.item() < loss_warm.item(), (
            f"Expected loss(tau=0.1)={loss_cold.item():.4f} < "
            f"loss(tau=1.0)={loss_warm.item():.4f}"
        )

    def test_nt_xent_loss_gradient_nonzero(self):
        """Spec: backward pass yields finite, non-None gradients on z1."""
        mod = _require_module()
        torch.manual_seed(99)
        z1 = torch.randn(8, 16, requires_grad=True)
        z2 = torch.randn(8, 16)
        loss = mod.nt_xent_loss(z1, z2, tau=0.2)
        loss.backward()

        assert z1.grad is not None, "z1.grad is None after backward()"
        assert torch.isfinite(z1.grad).all(), "z1.grad contains non-finite values"
        assert z1.grad.abs().sum().item() > 0.0, "z1.grad is all-zero (no gradient flow)"

    def test_nt_xent_loss_single_pair(self):
        """Edge case: B=1 — with a single pair the only candidate is the positive.

        With B=1, after concatenation we have 2 rows. Diagonal is masked.
        For each row, 1 remaining entry is the positive — loss should be small
        but not NaN/inf (it approaches 0 for perfectly aligned positive).
        """
        mod = _require_module()
        # B=1, z1 and z2 identical -> perfect positive alignment
        z = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        loss = mod.nt_xent_loss(z, z.clone(), tau=0.2)

        assert torch.isfinite(loss), f"B=1 loss is not finite: {loss.item()}"
        # With 2B-1=1 candidate there's only the positive -> CE = log(1) = 0
        assert abs(loss.item()) < 1e-4, (
            f"B=1 with perfect positive: expected loss~0, got {loss.item()}"
        )


# ===========================================================================
# augment tests
# ===========================================================================


def _make_bar_patch(B: int, H: int, W: int, bar_lo: int, bar_hi: int) -> torch.Tensor:
    """Create (B,1,H,W) with a horizontal energy bar between rows bar_lo..bar_hi."""
    x = torch.zeros(B, 1, H, W)
    x[:, 0, bar_lo:bar_hi, :] = 1.0
    return x


def _make_freqs_khz(H: int, lo_khz: float = 20.0, hi_khz: float = 120.0) -> torch.Tensor:
    """Return (H,) tensor of evenly-spaced frequencies in kHz."""
    return torch.linspace(lo_khz, hi_khz, H)


class TestAugment:
    """Tests for the augment(x, freqs_khz, ...) function."""

    def test_augment_shape_preserved(self):
        """Spec: output shape == input shape (B,1,H,W)."""
        mod = _require_module()
        B, H, W = 4, 40, 48
        x = torch.randn(B, 1, H, W)
        freqs = _make_freqs_khz(H)
        out = mod.augment(x, freqs, max_df_khz=10.0, max_dt_frames=5)

        assert out.shape == x.shape, (
            f"Shape mismatch: expected {x.shape}, got {out.shape}"
        )

    def test_augment_zero_input_zero_output(self):
        """Spec: zero input -> zero output (no spurious energy injected)."""
        mod = _require_module()
        B, H, W = 4, 40, 48
        x = torch.zeros(B, 1, H, W)
        freqs = _make_freqs_khz(H)
        gen = torch.Generator()
        gen.manual_seed(0)
        out = mod.augment(x, freqs, max_df_khz=10.0, max_dt_frames=5, generator=gen)

        assert torch.allclose(out, torch.zeros_like(out)), (
            f"Zero input produced non-zero output (max={out.abs().max().item():.6e})"
        )

    def test_augment_deterministic_with_seed(self):
        """Spec: same generator seed yields identical output on two calls."""
        mod = _require_module()
        B, H, W = 4, 40, 48
        torch.manual_seed(0)
        x = torch.randn(B, 1, H, W)
        freqs = _make_freqs_khz(H)

        gen1 = torch.Generator()
        gen1.manual_seed(42)
        out1 = mod.augment(x, freqs, max_df_khz=10.0, max_dt_frames=8, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(42)
        out2 = mod.augment(x, freqs, max_df_khz=10.0, max_dt_frames=8, generator=gen2)

        assert torch.allclose(out1, out2), (
            "Two calls with the same generator seed produced different outputs"
        )

    def test_augment_inband_no_wraparound(self):
        """Spec (critical): energy bar never pushed off-edge or wrapped around.

        Source bar at interior rows [18,23) of H=40 (5 rows wide).
        Repeat 200 augmentations with large max_df_khz that *would* push it
        off-edge if unclamped.  For every output:
          - The lit (nonzero) rows form a contiguous block of height in [4,6]
            (bar_h=5 ± 1 interpolation slack).
          - Neither row 0 nor row H-1 is lit at the same time as the interior
            block (no wraparound).
        """
        mod = _require_module()
        H, W = 40, 48
        bar_lo, bar_hi = 18, 23  # 5-row interior bar
        bar_h = bar_hi - bar_lo

        freqs = _make_freqs_khz(H)  # 20..120 kHz
        # max_df_khz large enough to potentially shift > 10 rows if unclamped
        df_large = 60.0  # spans most of the 100 kHz band

        for trial in range(200):
            x = _make_bar_patch(1, H, W, bar_lo, bar_hi)
            gen = torch.Generator()
            gen.manual_seed(trial)
            out = mod.augment(
                x, freqs,
                max_df_khz=df_large,
                max_dt_frames=0,   # isolate vertical shift
                warp_lo=1.0, warp_hi=1.0,
                generator=gen,
            )
            # (1,1,H,W) -> (H, W)
            out_2d = out[0, 0]

            # Find rows with any nonzero energy
            row_has_energy = (out_2d.abs() > 1e-6).any(dim=1)  # (H,)
            lit_rows = row_has_energy.nonzero(as_tuple=True)[0]

            if len(lit_rows) == 0:
                # All-zero output only acceptable if the shift was exactly >=H
                # (which the clamp should prevent), so this is a failure
                pytest.fail(
                    f"Trial {trial}: all energy lost — bar pushed entirely off edge"
                )

            lo_lit = lit_rows.min().item()
            hi_lit = lit_rows.max().item()
            observed_height = hi_lit - lo_lit + 1

            # Height check: interpolation may cost at most 1 row on each end
            assert 4 <= observed_height <= 6, (
                f"Trial {trial}: lit-row height={observed_height}, expected [4,6] "
                f"(bar_h={bar_h}). Lo={lo_lit}, Hi={hi_lit}"
            )

            # No wraparound: the lit block must not straddle rows 0 AND H-1
            # simultaneously (that would only happen with modular wrapping)
            if row_has_energy[0].item() and row_has_energy[H - 1].item():
                pytest.fail(
                    f"Trial {trial}: energy at both row 0 and row {H-1} simultaneously — "
                    "suggests wraparound rather than clamping"
                )

    def test_augment_two_views_differ(self):
        """Spec: two independent augment calls on the same input are NOT allclose."""
        mod = _require_module()
        B, H, W = 4, 40, 48
        torch.manual_seed(0)
        x = torch.randn(B, 1, H, W)
        freqs = _make_freqs_khz(H)

        gen1 = torch.Generator()
        gen1.manual_seed(1)
        view1 = mod.augment(x, freqs, max_df_khz=10.0, max_dt_frames=8, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(2)  # different seed -> different augmentation
        view2 = mod.augment(x, freqs, max_df_khz=10.0, max_dt_frames=8, generator=gen2)

        assert not torch.allclose(view1, view2), (
            "Two augmentations with different seeds produced identical outputs — "
            "augment is not stochastic or seeds are not respected"
        )

    def test_augment_identity_no_op(self):
        """Spec: warp_lo=warp_hi=1.0, max_df_khz=0, max_dt_frames=0 -> identity."""
        mod = _require_module()
        B, H, W = 4, 40, 48
        torch.manual_seed(0)
        x = torch.randn(B, 1, H, W)
        freqs = _make_freqs_khz(H)

        gen = torch.Generator()
        gen.manual_seed(0)
        out = mod.augment(
            x, freqs,
            max_df_khz=0.0,
            max_dt_frames=0,
            warp_lo=1.0,
            warp_hi=1.0,
            generator=gen,
        )

        assert torch.allclose(out, x, atol=1e-5), (
            f"Identity augmentation not allclose to input "
            f"(max diff={( out - x).abs().max().item():.2e})"
        )

    def test_augment_single_sample_shape(self):
        """Edge case: B=1 should still produce (1,1,H,W) output."""
        mod = _require_module()
        H, W = 40, 48
        x = torch.randn(1, 1, H, W)
        freqs = _make_freqs_khz(H)
        out = mod.augment(x, freqs, max_df_khz=5.0, max_dt_frames=4)

        assert out.shape == (1, 1, H, W), (
            f"B=1: expected shape (1,1,{H},{W}), got {out.shape}"
        )

    def test_augment_per_sample_independence(self):
        """Each sample in a batch receives an INDEPENDENT random augmentation.

        Strategy: run augment on a batch where every sample is identical.
        If augmentation is truly per-sample, the outputs should NOT all be equal.
        """
        mod = _require_module()
        H, W = 40, 48
        single = torch.randn(1, 1, H, W)
        x = single.expand(8, -1, -1, -1).contiguous()
        freqs = _make_freqs_khz(H)

        gen = torch.Generator()
        gen.manual_seed(0)
        out = mod.augment(x, freqs, max_df_khz=10.0, max_dt_frames=8, generator=gen)

        # Not all outputs identical (per-sample random shifts differ)
        first = out[0]
        all_same = all(torch.allclose(out[i], first) for i in range(1, 8))
        assert not all_same, (
            "All 8 samples received identical augmentation — "
            "shifts are not per-sample independent"
        )


# ===========================================================================
# ContrastiveEncoder tests
# ===========================================================================


class TestContrastiveEncoder:
    """Tests for ContrastiveEncoder(embed_dim, proj_dim)."""

    def test_encoder_forward_shape_prod_size(self):
        """Spec: forward on (B,1,257,234) returns (B,embed_dim), (B,proj_dim)."""
        mod = _require_module()
        B, embed_dim, proj_dim = 4, 128, 64
        model = mod.ContrastiveEncoder(embed_dim=embed_dim, proj_dim=proj_dim)
        model.eval()

        x = torch.randn(B, 1, 257, 234)
        with torch.no_grad():
            embedding, projection = model(x)

        assert embedding.shape == (B, embed_dim), (
            f"embedding shape: expected ({B},{embed_dim}), got {embedding.shape}"
        )
        assert projection.shape == (B, proj_dim), (
            f"projection shape: expected ({B},{proj_dim}), got {projection.shape}"
        )
        assert torch.isfinite(embedding).all(), "embedding contains non-finite values"
        assert torch.isfinite(projection).all(), "projection contains non-finite values"

    def test_encoder_forward_shape_small_input(self):
        """Spec: AdaptiveAvgPool makes encoder size-agnostic; (2,1,40,48) must work."""
        mod = _require_module()
        B, embed_dim, proj_dim = 2, 128, 64
        model = mod.ContrastiveEncoder(embed_dim=embed_dim, proj_dim=proj_dim)
        model.eval()

        x = torch.randn(B, 1, 40, 48)
        with torch.no_grad():
            embedding, projection = model(x)

        assert embedding.shape == (B, embed_dim), (
            f"Small input embedding shape: expected ({B},{embed_dim}), got {embedding.shape}"
        )
        assert projection.shape == (B, proj_dim), (
            f"Small input projection shape: expected ({B},{proj_dim}), got {projection.shape}"
        )

    def test_encoder_embed_and_proj_dims_independent(self):
        """Additional: embed_dim and proj_dim can be set independently."""
        mod = _require_module()
        embed_dim, proj_dim = 256, 32
        model = mod.ContrastiveEncoder(embed_dim=embed_dim, proj_dim=proj_dim)
        model.eval()

        x = torch.randn(3, 1, 40, 48)
        with torch.no_grad():
            emb, proj = model(x)

        assert emb.shape == (3, embed_dim), f"embed_dim mismatch: {emb.shape}"
        assert proj.shape == (3, proj_dim), f"proj_dim mismatch: {proj.shape}"

    def test_encoder_finite_on_zero_input(self):
        """Additional: zero input should produce finite (not NaN/inf) outputs.

        BatchNorm should handle this — confirm no division-by-zero blowup.
        Use eval() mode to bypass BatchNorm running_stats issues with B=2.
        """
        mod = _require_module()
        model = mod.ContrastiveEncoder(embed_dim=128, proj_dim=64)
        model.eval()

        x = torch.zeros(2, 1, 40, 48)
        with torch.no_grad():
            emb, proj = model(x)

        assert torch.isfinite(emb).all(), "embedding has non-finite values on zero input"
        assert torch.isfinite(proj).all(), "projection has non-finite values on zero input"

    def test_encoder_projection_gradient_flows(self):
        """Additional: gradients flow all the way from projection head to conv weights."""
        mod = _require_module()
        model = mod.ContrastiveEncoder(embed_dim=128, proj_dim=64)
        model.train()

        x = torch.randn(4, 1, 40, 48)
        emb, proj = model(x)
        loss = proj.pow(2).mean()
        loss.backward()

        # Check that the first conv layer received gradients
        first_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                break

        assert first_conv is not None, "No Conv2d found in model"
        assert first_conv.weight.grad is not None, (
            "No gradient on first Conv2d weight after backward()"
        )
        assert torch.isfinite(first_conv.weight.grad).all(), (
            "First Conv2d weight grad contains non-finite values"
        )
        assert first_conv.weight.grad.abs().sum().item() > 0.0, (
            "First Conv2d weight grad is all-zero (gradient did not flow)"
        )
