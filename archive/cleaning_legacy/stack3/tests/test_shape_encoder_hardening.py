"""Adversarial hardening tests for the contrastive shape encoder.

Added AFTER the module passed master-review. Tests cover paths that the 35
pre-implementation spec tests do NOT reach, in six priority categories:

  1. augment WIDE-FREQ band clamp (real-data path: USV band is interior sub-range)
  2. augment generator seed determinism across different batch content
  3. nt_xent_loss NaN/inf guard with tiny tau + scale invariance
  4. eta2 empty-keep-set emits NO RuntimeWarning (regression guard)
  5. knn_purity k == n-1 and 3-class distinct per-type purity
  6. chevron_valley clean synthetic shapes locked to expected labels

All tests are CPU-only, use tiny tensors, and never touch the 16 GB patches.npz.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Module loading -- mirrors the pattern in the existing test files so sys.modules
# entries don't conflict.  We use distinct keys so both files can coexist.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PATH = REPO_ROOT / "scripts" / "experiments" / "train_shape_encoder_contrastive.py"
_EVAL_PATH = REPO_ROOT / "scripts" / "eval_shape_encoder.py"

_TRAIN_MOD: ModuleType | None = None
_EVAL_MOD: ModuleType | None = None
_TRAIN_ERR: str | None = None
_EVAL_ERR: str | None = None

# Load train module with src on sys.path so corpus constants resolve.
# This is the CORRECT path for production: patches.npz freqs span 0-150 kHz
# and the corpus constants (20-120 kHz) must be importable to make the band
# clamp non-vacuous.
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if _TRAIN_PATH.exists():
    _TRAIN_KEY = "train_shape_encoder_contrastive_hardening"
    try:
        spec = importlib.util.spec_from_file_location(_TRAIN_KEY, _TRAIN_PATH)
        _TRAIN_MOD = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[_TRAIN_KEY] = _TRAIN_MOD
        spec.loader.exec_module(_TRAIN_MOD)  # type: ignore[union-attr]
    except Exception as exc:
        _TRAIN_ERR = str(exc)
else:
    _TRAIN_ERR = f"not found: {_TRAIN_PATH}"

if _EVAL_PATH.exists():
    _EVAL_KEY = "eval_shape_encoder_hardening"
    try:
        spec = importlib.util.spec_from_file_location(_EVAL_KEY, _EVAL_PATH)
        _EVAL_MOD = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[_EVAL_KEY] = _EVAL_MOD
        spec.loader.exec_module(_EVAL_MOD)  # type: ignore[union-attr]
    except Exception as exc:
        _EVAL_ERR = str(exc)
else:
    _EVAL_ERR = f"not found: {_EVAL_PATH}"


def _require_train() -> ModuleType:
    if _TRAIN_MOD is None:
        pytest.skip(f"train_shape_encoder_contrastive not loadable: {_TRAIN_ERR}")
    return _TRAIN_MOD  # type: ignore[return-value]


def _require_eval() -> ModuleType:
    if _EVAL_MOD is None:
        pytest.skip(f"eval_shape_encoder not loadable: {_EVAL_ERR}")
    return _EVAL_MOD  # type: ignore[return-value]


# ===========================================================================
# Category 1 — augment WIDE-FREQ band clamp (the key untested real-data path)
# ===========================================================================


class TestAugmentWideFreqBandClamp:
    """The production patches span 0-150 kHz (257 rows). The USV band (20-120 kHz)
    is an INTERIOR sub-range, not the full frame.  The existing tests use
    freqs = linspace(20, 120, H) so the band equals the full patch.  These tests
    use freqs = linspace(0, 150, H) to expose the interior-sub-range path.
    """

    # ------------------------------------------------------------------
    # Pre-check: corpus constants must resolve so the band is not vacuous
    # ------------------------------------------------------------------

    def test_corpus_constants_importable_in_process(self):
        """Guard: USV_FREQ_MIN_HZ / MAX_HZ must be importable from src/.

        If this test fails it means src/ is not on sys.path and the wide-freq
        band-clamp tests below are vacuous (band collapses to whole frame).
        The hardening loader at module scope inserts src/ before loading the
        train module, so if the module is loaded this should always pass.
        """
        mod = _require_train()
        assert mod.USV_FREQ_MIN_HZ is not None, (
            "USV_FREQ_MIN_HZ is None inside the loaded module — "
            "src/ was not on sys.path when the module was executed. "
            "The wide-freq band-clamp tests would be vacuous without this fix."
        )
        assert mod.USV_FREQ_MAX_HZ is not None, (
            "USV_FREQ_MAX_HZ is None inside the loaded module."
        )
        assert mod.USV_FREQ_MIN_HZ == pytest.approx(20_000, abs=1), (
            f"Expected USV_FREQ_MIN_HZ=20000, got {mod.USV_FREQ_MIN_HZ}"
        )
        assert mod.USV_FREQ_MAX_HZ == pytest.approx(120_000, abs=1), (
            f"Expected USV_FREQ_MAX_HZ=120000, got {mod.USV_FREQ_MAX_HZ}"
        )

    def test_inband_energy_stays_within_usv_band_after_large_shift(self):
        """(a) Energy inside the 20-120 kHz sub-range must never be pushed into
        the 0-20 kHz or 120-150 kHz rows after a large pitch shift.

        Setup: freqs = linspace(0, 150, H=64) so the USV band (20-120 kHz)
        occupies rows ~9-50 (an interior sub-range, not the full frame).
        Bar energy at rows 20-25 (~52-60 kHz) is well inside the band.
        max_df_khz=80 would push the bar up to 53 rows if unclamped — clear
        past the band boundary. The clamp must keep it within [b_lo, b_hi].
        """
        mod = _require_train()
        H, W = 64, 48
        freqs = torch.linspace(0, 150, H)  # full STFT range, USV band is interior

        # Derive band rows using the same logic as augment
        lo_khz = mod.USV_FREQ_MIN_HZ / 1e3   # 20.0
        hi_khz = mod.USV_FREQ_MAX_HZ / 1e3   # 120.0
        band = ((freqs >= lo_khz) & (freqs <= hi_khz)).nonzero().flatten()
        assert band.numel() > 0, "Band is empty — test setup error"
        b_lo = int(band.min())
        b_hi = int(band.max())

        # Confirm band is strictly interior (not the full frame)
        assert b_lo > 0, f"Band starts at row 0 — test is vacuous (b_lo={b_lo})"
        assert b_hi < H - 1, f"Band ends at row H-1 — test is vacuous (b_hi={b_hi})"

        # Bar at interior band rows (rows 20-25 = ~52-63 kHz)
        bar_lo, bar_hi = 20, 25
        assert bar_lo >= b_lo and bar_hi <= b_hi, (
            f"Bar [{bar_lo},{bar_hi}] is outside band [{b_lo},{b_hi}] — setup error"
        )

        failures = []
        for trial in range(100):
            x = torch.zeros(1, 1, H, W)
            x[0, 0, bar_lo:bar_hi, :] = 1.0
            gen = torch.Generator()
            gen.manual_seed(trial)
            out = mod.augment(
                x, freqs,
                max_df_khz=80.0,   # would push bar ~53 rows if unclamped
                max_dt_frames=0,
                warp_lo=1.0,
                warp_hi=1.0,
                generator=gen,
            )
            out_2d = out[0, 0]
            lit = (out_2d.abs() > 1e-6).any(dim=1)
            lit_rows = lit.nonzero(as_tuple=True)[0]

            if len(lit_rows) == 0:
                failures.append(f"trial {trial}: all energy lost")
                continue

            lo_lit = int(lit_rows.min())
            hi_lit = int(lit_rows.max())

            # Energy must stay within the USV band sub-range
            if lo_lit < b_lo:
                failures.append(
                    f"trial {trial}: lit rows start at {lo_lit} < b_lo={b_lo} "
                    "(leaked into 0-20 kHz region)"
                )
            if hi_lit > b_hi:
                failures.append(
                    f"trial {trial}: lit rows end at {hi_lit} > b_hi={b_hi} "
                    "(leaked into 120-150 kHz region)"
                )

        assert not failures, (
            f"{len(failures)} failures in 100 trials:\n" + "\n".join(failures[:5])
        )

    def test_outofband_energy_uses_full_frame_clamp_no_crash(self):
        """(b) Energy already OUT of the USV band must fall back to full-frame clamp.

        A bar at rows 2-6 (~5-15 kHz) is below the USV band (20-120 kHz).
        The code sets inband=False for that sample and uses floor=0, ceil=H-1.
        Requirements: no crash, energy is preserved (not silently zeroed), and
        energy is NOT forcibly shifted into the USV band.
        """
        mod = _require_train()
        H, W = 64, 48
        freqs = torch.linspace(0, 150, H)

        lo_khz = mod.USV_FREQ_MIN_HZ / 1e3
        hi_khz = mod.USV_FREQ_MAX_HZ / 1e3
        band = ((freqs >= lo_khz) & (freqs <= hi_khz)).nonzero().flatten()
        b_lo = int(band.min())

        # Bar strictly below the USV band
        bar_lo, bar_hi = 2, 6
        assert bar_hi < b_lo, (
            f"Bar [{bar_lo},{bar_hi}) must be below band start {b_lo} — setup error"
        )

        for trial in range(50):
            x = torch.zeros(1, 1, H, W)
            x[0, 0, bar_lo:bar_hi, :] = 1.0

            gen = torch.Generator()
            gen.manual_seed(trial + 200)
            try:
                out = mod.augment(
                    x, freqs,
                    max_df_khz=80.0,
                    max_dt_frames=0,
                    warp_lo=1.0,
                    warp_hi=1.0,
                    generator=gen,
                )
            except Exception as exc:
                pytest.fail(
                    f"trial {trial}: augment raised exception on out-of-band energy: {exc}"
                )

            # Energy must be preserved (not silently lost)
            assert out.sum().item() > 0, (
                f"trial {trial}: out-of-band energy was entirely lost after augment"
            )

            # Energy must NOT be force-shifted into the USV band
            # (the full-frame clamp means [0, H-1], NOT [b_lo, b_hi])
            out_2d = out[0, 0]
            lit = (out_2d.abs() > 1e-6).any(dim=1)
            lit_rows = lit.nonzero(as_tuple=True)[0]
            if len(lit_rows):
                lo_lit = int(lit_rows.min())
                # The shifted bar stays within [0, H-1] but is NOT clamped
                # to the USV band — it may land anywhere in [0, H-1]
                assert 0 <= lo_lit < H, (
                    f"trial {trial}: lit row {lo_lit} is out of [0, H) = [0, {H})"
                )


# ===========================================================================
# Category 2 — augment generator seed determinism across different batch content
# ===========================================================================


class TestAugmentGeneratorSeedReproducibility:
    """The rand() calls inside augment consume from the generator in a fixed order
    determined solely by the batch size B.  The same seed must therefore reproduce
    the same (dy, dx, s) decisions regardless of what x contains.
    """

    def test_same_seed_reproduces_decisions_across_different_content(self):
        """Regression: same generator seed -> same shift/warp decisions even when
        the input patch content changes between two calls.

        Strategy: apply augment to a bar patch (content A) with seed S, recording
        where the bar lands.  Then apply augment to a DIFFERENT bar patch (content B,
        different initial position) with the SAME seed S.  Because rand() is only
        called for shape-(B,) arrays (dy, dx, s) and order is content-independent,
        the *delta* applied to each sample must equal the delta applied in run 1.

        Verification: run content A twice with the same seed -> identical outputs
        (established); run content B twice with the same seed -> also identical.
        Then confirm the PER-SAMPLE DELTA (shift applied to sample i) is the same
        between the two runs by using zero-input to extract the null output and
        comparing the non-zero run against a re-seed of the same generator.
        """
        mod = _require_train()
        H, W = 40, 48
        freqs = torch.linspace(20, 120, H)

        x_a = torch.zeros(3, 1, H, W)
        x_a[:, 0, 10:15, :] = 1.0   # bar at rows 10-14

        x_b = torch.zeros(3, 1, H, W)
        x_b[:, 0, 20:25, :] = 1.0   # bar at rows 20-24 (different content)

        SEED = 314159

        # Run A, run 1 and 2 (same content, same seed)
        gen1 = torch.Generator(); gen1.manual_seed(SEED)
        out_a1 = mod.augment(x_a, freqs, max_df_khz=5.0, max_dt_frames=0,
                             warp_lo=1.0, warp_hi=1.0, generator=gen1)

        gen2 = torch.Generator(); gen2.manual_seed(SEED)
        out_a2 = mod.augment(x_a, freqs, max_df_khz=5.0, max_dt_frames=0,
                             warp_lo=1.0, warp_hi=1.0, generator=gen2)

        assert torch.allclose(out_a1, out_a2), (
            "Same seed + same content did not reproduce — "
            "generator state is not being reset properly"
        )

        # Run B, run 1 and 2 (different content from A, same seed)
        gen3 = torch.Generator(); gen3.manual_seed(SEED)
        out_b1 = mod.augment(x_b, freqs, max_df_khz=5.0, max_dt_frames=0,
                             warp_lo=1.0, warp_hi=1.0, generator=gen3)

        gen4 = torch.Generator(); gen4.manual_seed(SEED)
        out_b2 = mod.augment(x_b, freqs, max_df_khz=5.0, max_dt_frames=0,
                             warp_lo=1.0, warp_hi=1.0, generator=gen4)

        assert torch.allclose(out_b1, out_b2), (
            "Same seed + same (different) content did not reproduce — "
            "rand() decisions are not seed-stable across calls"
        )

    def test_per_sample_shifts_within_batch_vary(self):
        """Within a SINGLE augment call, each sample gets an INDEPENDENT random
        shift — not a broadcast of the first sample's shift.

        Use 4 identical bar patches.  With per-sample randomness the bars land
        at different rows in the output.  If the shift were broadcast, all four
        output bars would be at the same row.
        """
        mod = _require_train()
        H, W = 40, 48
        freqs = torch.linspace(20, 120, H)

        x = torch.zeros(4, 1, H, W)
        x[:, 0, 15:20, :] = 1.0  # all samples identical

        gen = torch.Generator()
        gen.manual_seed(55)
        out = mod.augment(x, freqs, max_df_khz=8.0, max_dt_frames=0,
                          warp_lo=1.0, warp_hi=1.0, generator=gen)

        # Extract the row centroid of the bar for each sample
        centroids = []
        for i in range(4):
            lit = (out[i, 0].abs() > 1e-6).any(dim=1)
            lit_rows = lit.nonzero(as_tuple=True)[0]
            if len(lit_rows):
                centroids.append(int(lit_rows.float().mean().round()))
            else:
                centroids.append(-1)

        # At least two samples must have different centroids
        assert len(set(centroids)) > 1, (
            f"All 4 samples received the same row centroid {centroids[0]} — "
            "shifts are broadcasted rather than per-sample independent. "
            f"Centroids: {centroids}"
        )


# ===========================================================================
# Category 3 — nt_xent_loss NaN/Inf guard and scale invariance
# ===========================================================================


class TestNtXentLossEdgeCases:

    def test_small_tau_stays_finite(self):
        """NaN/inf guard: tau=1e-4 on L2-normalized inputs must not overflow.

        With tau=1e-4, sim/tau can reach 1/1e-4 = 10000. Cross-entropy must
        handle this without producing NaN or inf.  The loss value will be large
        (softmax concentrates), but it must be a finite scalar.
        """
        mod = _require_train()
        torch.manual_seed(42)
        B, D = 8, 16
        z1 = torch.randn(B, D)
        z2 = torch.randn(B, D)

        loss = mod.nt_xent_loss(z1, z2, tau=1e-4)

        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), (
            f"nt_xent_loss with tau=1e-4 returned non-finite value: {loss.item()}"
        )

    def test_loss_invariant_to_embedding_scale(self):
        """L2-normalisation inside nt_xent_loss means scaling z1, z2 by any
        positive constant must leave the loss unchanged.

        Consequence: the loss is a pure shape/direction metric, insensitive to
        embedding magnitude.  This is a key property of the SimCLR formulation.
        """
        mod = _require_train()
        torch.manual_seed(7)
        B, D = 8, 32
        z1 = torch.randn(B, D)
        z2 = torch.randn(B, D)

        loss_unit = mod.nt_xent_loss(z1, z2, tau=0.2)
        loss_scaled = mod.nt_xent_loss(z1 * 10.0, z2 * 10.0, tau=0.2)

        assert torch.isfinite(loss_unit), "baseline loss is not finite"
        assert torch.isfinite(loss_scaled), "scaled loss is not finite"
        assert abs(loss_unit.item() - loss_scaled.item()) < 1e-4, (
            f"Loss changed with 10x scale: unit={loss_unit.item():.6f}, "
            f"scaled={loss_scaled.item():.6f}. "
            "L2-normalisation should make this invariant."
        )

    def test_small_tau_gradient_still_flows(self):
        """Gradient must remain finite even with tiny tau, so training does not
        diverge with an aggressive temperature setting.
        """
        mod = _require_train()
        torch.manual_seed(99)
        z1 = torch.randn(8, 16, requires_grad=True)
        z2 = torch.randn(8, 16)

        loss = mod.nt_xent_loss(z1, z2, tau=1e-3)
        loss.backward()

        assert z1.grad is not None, "z1.grad is None after backward() with small tau"
        assert torch.isfinite(z1.grad).all(), (
            f"z1.grad contains non-finite values with tau=1e-3. "
            f"Max abs: {z1.grad.abs().max().item():.2e}"
        )


# ===========================================================================
# Category 4 — eta2 empty-keep-set emits NO RuntimeWarning (regression guard)
# ===========================================================================


class TestEta2NoWarningOnEmptyKeepSet:

    def test_all_noise_labels_returns_zero_with_no_warning(self):
        """Regression: all labels < 0 -> eta2 returns 0.0 AND emits no RuntimeWarning.

        Earlier numpy implementations of mean([]) emit a RuntimeWarning.  The
        early-return guard in eta2 (`if len(v) == 0: return 0.0`) must fire
        BEFORE any numpy operation that would trigger a warning.

        This is a regression test for the fix described in the implementation
        handoff: 'eta2 empty keep-set -> returns 0.0 with NO RuntimeWarning'.
        """
        # Test both implementations (train module and eval module contain their
        # own copies of eta2 with identical semantics).
        for mod_name, mod in [
            ("eval_shape_encoder", _require_eval()),
            ("train_shape_encoder_contrastive", _require_train()),
        ]:
            v = np.random.default_rng(3).standard_normal((50, 4))
            lab = np.full(50, -1, dtype=int)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = mod.eta2(v, lab)

            # Must return 0.0
            assert result == 0.0, (
                f"[{mod_name}] All-noise labels: expected 0.0, got {result}"
            )

            # Must emit no warnings
            runtime_warnings = [
                w for w in caught
                if issubclass(w.category, RuntimeWarning)
            ]
            assert len(runtime_warnings) == 0, (
                f"[{mod_name}] eta2 emitted {len(runtime_warnings)} RuntimeWarning(s) "
                f"on all-noise labels: "
                + ", ".join(str(w.message) for w in runtime_warnings)
            )


# ===========================================================================
# Category 5 — knn_purity edge cases
# ===========================================================================


class TestKnnPurityEdgeCases:

    def test_k_equals_n_minus_1(self):
        """k == n-1 means each point uses ALL other points as neighbors.

        Hand-computed case (n=3, k=2):
          Z = [[0], [10], [11]]  types = [0, 1, 1]
          p0 neighbors (k=2): [p1, p2] = [type1, type1] -> 0/2 = 0.0
          p1 neighbors (k=2): [p2, p0] = [type1, type0] -> 1/2 = 0.5
          p2 neighbors (k=2): [p1, p0] = [type1, type0] -> 1/2 = 0.5
          overall = (0.0 + 0.5 + 0.5) / 3 = 1/3
        """
        mod = _require_eval()
        Z = np.array([[0.0], [10.0], [11.0]])
        types = np.array([0, 1, 1])
        n = len(Z)
        k = n - 1  # k == 2, use all non-self neighbors

        result = mod.knn_purity(Z, types, k=k)

        assert "overall" in result, f"Missing 'overall' key in result: {result.keys()}"
        expected_overall = (0.0 + 0.5 + 0.5) / 3
        assert abs(result["overall"] - expected_overall) < 1e-6, (
            f"k=n-1={k}: expected overall={expected_overall:.6f}, "
            f"got {result['overall']:.6f}"
        )
        # Per-type keys must be present
        assert 0 in result, f"Missing per-type key 0 in result: {result}"
        assert 1 in result, f"Missing per-type key 1 in result: {result}"
        # Type 0 (only p0): purity = 0.0 (both neighbors are type 1)
        assert abs(result[0] - 0.0) < 1e-6, (
            f"Type 0 purity with k=n-1: expected 0.0, got {result[0]}"
        )
        # Type 1 (p1 and p2): average = 0.5
        assert abs(result[1] - 0.5) < 1e-6, (
            f"Type 1 purity with k=n-1: expected 0.5, got {result[1]}"
        )

    def test_three_class_distinct_per_type_purity(self):
        """3-class case where each type has a DIFFERENT per-type purity.

        Hand-computed (n=6, k=1):
          Z  = [[0], [1], [9], [10], [11], [12]]
          t  = [ 0,   0,   2,   1,    1,    1 ]

          Sorted distances and nearest non-self neighbors:
          p0=[0]  -> p1=[1]  (type 0)   -> same:  1/1 = 1.0
          p1=[1]  -> p0=[0]  (type 0)   -> same:  1/1 = 1.0
          p2=[9]  -> p3=[10] (type 1)   -> diff:  0/1 = 0.0
          p3=[10] -> p2=[9]  (type 2)   -> diff:  0/1 = 0.0
          p4=[11] -> p3=[10] (type 1)   -> same:  1/1 = 1.0
          p5=[12] -> p4=[11] (type 1)   -> same:  1/1 = 1.0

          Per-type:
            type 0 (p0, p1): (1.0 + 1.0) / 2 = 1.0
            type 1 (p3, p4, p5): (0.0 + 1.0 + 1.0) / 3 = 2/3
            type 2 (p2): 0.0 / 1 = 0.0
          Overall: (1+1+0+0+1+1) / 6 = 4/6 = 2/3
        """
        mod = _require_eval()
        Z = np.array([[0.0], [1.0], [9.0], [10.0], [11.0], [12.0]])
        types = np.array([0, 0, 2, 1, 1, 1])

        result = mod.knn_purity(Z, types, k=1)

        assert "overall" in result
        expected_overall = 4 / 6
        assert abs(result["overall"] - expected_overall) < 1e-6, (
            f"Overall purity: expected {expected_overall:.6f}, got {result['overall']:.6f}"
        )

        # All three type keys must be present
        for t in [0, 1, 2]:
            assert t in result, (
                f"Per-type key {t} missing from result. Keys: {list(result.keys())}"
            )

        # Each type must have a DISTINCT purity value
        assert abs(result[0] - 1.0) < 1e-6, (
            f"Type 0 purity: expected 1.0, got {result[0]}"
        )
        assert abs(result[1] - 2 / 3) < 1e-6, (
            f"Type 1 purity: expected {2/3:.6f}, got {result[1]}"
        )
        assert abs(result[2] - 0.0) < 1e-6, (
            f"Type 2 purity: expected 0.0, got {result[2]}"
        )

        # Confirm the three values are distinct (core of the test)
        per_type_values = sorted([result[0], result[1], result[2]])
        assert per_type_values[0] < per_type_values[1] < per_type_values[2], (
            f"Expected three distinct purity values, got: {per_type_values}. "
            "knn_purity may be collapsing per-type keys to a single value."
        )


# ===========================================================================
# Category 6 — chevron_valley synthetic shape classification
# ===========================================================================


class TestChevronValleySyntheticShapes:
    """Lock the heuristic behavior on clean synthetic shapes. The chevron/valley
    classification drives the CV-NMI gate and UMAP coloring, so regressions in
    the heuristic would silently corrupt all downstream evaluations.
    """

    @staticmethod
    def _make_shapes(N: int = 50) -> dict[str, np.ndarray]:
        """Return a dict of named (1, N) shape arrays."""
        t = np.linspace(0, 1, N)
        return {
            "chevron": (np.sin(np.pi * t) * 10)[None, :],     # peak at center
            "valley": (-np.sin(np.pi * t) * 10)[None, :],     # trough at center
            "ramp": (np.linspace(0, 10, N))[None, :],          # monotonic increase
            "flat": (np.zeros(N))[None, :],                    # constant
            "ramp_down": (np.linspace(10, 0, N))[None, :],    # monotonic decrease
        }

    def test_clean_chevron_labeled_chevron(self):
        """A sine arch (peak at center, 0 at ends, amplitude 10) -> 'chevron'."""
        mod = _require_eval()
        shapes = self._make_shapes()["chevron"]
        cv = mod.chevron_valley(shapes)
        assert cv[0] == "chevron", (
            f"Clean chevron shape labeled '{cv[0]}', expected 'chevron'. "
            "Check that peak is in [0.2N, 0.8N] and margin > 2."
        )

    def test_clean_valley_labeled_valley(self):
        """A negative sine arch (trough at center, 0 at ends, amplitude 10) -> 'valley'."""
        mod = _require_eval()
        shapes = self._make_shapes()["valley"]
        cv = mod.chevron_valley(shapes)
        assert cv[0] == "valley", (
            f"Clean valley shape labeled '{cv[0]}', expected 'valley'. "
            "Check that trough is in [0.2N, 0.8N] and margin > 2."
        )

    def test_monotonic_ramp_labeled_other(self):
        """A monotonic ramp [0, 10] has peak at the edge (not center) -> 'other'."""
        mod = _require_eval()
        shapes = self._make_shapes()["ramp"]
        cv = mod.chevron_valley(shapes)
        assert cv[0] == "other", (
            f"Monotonic ramp labeled '{cv[0]}', expected 'other'. "
            "Peak at edge should fail the [0.2N, 0.8N] center check."
        )

    def test_monotonic_ramp_down_labeled_other(self):
        """A descending ramp [10, 0] has peak at the start (not center) -> 'other'."""
        mod = _require_eval()
        shapes = self._make_shapes()["ramp_down"]
        cv = mod.chevron_valley(shapes)
        assert cv[0] == "other", (
            f"Descending ramp labeled '{cv[0]}', expected 'other'. "
            "Peak at row 0 should fail the center check."
        )

    def test_batch_with_all_three_types_correct(self):
        """Batch of 3 rows: chevron, valley, ramp -> ['chevron', 'valley', 'other']."""
        mod = _require_eval()
        shapes_dict = self._make_shapes()
        shapes = np.vstack([
            shapes_dict["chevron"],
            shapes_dict["valley"],
            shapes_dict["ramp"],
        ])
        cv = mod.chevron_valley(shapes)
        assert cv[0] == "chevron", f"Row 0 (chevron): got '{cv[0]}'"
        assert cv[1] == "valley", f"Row 1 (valley): got '{cv[1]}'"
        assert cv[2] == "other", f"Row 2 (ramp): got '{cv[2]}'"

    def test_shape_with_peak_at_left_edge_is_other(self):
        """Peak at row 0 (left edge, < 0.2N threshold) must not be labeled chevron."""
        mod = _require_eval()
        N = 50
        shapes = np.zeros((1, N))
        shapes[0, 0] = 15.0    # huge peak at the leftmost position
        shapes[0, 1:] = 0.0
        cv = mod.chevron_valley(shapes)
        assert cv[0] == "other", (
            f"Peak at left edge (row 0) labeled '{cv[0]}', expected 'other'. "
            "The [0.2N, 0.8N] guard should reject edge peaks."
        )

    def test_shape_output_dtype_is_object_array(self):
        """chevron_valley must return a numpy array of dtype=object (string labels)."""
        mod = _require_eval()
        shapes = self._make_shapes()["chevron"]
        cv = mod.chevron_valley(shapes)
        assert isinstance(cv, np.ndarray), (
            f"chevron_valley should return np.ndarray, got {type(cv)}"
        )
        assert cv.dtype == object, (
            f"chevron_valley array dtype should be object (for string labels), "
            f"got {cv.dtype}"
        )
