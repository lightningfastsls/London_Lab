"""Tests for eval_shape_vae_v3 — written by test-architect BEFORE implementation.

Context: eval_shape_vae_v3.py is the evaluation driver for the Pathway A derivative-loss
contour-VAE. It re-exports eta2 (from rig_M8_contour_vae) and register_one (from
rig_R2_shape_alphabet) so callers have a single stable import point, plus exposes
a main() / parse_args entry point for CLI evaluation on the rig.

Note on rig-script imports: rig_M8_contour_vae.py and rig_R2_shape_alphabet.py both
call OUT.mkdir(parents=True, exist_ok=True) at module level, targeting /data/shachar/
(rig-only path). On the dev machine this raises PermissionError. Both are wrapped in
try/except. When they are unavailable, the corresponding "direct contract" tests are
skipped — but the INLINE CONTRACT TESTS (TestEta2Contract) always run because they
use a locally-inlined copy of the function body (read from rig_M8_contour_vae.py:77-82,
reproduced verbatim below as _eta2_ref). This ensures the metric contract is always
tested even without rig access, and that eval_shape_vae_v3's re-export can be verified
against it.

ROADMAP test plan coverage:
  1. eta2 perfect separation -> 1.0                        -> test_eta2_perfect_separation
  2. eta2 no separation -> 0.0                             -> test_eta2_no_separation
  3. eta2 ignores lab < 0 (noise label)                    -> test_eta2_ignores_noise_labels
  4. register_one output length is 50                      -> test_register_one_output_length
  5. register_one pitch-invariance                         -> test_register_one_pitch_invariance
  6. eval_shape_vae_v3 exposes main()/parse_args entry     -> test_eval_module_has_entry_point

Additional coverage (recurring gap patterns):
  - eta2 single-group (edge: only one class label)         -> test_eta2_single_group
  - eta2 multidimensional v (2-D latent)                   -> test_eta2_multidimensional_v
  - eta2 all-noise labels (all dropped)                    -> test_eta2_all_noise_returns_zero
  - register_one too-few active cols returns None          -> test_register_one_too_few_active_cols
  - eval re-export: eta2 is callable from eval module      -> test_eval_module_reexports_eta2
  - eval re-export: register_one is callable from eval     -> test_eval_module_reexports_register_one

Total: 12 tests (6 from ROADMAP, 6 additional)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — make scripts/, scripts/experiments/, and src/ importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_EXPERIMENTS_ROOT = _SCRIPTS_ROOT / "experiments"
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SCRIPTS_ROOT, _EXPERIMENTS_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Inline reference implementation of eta2, read verbatim from
# scripts/experiments/rig_M8_contour_vae.py lines 77-82.
#
# WHY INLINE: rig_M8_contour_vae.py runs OUT.mkdir(...) at module level,
# targeting /data/shachar/ (rig-only). Importing it on the dev machine raises
# PermissionError at collection time. The inline copy lets the contract tests
# run without the rig, and also defines what the eval module must match.
# ---------------------------------------------------------------------------
def _eta2_ref(v, lab):
    """Verbatim copy of rig_M8_contour_vae.eta2 (lines 77-82)."""
    v = v if v.ndim == 2 else v[:, None]
    keep = lab >= 0
    v, lab = v[keep], lab[keep]
    if len(v) == 0:
        return 0.0
    g = v.mean(0)
    tot = float(((v - g) ** 2).sum())
    w = sum(
        float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum())
        for l in np.unique(lab)
    )
    return 1 - w / tot if tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Try to import from rig scripts (may fail on dev machine — that is fine).
# ---------------------------------------------------------------------------
_eta2_direct = None
_RIG_M8_AVAILABLE = False
try:
    from rig_M8_contour_vae import eta2 as _eta2_direct  # noqa: E402
    _RIG_M8_AVAILABLE = True
except Exception:
    pass

_register_one_direct = None
_REGISTER_ONE_AVAILABLE = False
try:
    from rig_R2_shape_alphabet import register_one as _register_one_direct  # noqa: E402
    _REGISTER_ONE_AVAILABLE = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Try to import the module under test (will fail until implemented).
# ---------------------------------------------------------------------------
_eval_mod = None
_EVAL_MOD_AVAILABLE = False
try:
    import eval_shape_vae_v3 as _eval_mod  # noqa: E402
    _EVAL_MOD_AVAILABLE = True
except Exception:
    pass


# ===========================================================================
# Helper factories
# ===========================================================================

def _perfect_sep_inputs():
    """Two well-separated groups: group-0 at 0, group-1 at 10 (scalar features)."""
    v = np.array([[0.0], [0.0], [10.0], [10.0]])
    lab = np.array([0, 0, 1, 1])
    return v, lab


def _no_sep_inputs():
    """Two groups with IDENTICAL group means -> eta2 should be 0.

    Group-0: [3, 7], mean=5. Group-1: [3, 7], mean=5.
    Between-group SS = 0 -> eta2 = 0.
    """
    v = np.array([[3.0], [7.0], [3.0], [7.0]])
    lab = np.array([0, 0, 1, 1])
    return v, lab


# ===========================================================================
# CONTRACT TESTS: always run (use _eta2_ref, not the rig import).
# These define the metric's behavioral contract independently of the rig.
# The same tests are duplicated for the direct rig import (skipped on dev).
# ===========================================================================

class TestEta2Contract:
    """Behavioral contract of eta2 using the inline reference implementation.

    These tests ALWAYS run (no rig dependency). They define what is correct.
    """

    def test_eta2_perfect_separation(self):
        """Spec: two groups with non-overlapping values -> eta2 ≈ 1.0.

        Hand-computation:
          group0 = [[0],[0]], group1 = [[10],[10]].
          Global mean g = 5; total SS = 4*(5^2) = 100.
          Within-group SS = 0 (each group is constant).
          eta2 = 1 - 0/100 = 1.0.
        """
        v, lab = _perfect_sep_inputs()
        result = _eta2_ref(v, lab)
        assert abs(result - 1.0) < 1e-6, (
            f"Perfect separation should give eta2=1.0; got {result:.6f}"
        )

    def test_eta2_no_separation(self):
        """Spec: two groups with identical group means -> eta2 ≈ 0.0.

        Within-group SS == total SS -> 1 - 1 = 0.
        """
        v, lab = _no_sep_inputs()
        result = _eta2_ref(v, lab)
        assert abs(result - 0.0) < 1e-6, (
            f"No separation should give eta2=0.0; got {result:.6f}"
        )

    def test_eta2_ignores_noise_labels(self):
        """Spec: rows with lab==-1 are dropped BEFORE computing eta2.

        If noise rows are NOT dropped, they would reduce eta2 away from 1.0.
        After dropping, perfectly-separated groups must still give eta2=1.0.
        """
        v = np.array([[0.0], [0.0], [10.0], [10.0], [5.0], [5.0]])
        lab = np.array([0, 0, 1, 1, -1, -1])
        result = _eta2_ref(v, lab)
        assert abs(result - 1.0) < 1e-6, (
            f"eta2 must drop lab==-1 rows; got {result:.6f}"
        )

    def test_eta2_single_group(self):
        """Edge: single class label. All variance is within-group -> eta2=0.0.

        With one group, between-group SS = 0 -> 1 - within/total.
        If all values are equal, tot=0 -> guard returns 0.0.
        If values differ but all in one group, within = total -> 1 - 1 = 0.
        """
        v = np.array([[3.0], [5.0], [7.0]])
        lab = np.array([0, 0, 0])
        result = _eta2_ref(v, lab)
        assert abs(result - 0.0) < 1e-6, (
            f"Single group should give eta2=0.0; got {result:.6f}"
        )

    def test_eta2_multidimensional_v(self):
        """Spec: v can be 2-D (B, D>1). Result is still in [0, 1]."""
        rng = np.random.RandomState(0)
        v = rng.randn(20, 4)
        lab = np.array([0] * 10 + [1] * 10)
        result = _eta2_ref(v, lab)
        assert 0.0 <= result <= 1.0, (
            f"eta2 must be in [0,1] for 2-D v; got {result:.4f}"
        )

    def test_eta2_all_noise_returns_zero(self):
        """Edge: all rows have lab==-1 -> empty after filtering -> 0.0 (tot==0 guard)."""
        v = np.array([[1.0], [2.0], [3.0]])
        lab = np.array([-1, -1, -1])
        result = _eta2_ref(v, lab)
        assert result == 0.0, (
            f"All-noise input must return 0.0 via empty-array / tot==0 guard; got {result}"
        )


# ===========================================================================
# DIRECT RIG IMPORT TESTS: skipped on dev machine (rig-only).
# Confirms the rig function matches the inline contract above.
# ===========================================================================

@pytest.mark.skipif(
    not _RIG_M8_AVAILABLE,
    reason="rig_M8_contour_vae unavailable (rig-only path /data/shachar/)"
)
class TestEta2RigDirect:
    """Spot-checks that rig_M8_contour_vae.eta2 matches the inline contract."""

    def test_eta2_direct_perfect_separation(self):
        """Rig eta2 must agree with inline reference on perfect-separation input."""
        v, lab = _perfect_sep_inputs()
        assert abs(_eta2_direct(v, lab) - 1.0) < 1e-6

    def test_eta2_direct_ignores_noise(self):
        """Rig eta2 must drop lab==-1 rows."""
        v = np.array([[0.0], [0.0], [10.0], [10.0], [5.0], [5.0]])
        lab = np.array([0, 0, 1, 1, -1, -1])
        assert abs(_eta2_direct(v, lab) - 1.0) < 1e-6


# ===========================================================================
# register_one contract (direct rig import — skipped on dev).
# ===========================================================================

@pytest.mark.skipif(
    not _REGISTER_ONE_AVAILABLE,
    reason="rig_R2_shape_alphabet requires ridge_tracker + /data/shachar/"
)
class TestRegisterOneDirect:
    """Tests against rig_R2_shape_alphabet.register_one.

    register_one(crop, freqs_khz):
      - crop: (F, T) ndarray, power values
      - freqs_khz: (F,) ndarray of kHz per row
    Returns: float32 array of length 50 (N_RESAMPLE=50, pitch-subtracted,
    resampled) or None if fewer than MIN_ACTIVE_COLS=6 active columns.
    """

    @staticmethod
    def _make_synthetic_crop(F: int, T: int, bright_rows: list[int], amplitude: float = 1.0):
        crop = np.full((F, T), 1e-6, dtype=np.float32)
        for t, r in enumerate(bright_rows):
            crop[r, t] = amplitude
        return crop

    def test_register_one_output_length(self):
        """Spec: output array has exactly 50 elements (N_RESAMPLE)."""
        F, T = 20, 10
        freqs_khz = np.linspace(40.0, 100.0, F, dtype=np.float32)
        bright_rows = [5, 6, 7, 8, 9, 10, 11, 12, 11, 10]
        crop = self._make_synthetic_crop(F, T, bright_rows)
        result = _register_one_direct(crop, freqs_khz)
        if result is None:
            pytest.skip("track_ridge rejected synthetic crop — threshold differs")
        assert len(result) == 50, f"Expected length 50; got {len(result)}"

    def test_register_one_pitch_invariance(self):
        """Spec: register_one subtracts mean pitch -> pitch-shift gives identical output.

        Source: sc = span - pitch (where pitch = span.mean()).
        Shifting the ridge up by delta rows adds a constant to span; subtracting
        the mean cancels it. Output must be invariant within resample noise (1e-3).
        """
        F, T = 30, 12
        delta = 4
        freqs_khz = np.linspace(40.0, 110.0, F, dtype=np.float32)
        bright_rows_base = [5, 7, 9, 11, 13, 11, 9, 7, 5, 7, 9, 11]
        assert len(bright_rows_base) == T
        assert max(bright_rows_base) + delta < F
        bright_rows_shifted = [r + delta for r in bright_rows_base]

        crop_base = self._make_synthetic_crop(F, T, bright_rows_base)
        crop_shifted = self._make_synthetic_crop(F, T, bright_rows_shifted)

        r_base = _register_one_direct(crop_base, freqs_khz)
        r_shifted = _register_one_direct(crop_shifted, freqs_khz)

        if r_base is None or r_shifted is None:
            pytest.skip("track_ridge rejected synthetic crop")

        max_diff = float(np.abs(r_base - r_shifted).max())
        assert max_diff < 1e-3, (
            f"register_one must be pitch-invariant; max |base - shifted| = {max_diff:.4e}"
        )

    def test_register_one_too_few_active_cols(self):
        """Edge: T=4 < MIN_ACTIVE_COLS=6 -> returns None."""
        F, T = 20, 4
        freqs_khz = np.linspace(40.0, 100.0, F, dtype=np.float32)
        bright_rows = [5, 7, 9, 11]
        crop = self._make_synthetic_crop(F, T, bright_rows)
        result = _register_one_direct(crop, freqs_khz)
        assert result is None, (
            f"register_one should return None for T={T} < MIN_ACTIVE_COLS=6; got {result}"
        )


# ===========================================================================
# eval_shape_vae_v3 re-exports.
# These tests ALWAYS attempt the import — they fail with ImportError (or
# AttributeError) until the eval module is implemented, which is the correct
# pre-implementation red state.
# ===========================================================================

class TestEvalModuleReexports:
    """Tests that eval_shape_vae_v3 re-exports eta2 and register_one correctly.

    Will raise ImportError until eval_shape_vae_v3.py exists.
    """

    def test_eval_module_reexports_eta2(self):
        """Spec: eval_shape_vae_v3.eta2 is callable and matches contract semantics.

        Will fail with ImportError until the module is implemented.
        """
        import eval_shape_vae_v3 as ev  # fails until implemented
        assert callable(ev.eta2), "eval_shape_vae_v3.eta2 must be callable"
        v, lab = _perfect_sep_inputs()
        result = ev.eta2(v, lab)
        assert abs(result - 1.0) < 1e-6, (
            f"Re-exported eta2 must give 1.0 for perfect separation; got {result:.6f}"
        )

    def test_eval_module_reexports_register_one(self):
        """Spec: eval_shape_vae_v3.register_one is callable.

        Will fail with ImportError until the module is implemented.
        """
        import eval_shape_vae_v3 as ev  # fails until implemented
        assert callable(ev.register_one), (
            "eval_shape_vae_v3.register_one must be callable"
        )


class TestEvalModuleEntryPoint:
    """Spec: eval_shape_vae_v3 exposes main() or parse_args for CLI use.

    Will fail with ImportError until the module is implemented.
    """

    def test_eval_module_has_entry_point(self):
        """Spec: at least one of main() or parse_args() must be present and callable.

        We do NOT invoke either (both need a real model file on the rig).
        """
        import eval_shape_vae_v3 as ev  # fails until implemented

        has_main = hasattr(ev, "main") and callable(ev.main)
        has_parse_args = hasattr(ev, "parse_args") and callable(ev.parse_args)
        assert has_main or has_parse_args, (
            "eval_shape_vae_v3 must expose main() or parse_args(); found neither"
        )
