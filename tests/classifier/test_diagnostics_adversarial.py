"""Adversarial tests for diagnostics.py — added by test-hardener.

Targets gaps NOT covered by the 17 original tests:

  A. Cage-tone scaling regression (the just-fixed MEDIUM issue):
     - _inject_cage_tone fallback uses _INJECTION_FALLBACK=0.1, not +20
     - sigma-scaled offset on [0,1] input stays < 1.0

  B. Diagnostic VAE edge cases:
     - n_epochs=0 returns valid shape (implementation clamps to 1)
     - latent_dim=1 does not crash
     - constant input produces identical embeddings (constant-output regime)

  C. per_band_cohens_d edge cases:
     - single cohort returns graceful skip (passed=True, value=0.0)
     - three cohorts returns the maximum pairwise d (not just the first pair)
     - band entirely outside frequency range returns finite value

  D. knn_same_cohort_rate edge cases:
     - k=1 (degenerate K-NN) still works
     - k > n_samples per cohort falls back gracefully (clips to n_total-1)

  E. notch_injection_test / raw_pixel_pca_d single-cohort graceful skip

  F. _cohens_d degenerate inputs:
     - a.size < 2 returns 0.0
     - both constant (pooled_std=0) returns 0.0

  G. render_markdown_report paths:
     - GO path: all 4 diagnostics passing -> "**GO**" in output
     - NO-GO path: at least one failure -> "**NO-GO**" and failed name in output

  H. CLI --smoke flag: runs without real WAV dirs, exits 0 or 1, writes a report

Total added: 17 tests
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.classifier.diagnostics import (  # noqa: E402
    DiagnosticResult,
    _cohens_d,
    _inject_cage_tone,
    knn_same_cohort_rate,
    notch_injection_test,
    per_band_cohens_d,
    raw_pixel_pca_d,
    train_diagnostic_vae,
)
from usv_spectrogram.classifier.diagnostics import (  # noqa: E402
    _INJECTION_FALLBACK,
    _INJECTION_STD_EPS,
    INJECTION_SIGMA,
)

_SCRIPT_PATH = REPO_ROOT / "scripts" / "cnn_cleaning_validation.py"


def _load_script_module():
    """Import cnn_cleaning_validation.py as a module; skip if not present."""
    if not _SCRIPT_PATH.exists():
        pytest.skip(f"CLI script not yet implemented: {_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("cnn_cv", _SCRIPT_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# A. Cage-tone scaling regression guard
# ---------------------------------------------------------------------------


def test_inject_cage_tone_fallback_uses_injection_fallback_constant():
    """When local_std < _INJECTION_STD_EPS (e.g. all-zeros input), _inject_cage_tone
    must use _INJECTION_FALLBACK (0.1) as the offset — NOT a legacy +20 dB fixed shift.

    This guards against re-introduction of the saturation bug that caused false-FAIL
    migration on the all_layers ablation (normalized [0,1] input). A +20 offset would
    completely saturate [0,1] cells; 0.1 is safe on both domains.
    """
    spec = np.zeros((10, 32, 32), dtype=np.float32)
    injected = _inject_cage_tone(spec, (50.0, 51.0), notch_depth_db=20.0)

    max_offset = float((injected - spec).max())
    assert abs(max_offset - _INJECTION_FALLBACK) < 1e-5, (
        f"Fallback injection offset should be _INJECTION_FALLBACK={_INJECTION_FALLBACK}, "
        f"got {max_offset:.6f}. The +20 dB legacy bug may have been re-introduced."
    )
    assert _INJECTION_FALLBACK < 1.0, (
        f"_INJECTION_FALLBACK={_INJECTION_FALLBACK} must be < 1.0 to avoid saturating "
        "normalized [0,1] input (the saturation bug was +20 dB)."
    )


def test_inject_cage_tone_sigma_scaled_offset_below_one_on_normalized_input():
    """On normalized [0,1] input, the sigma-scaled injection offset must stay < 1.0.

    The INJECTION_SIGMA * local_std formula produces an offset proportional to the
    local standard deviation. For [0,1]-range input, local_std < 0.5 (by definition),
    so INJECTION_SIGMA * local_std < INJECTION_SIGMA * 0.5 = 1.0 (INJECTION_SIGMA=2).
    This ensures the injected band does not saturate the [0,1] space.
    """
    rng = np.random.default_rng(99)
    # Uniform [0,1] input — representative of MAD/zscore-cleaned spectrograms.
    spec_norm = rng.random((20, 64, 64)).astype(np.float32)
    injected = _inject_cage_tone(spec_norm, (20.0, 30.0), notch_depth_db=20.0)

    max_offset = float((injected - spec_norm).max())
    # INJECTION_SIGMA=2, std of Uniform(0,1) ~ 0.29, so offset ~ 0.58 < 1.0.
    assert max_offset < 1.0, (
        f"Sigma-scaled injection offset on [0,1] input was {max_offset:.4f}, expected < 1.0. "
        "This indicates the injection is no longer sigma-scaled (regression to fixed-dB)."
    )
    assert max_offset > 0.0, (
        "Injection offset was 0 — no tone was injected at all."
    )


def test_inject_cage_tone_does_not_modify_outside_notch_band():
    """_inject_cage_tone must only modify bins inside the notch band.

    Cells outside the specified kHz range must be byte-identical to the input.
    This prevents the injection from contaminating the non-notch region and
    confounding the migration measurement.
    """
    rng = np.random.default_rng(42)
    spec = rng.normal(-40.0, 8.0, (10, 50, 40)).astype(np.float32)
    # Inject into a narrow band (50-51 kHz at 250 kHz SR -> ~bins 19-20 of 50)
    injected = _inject_cage_tone(spec, (50.0, 51.0), notch_depth_db=20.0)

    diff = injected - spec
    # Outside the notch band, diff must be exactly 0.
    from usv_spectrogram.classifier.diagnostics import _khz_to_bin_range
    lo_bin, hi_bin = _khz_to_bin_range(50.0, 51.0, n_freq=50, sample_rate_hz=250_000)
    diff_outside = diff.copy()
    diff_outside[:, lo_bin:hi_bin + 1, :] = 0.0  # zero out the expected notch region
    assert np.all(diff_outside == 0.0), (
        f"_inject_cage_tone modified cells outside the notch band [{lo_bin}, {hi_bin}]. "
        "Only bins inside notch_band_khz should be affected."
    )


# ---------------------------------------------------------------------------
# B. Diagnostic VAE edge cases
# ---------------------------------------------------------------------------


def test_train_diagnostic_vae_n_epochs_zero_returns_valid_shape():
    """n_epochs=0 is clamped to 1 by the implementation (max(1, int(n_epochs))).

    The result must still be a valid (n_input, latent_dim) float32 array with
    no NaN or Inf. This tests the clamping behavior, not that the model is
    well-trained.
    """
    rng = np.random.default_rng(7)
    specs = rng.normal(-35.0, 10.0, (8, 16, 16)).astype(np.float32)

    embeddings = train_diagnostic_vae(
        spectrograms=specs, latent_dim=32, n_epochs=0, device="cpu",
    )

    assert embeddings.shape == (8, 32), (
        f"n_epochs=0 must still return (n_input=8, latent_dim=32), got {embeddings.shape}"
    )
    assert not np.any(np.isnan(embeddings)), "n_epochs=0 produced NaN embeddings."
    assert not np.any(np.isinf(embeddings)), "n_epochs=0 produced Inf embeddings."


def test_train_diagnostic_vae_latent_dim_1_does_not_crash():
    """latent_dim=1 is a degenerate but valid configuration.

    The hidden_dim computation (max(64, latent_dim * 2) = 64) must not break.
    Output shape must be (n_input, 1).
    """
    rng = np.random.default_rng(13)
    specs = rng.normal(-35.0, 10.0, (8, 16, 16)).astype(np.float32)

    embeddings = train_diagnostic_vae(
        spectrograms=specs, latent_dim=1, n_epochs=1, device="cpu",
    )

    assert embeddings.shape == (8, 1), (
        f"latent_dim=1 must return shape (8, 1), got {embeddings.shape}"
    )
    assert np.all(np.isfinite(embeddings)), "latent_dim=1 produced non-finite embeddings."


def test_train_diagnostic_vae_all_identical_input_produces_identical_embeddings():
    """When all input spectrograms are identical, the VAE encoder should produce
    identical embeddings for all samples (constant-output regime).

    The standardisation step in _train_diagnostic_vae_with_encoder sets std=1.0
    when the input std < 1e-8 (constant), so all samples map to the same
    standardised value and the encoder output is deterministic per sample.
    """
    spec_single = np.full((16, 16), -40.0, dtype=np.float32)
    specs = np.stack([spec_single] * 8, axis=0)  # 8 identical spectrograms

    embeddings = train_diagnostic_vae(
        spectrograms=specs, latent_dim=32, n_epochs=1, device="cpu",
    )

    assert embeddings.shape == (8, 32)
    assert np.all(np.isfinite(embeddings))
    # All embeddings must be identical (same input -> same encoder output).
    for i in range(1, 8):
        np.testing.assert_array_almost_equal(
            embeddings[0], embeddings[i], decimal=5,
            err_msg=f"Embedding[0] != Embedding[{i}] for identical input spectrograms.",
        )


# ---------------------------------------------------------------------------
# C. per_band_cohens_d edge cases
# ---------------------------------------------------------------------------


def test_per_band_cohens_d_single_cohort_returns_graceful_skip():
    """With only one cohort, per_band_cohens_d cannot compute a pairwise d.

    Spec: must return passed=True with value=0.0 and include a 'reason' key
    in details (matching the <2 cohorts guard in the implementation).
    """
    rng = np.random.default_rng(42)
    specs = rng.normal(-40.0, 8.0, (20, 50, 30)).astype(np.float32)

    result = per_band_cohens_d({"only_cohort": specs})

    assert isinstance(result, DiagnosticResult)
    assert result.passed is True, (
        "Single-cohort per_band_cohens_d must report passed=True (skip, not failure)."
    )
    assert result.value == pytest.approx(0.0), (
        f"Single-cohort skip must report value=0.0, got {result.value}"
    )
    assert "reason" in result.details, (
        "Single-cohort skip result must include a 'reason' key in details."
    )


def test_per_band_cohens_d_three_cohorts_returns_max_pairwise_d():
    """With 3 cohorts (A, B, C), per_band_cohens_d must return the max pairwise d
    across ALL pairs: (A,B), (A,C), (B,C).

    Test setup: A ~ N(-10, 5), B ~ N(0, 5), C ~ N(+10, 5).
    The A-C pair has the largest mean difference (20 dB) and should be selected.
    Expected: max pair is (A, C) and |d| is approximately 20/5 = 4.

    This guards against an implementation that only checks the first pair.
    """
    rng = np.random.default_rng(7)
    n_specs, n_freq, n_time = 50, 50, 30

    specs_A = rng.normal(-10.0, 5.0, (n_specs, n_freq, n_time)).astype(np.float32)
    specs_B = rng.normal(0.0, 5.0, (n_specs, n_freq, n_time)).astype(np.float32)
    specs_C = rng.normal(+10.0, 5.0, (n_specs, n_freq, n_time)).astype(np.float32)

    result = per_band_cohens_d({"A": specs_A, "B": specs_B, "C": specs_C})

    assert isinstance(result, DiagnosticResult)
    # The A-C pair dominates: d ~ 4. The A-B pair has d ~ 2.
    # If only the first pair (A,B) were checked, |d| would be ~2, not ~4.
    assert abs(result.value) > 2.5, (
        f"With 3 cohorts where A-C d~4 and A-B d~2, the result should reflect the "
        f"maximum pairwise d (>2.5), got {result.value:.4f}. "
        "The implementation may only be checking the first pair."
    )
    # The max pair should involve A and C (not A and B).
    max_pair = result.details.get("max_d_pair")
    assert max_pair is not None, "DiagnosticResult.details must include 'max_d_pair'."
    assert set(max_pair) == {"A", "C"}, (
        f"Expected max-d pair to be (A, C), got {max_pair}. "
        "The A-C pair has the largest mean difference."
    )


def test_per_band_cohens_d_out_of_range_band_returns_finite():
    """A band entirely outside the spectrogram's frequency range (e.g., 200-300 kHz
    at 250 kHz SR where nyq=125 kHz) must not crash or return NaN.

    _khz_to_bin_range clips to [0, n_freq-1], so the out-of-range band maps to
    the last bin(s). Cohen's d on the last bin is finite (it has valid data).
    """
    rng = np.random.default_rng(11)
    specs_A = rng.normal(-40.0, 8.0, (20, 10, 30)).astype(np.float32)
    specs_B = rng.normal(-38.0, 8.0, (20, 10, 30)).astype(np.float32)

    result = per_band_cohens_d(
        {"A": specs_A, "B": specs_B},
        band_edges_khz=[(200.0, 300.0)],  # way above nyquist for n_freq=10, sr=250kHz
    )

    assert isinstance(result, DiagnosticResult)
    assert np.isfinite(result.value), (
        f"Out-of-range band produced non-finite d: {result.value}. "
        "Band clipping to the last bin should still yield a valid Cohen's d."
    )


# ---------------------------------------------------------------------------
# D. knn_same_cohort_rate edge cases
# ---------------------------------------------------------------------------


def test_knn_same_cohort_rate_k1_degenerate_still_works():
    """k=1 is the minimum valid K for KNN. With a single nearest neighbor,
    the same-cohort rate is a binary 0/1 per sample.

    The function must not crash and must return a finite rate in [0, 1].
    """
    rng = np.random.default_rng(55)
    emb_A = rng.normal(0.0, 1.0, (20, 8)).astype(np.float32)
    emb_B = rng.normal(0.0, 1.0, (20, 8)).astype(np.float32)

    result = knn_same_cohort_rate({"A": emb_A, "B": emb_B}, k=1)

    assert isinstance(result, DiagnosticResult)
    assert np.isfinite(result.value), f"k=1 produced non-finite rate: {result.value}"
    assert 0.0 <= result.value <= 1.0, (
        f"k=1 KNN rate must be in [0, 1], got {result.value}"
    )


def test_knn_same_cohort_rate_k_larger_than_n_samples_clips_gracefully():
    """When k > n_samples per cohort, the implementation clips k to the available
    reference size (k_eff = min(k+1, n_total)).

    Spec behavior: must not crash, must return a finite rate.  The clipped k may
    produce a rate that differs from the k requested, but it must not raise.

    This guards against a naive ``nn.fit().kneighbors(n_neighbors=k)`` that would
    fail when k >= n_samples.
    """
    rng = np.random.default_rng(66)
    # Only 5 samples per cohort, but k=100 requested.
    emb_A = rng.normal(0.0, 1.0, (5, 8)).astype(np.float32)
    emb_B = rng.normal(0.0, 1.0, (5, 8)).astype(np.float32)

    result = knn_same_cohort_rate({"A": emb_A, "B": emb_B}, k=100)

    assert isinstance(result, DiagnosticResult)
    assert np.isfinite(result.value), (
        f"k=100 with only 5 samples per cohort produced non-finite rate: {result.value}"
    )
    assert 0.0 <= result.value <= 1.0


# ---------------------------------------------------------------------------
# E. Single-cohort graceful skip for notch_injection_test and raw_pixel_pca_d
# ---------------------------------------------------------------------------


def test_notch_injection_test_single_cohort_returns_graceful_skip():
    """With only one cohort, notch_injection_test cannot inject into a second cohort.

    Spec: must return passed=True with value=0.0 and details['reason'] explaining
    the skip (matching the <2 cohorts guard at line 462-469 of diagnostics.py).
    """
    rng = np.random.default_rng(77)
    specs = rng.normal(-40.0, 8.0, (20, 32, 32)).astype(np.float32)

    result = notch_injection_test({"only_cohort": specs})

    assert isinstance(result, DiagnosticResult)
    assert result.passed is True, (
        "Single-cohort notch_injection_test must report passed=True (skip)."
    )
    assert result.value == pytest.approx(0.0)
    assert "reason" in result.details


def test_raw_pixel_pca_d_single_cohort_returns_graceful_skip():
    """With only one cohort, raw_pixel_pca_d cannot compute a between-cohort d.

    Spec: must return passed=True with value=0.0 and details['reason'] explaining
    the skip.
    """
    rng = np.random.default_rng(88)
    specs = rng.normal(-40.0, 8.0, (20, 32, 32)).astype(np.float32)

    result = raw_pixel_pca_d({"only_cohort": specs})

    assert isinstance(result, DiagnosticResult)
    assert result.passed is True
    assert result.value == pytest.approx(0.0)
    assert "reason" in result.details


# ---------------------------------------------------------------------------
# F. _cohens_d degenerate inputs
# ---------------------------------------------------------------------------


def test_cohens_d_single_element_array_returns_zero():
    """_cohens_d with a.size=1 or b.size=1 must return 0.0 (not crash on ddof=1).

    With fewer than 2 samples, np.var(ddof=1) would return NaN.
    The ``if a.size < 2 or b.size < 2`` guard at line 148 must catch this.
    """
    result = _cohens_d(np.array([5.0]), np.array([3.0, 4.0, 5.0]))
    assert result == pytest.approx(0.0), (
        f"_cohens_d with a single-element a must return 0.0, got {result}."
    )


def test_cohens_d_empty_array_returns_zero():
    """_cohens_d with an empty array must return 0.0, not crash."""
    result = _cohens_d(np.array([]), np.array([1.0, 2.0, 3.0]))
    assert result == pytest.approx(0.0)


def test_cohens_d_both_constant_arrays_returns_zero():
    """When both arrays are constant, pooled_std=0 and d is undefined.

    The ``if pooled <= 0.0: return 0.0`` guard must activate.
    Returning NaN or raising ZeroDivisionError would propagate into the
    diagnostic value and corrupt the pass/fail decision.
    """
    a = np.full(50, 7.0)
    b = np.full(50, 7.0)  # same constant as a -> zero pooled std
    result = _cohens_d(a, b)

    assert result == pytest.approx(0.0), (
        f"Identical-constant arrays must return d=0.0, got {result}. "
        "pooled_std=0 path must return 0 not NaN."
    )
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# G. render_markdown_report paths
# ---------------------------------------------------------------------------


def test_render_markdown_report_go_path_when_all_diagnostics_pass(tmp_path: Path):
    """render_markdown_report must include '**GO**' and unlock message when all 4
    diagnostics in 'all_layers' pass.

    This exercises the ``passed_all and len(all_layers) >= 4`` branch.
    """
    m = _load_script_module()

    passing = [
        DiagnosticResult("notch_injection_migration", 0.10, 0.30, "less_than", True),
        DiagnosticResult("per_band_cohens_d", 0.05, 0.30, "less_than", True),
        DiagnosticResult("knn_same_cohort_rate", 0.60, 0.85, "less_than", True),
        DiagnosticResult("raw_pixel_pca_d", 0.50, 1.50, "less_than", True),
    ]
    results = {"all_layers": passing}
    report_path = tmp_path / "report.md"

    m.render_markdown_report(results, {"cohort_A": 20, "cohort_B": 20}, 1.5, report_path)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "**GO**" in content, (
        "Report must contain '**GO**' when all 4 diagnostics pass under all_layers."
    )
    assert "**NO-GO**" not in content, (
        "Report must NOT contain '**NO-GO**' when all diagnostics pass."
    )
    assert "Module 18.2" in content, (
        "GO report must mention 'Module 18.2 is unlocked' per spec."
    )


def test_render_markdown_report_no_go_path_when_diagnostic_fails(tmp_path: Path):
    """render_markdown_report must include '**NO-GO**' and list the failed criterion
    when at least one diagnostic in 'all_layers' fails.

    This exercises the else branch and the failed_criteria list.
    """
    m = _load_script_module()

    results = {
        "all_layers": [
            DiagnosticResult("notch_injection_migration", 0.95, 0.30, "less_than", False),
            DiagnosticResult("per_band_cohens_d", 0.05, 0.30, "less_than", True),
        ]
    }
    report_path = tmp_path / "report.md"

    m.render_markdown_report(results, {"A": 10}, 3.0, report_path)

    content = report_path.read_text(encoding="utf-8")
    assert "**NO-GO**" in content, (
        "Report must contain '**NO-GO**' when a diagnostic fails under all_layers."
    )
    assert "notch_injection_migration" in content, (
        "NO-GO report must list the name of the failed criterion."
    )
    assert "**GO**" not in content


def test_render_markdown_report_creates_parent_directory(tmp_path: Path):
    """render_markdown_report must create the report's parent directory if it does
    not exist (uses ``output_path.parent.mkdir(parents=True, exist_ok=True)``).
    """
    m = _load_script_module()

    report_path = tmp_path / "deeply" / "nested" / "report.md"
    assert not report_path.parent.exists()

    results = {"all_layers": [
        DiagnosticResult("per_band_cohens_d", 0.1, 0.30, "less_than", True),
    ]}
    m.render_markdown_report(results, {}, 0.5, report_path)

    assert report_path.exists(), (
        "render_markdown_report must create parent directories if they do not exist."
    )


# ---------------------------------------------------------------------------
# H. CLI --smoke flag
# ---------------------------------------------------------------------------


def test_cli_smoke_flag_runs_without_wav_dirs_and_writes_report(tmp_path: Path):
    """--smoke mode must run end-to-end without requiring any real WAV directories
    and must write a Markdown report to the specified output-dir.

    The test accepts exit code 0 (all diagnostics pass) or 1 (some fail) — both
    are valid outcomes on synthetic data.  Only exit code 2+ (argument error or
    crash) is a failure.
    """
    if not _SCRIPT_PATH.exists():
        pytest.skip(f"CLI script not found: {_SCRIPT_PATH}")

    output_dir = tmp_path / "smoke_out"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--smoke",
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode in (0, 1), (
        f"--smoke exit code must be 0 (GO) or 1 (NO-GO), got {result.returncode}. "
        f"stdout: {result.stdout[:300]}, stderr: {result.stderr[:300]}"
    )

    report = output_dir / "cleaning-validation-report.md"
    assert report.exists(), (
        f"--smoke must write cleaning-validation-report.md to output-dir={output_dir}. "
        f"stdout: {result.stdout[:300]}"
    )

    content = report.read_text(encoding="utf-8")
    assert "# Cleaning Validation Report" in content, (
        "Report file must start with the expected Markdown heading."
    )
    # Must contain ablation table entries for at least one layer config.
    assert "| Diagnostic" in content, (
        "Report must contain at least one ablation table."
    )
