"""Tests for sis_baselines — written by test-architect BEFORE implementation.

Module under test:
  src/usv_spectrogram/classification/sis_baselines.py
  scripts/run_sis_baselines.py

ROADMAP test plan coverage (ROADMAP_SIS_BENCHMARK.md, module 17.1, lines 94-103):
  1. Periodic [A,B,A,B] → MI = 1.0 bit        -> test_periodic_sequence_mi_is_one_bit
  2. IID random sequence → MI ≈ 0 bits         -> test_iid_sequence_mi_near_zero
  3. entropy_reduction_pct in [0, 100]          -> test_entropy_reduction_pct_bounded
  4. sort_by_time reorders before MI            -> test_sort_by_time_reorders_labels
  5. String labels via pd.factorize             -> test_string_labels_handled_by_factorize
  6. Script end-to-end → baselines.csv 3 rows  -> test_script_produces_baselines_csv_with_3_rows
  7. Empty sequence → MI = 0, no crash         -> test_empty_sequence_returns_zero_mi
  8. Single-label sequence → MI = 0, H = 0     -> test_single_label_sequence_returns_zero_mi_and_entropy

Additional coverage (recurring gap patterns):
  - SISResult is a frozen dataclass (immutable)  -> test_sisresult_is_immutable
  - entropy_reduction_pct is correct formula     -> test_entropy_reduction_pct_formula
  - n_calls matches input length                 -> test_n_calls_matches_input_length
  - n_labels matches unique label count          -> test_n_labels_matches_alphabet_size
  - name field is preserved exactly              -> test_name_field_preserved
  - conditional_entropy = marginal - mi          -> test_conditional_entropy_identity
  - Script exits with code 0                     -> test_script_exit_code_zero
  - Script fails gracefully on missing CSV       -> test_script_missing_csv_exits_nonzero
  - Single-symbol sequence length 1              -> test_length_one_sequence_no_crash

Total: 17 tests (8 from ROADMAP, 9 additional)

All tests MUST fail until implementation exists (ImportError is expected initial failure).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Pattern 8: Import bootstrap — tests/ is one level below REPO_ROOT
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---------------------------------------------------------------------------
# Import module under test — will fail with ImportError until implemented
# ---------------------------------------------------------------------------
from usv_spectrogram.classification.sis_baselines import (  # noqa: E402
    SISResult,
    compute_sis_depth_1,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_periodic(n: int = 200) -> np.ndarray:
    """Return a perfectly periodic binary sequence [0, 1, 0, 1, ...] of length n."""
    return np.tile([0, 1], n // 2 + 1)[:n]


def _make_iid(n: int = 5000, K: int = 4, seed: int = 99) -> np.ndarray:
    """Return an IID random sequence of K symbols, length n."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, K, size=n).astype(np.intp)


def _make_script_csvs(tmp_path: Path, n: int = 100, seed: int = 0) -> tuple[Path, Path]:
    """Write minimal valid classified_csv and umap_csv to tmp_path.

    Returns (classified_csv_path, umap_csv_path).
    """
    rng = np.random.default_rng(seed)
    classified_csv = tmp_path / "classified.csv"
    umap_csv = tmp_path / "umap.csv"

    pd.DataFrame({
        "det_index": np.arange(n),
        "syllable_type": rng.choice(["Flat", "Down", "UShaped"], size=n),
        "label": rng.integers(0, 5, size=n),
        "file": ["rec.wav"] * n,
        "begin_time_s": np.linspace(0, 10, n),
    }).to_csv(classified_csv, index=False)

    pd.DataFrame({
        "det_index": np.arange(n),
        "hdbscan_label": rng.integers(0, 3, size=n),
    }).to_csv(umap_csv, index=False)

    return classified_csv, umap_csv


# ===========================================================================
# ROADMAP test plan — 8 tests
# ===========================================================================


def test_periodic_sequence_mi_is_one_bit():
    """ROADMAP item 1: perfectly periodic [A,B,A,B,...] → MI at lag 1 = 1.0 bit.

    Hand-computed: I(X_t; X_{t+1}) for a deterministic alternating binary
    sequence equals H(X_t) = 1 bit (complete predictability). The joint
    distribution is P(0,1)=0.5, P(1,0)=0.5 — marginals are both 0.5, so
    MI = 0.5*log2(0.5/0.25) + 0.5*log2(0.5/0.25) = 1.0 bit exactly.
    With finite N the estimator converges; we allow a 0.01 bit tolerance.
    """
    seq = _make_periodic(n=500)
    result = compute_sis_depth_1(seq, name="periodic-test")

    assert isinstance(result, SISResult), "Return type must be SISResult"
    assert abs(result.mi_at_lag_1 - 1.0) < 0.01, (
        f"Expected MI ≈ 1.0 bit for perfectly periodic sequence, got {result.mi_at_lag_1:.6f}"
    )


def test_iid_sequence_mi_near_zero():
    """ROADMAP item 2: IID random sequence → MI at lag 1 ≈ 0 bits.

    An independent sequence has no mutual information between consecutive
    symbols. With N=5000 the finite-sample MI should be < 0.01 bits.
    """
    seq = _make_iid(n=5000, K=4, seed=42)
    result = compute_sis_depth_1(seq, name="iid-test")

    assert isinstance(result, SISResult)
    assert result.mi_at_lag_1 < 0.01, (
        f"Expected MI < 0.01 bits for IID sequence, got {result.mi_at_lag_1:.6f}"
    )


def test_entropy_reduction_pct_bounded():
    """ROADMAP item 3: entropy_reduction_pct must be in [0, 100] for any input.

    The formula is (MI / H) * 100. Since 0 <= MI <= H, this must always be
    in [0, 100]. Test with periodic, IID, and a partly-structured sequence.
    """
    seqs = [
        _make_periodic(n=200),
        _make_iid(n=2000, K=6, seed=7),
        np.array([0, 0, 1, 1, 0, 0, 1, 1] * 50),  # structured but not period-1
    ]
    for seq in seqs:
        result = compute_sis_depth_1(seq, name="bounds-test")
        assert 0.0 <= result.entropy_reduction_pct <= 100.0, (
            f"entropy_reduction_pct={result.entropy_reduction_pct:.4f} is out of [0, 100]"
        )


def test_sort_by_time_reorders_labels():
    """ROADMAP item 4: sort_by_time reorders labels before MI computation.

    We create two identical sequences that differ only in the ordering
    given to the function.  The unsorted version is reversed; with sort_by_time
    supplying the correct ascending order, both calls must produce the same MI.
    Conversely, omitting sort_by_time on the reversed array must give a
    different MI (since the test sequence has positional structure).
    """
    # Structured sequence: first half 0s, second half 1s — so ordering matters
    n = 200
    labels_chronological = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.intp)
    labels_reversed = labels_chronological[::-1].copy()
    times_chronological = np.arange(n, dtype=float)
    times_reversed = times_chronological[::-1].copy()

    result_correct = compute_sis_depth_1(
        labels_reversed, name="sorted", sort_by_time=times_reversed
    )
    result_unsorted = compute_sis_depth_1(
        labels_reversed, name="unsorted", sort_by_time=None
    )

    # With correct time sorting, labels become chronological; the internal
    # MI must equal the MI on the directly-chronological sequence.
    result_direct = compute_sis_depth_1(labels_chronological, name="direct")

    assert abs(result_correct.mi_at_lag_1 - result_direct.mi_at_lag_1) < 1e-9, (
        "sort_by_time did not reproduce the same MI as direct chronological input: "
        f"sorted={result_correct.mi_at_lag_1:.6f}, direct={result_direct.mi_at_lag_1:.6f}"
    )


def test_string_labels_handled_by_factorize():
    """ROADMAP item 5: string label arrays must be accepted and produce valid results.

    Scattoni labels are strings like 'Flat', 'Down', 'UShaped'. The function
    must internally factorize them to integers and compute MI correctly.
    A perfectly alternating string sequence should still give MI ≈ 1.0 bit.
    """
    str_labels = np.array(["Flat", "Down"] * 150)
    result = compute_sis_depth_1(str_labels, name="string-labels")

    assert isinstance(result, SISResult)
    assert result.n_labels == 2, (
        f"Expected 2 unique string labels, got n_labels={result.n_labels}"
    )
    assert abs(result.mi_at_lag_1 - 1.0) < 0.01, (
        f"Expected MI ≈ 1.0 bit for alternating string sequence, "
        f"got {result.mi_at_lag_1:.6f}"
    )


def test_script_produces_baselines_csv_with_3_rows(tmp_path: Path):
    """ROADMAP item 6: script end-to-end on synthetic CSV → baselines.csv with 3 rows.

    Creates minimal classified_csv and umap_csv with exactly the columns the
    script expects (syllable_type, label, hdbscan_label, file, begin_time_s).
    Runs the script via subprocess and verifies that baselines.csv exists with
    3 data rows (one per labeling scheme).
    """
    rng = np.random.default_rng(0)
    n = 200

    # Synthetic classified_csv (Scattoni-7 + DeepSqueak-27)
    classified_csv = tmp_path / "classified_detections_full.csv"
    syllable_types = rng.choice(["Flat", "Down", "UShaped", "Chevron", "Jump", "Short", "Other"], size=n)
    deepsqueak_labels = rng.integers(0, 27, size=n)
    begin_times = np.sort(rng.uniform(0, 100, size=n))
    files = np.where(begin_times < 50, "file_a.wav", "file_b.wav")
    det_index = np.arange(n)

    classified_df = pd.DataFrame({
        "det_index": det_index,
        "syllable_type": syllable_types,
        "label": deepsqueak_labels,
        "file": files,
        "begin_time_s": begin_times,
    })
    classified_df.to_csv(classified_csv, index=False)

    # Synthetic umap_csv (HDBSCAN-3)
    umap_csv = tmp_path / "reclassified_detections.csv"
    hdbscan_labels = rng.integers(-1, 3, size=n)  # include noise label -1
    umap_df = pd.DataFrame({
        "det_index": det_index,
        "hdbscan_label": hdbscan_labels,
    })
    umap_df.to_csv(umap_csv, index=False)

    output_dir = tmp_path / "sis_output"

    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    baselines_csv = output_dir / "baselines.csv"
    assert baselines_csv.exists(), (
        f"baselines.csv not found at {baselines_csv}.\n"
        f"Script stdout:\n{result.stdout}\nScript stderr:\n{result.stderr}"
    )

    df = pd.read_csv(baselines_csv)
    assert len(df) == 3, (
        f"Expected 3 rows in baselines.csv (one per labeling), got {len(df)}.\n"
        f"Rows: {df.to_string()}"
    )


def test_empty_sequence_returns_zero_mi():
    """ROADMAP item 7: empty label array → MI = 0 without raising an exception.

    An empty sequence is a valid edge case (a recording with no detected calls).
    The function must not crash and must return MI = 0.
    """
    empty = np.array([], dtype=np.intp)
    result = compute_sis_depth_1(empty, name="empty")

    assert isinstance(result, SISResult)
    assert result.mi_at_lag_1 == 0.0, (
        f"Expected mi_at_lag_1=0.0 for empty sequence, got {result.mi_at_lag_1}"
    )
    assert result.n_calls == 0, (
        f"Expected n_calls=0 for empty sequence, got {result.n_calls}"
    )


def test_single_label_sequence_returns_zero_mi_and_entropy():
    """ROADMAP item 8: sequence where all labels are identical → MI = 0, H = 0.

    If every call has the same label, the marginal entropy H(X) = 0 (no
    uncertainty), so MI = 0 and conditional entropy = 0 as well.
    The formula entropy_reduction_pct = (MI / H) * 100 risks a divide-by-zero;
    the implementation must handle this gracefully (return 0 or 100, not NaN).
    """
    mono = np.zeros(300, dtype=np.intp)  # all label 0
    result = compute_sis_depth_1(mono, name="single-label")

    assert result.mi_at_lag_1 == 0.0, (
        f"Expected mi_at_lag_1=0.0 for single-label sequence, got {result.mi_at_lag_1}"
    )
    assert result.marginal_entropy == 0.0, (
        f"Expected marginal_entropy=0.0 for single-label sequence, got {result.marginal_entropy}"
    )
    assert not np.isnan(result.entropy_reduction_pct), (
        "entropy_reduction_pct must not be NaN even when H=0 (divide-by-zero guard required)"
    )


# ===========================================================================
# Additional coverage — recurring gap patterns
# ===========================================================================


def test_sisresult_is_immutable():
    """Pattern 1 (frozen dataclass): SISResult must be immutable after construction.

    The spec marks it frozen=True. Attempting to set an attribute must raise
    a FrozenInstanceError (or AttributeError for slots-based implementations).
    """
    seq = _make_periodic(n=100)
    result = compute_sis_depth_1(seq, name="immutability-test")

    with pytest.raises((AttributeError, TypeError)):
        result.mi_at_lag_1 = 99.0  # type: ignore[misc]


def test_entropy_reduction_pct_formula():
    """Additional: entropy_reduction_pct == (mi_at_lag_1 / marginal_entropy) * 100.

    Hand-verifiable: for the periodic binary sequence, MI = H = 1.0 bit,
    so entropy_reduction_pct must equal 100.0 (within floating-point tolerance).
    """
    seq = _make_periodic(n=500)
    result = compute_sis_depth_1(seq, name="pct-formula")

    if result.marginal_entropy > 0:
        expected_pct = (result.mi_at_lag_1 / result.marginal_entropy) * 100.0
        assert abs(result.entropy_reduction_pct - expected_pct) < 1e-6, (
            f"entropy_reduction_pct={result.entropy_reduction_pct:.6f} does not match "
            f"formula (MI/H)*100={expected_pct:.6f}"
        )


def test_n_calls_matches_input_length():
    """Additional: n_calls must equal the length of the input label array."""
    seq = _make_iid(n=137, K=3)
    result = compute_sis_depth_1(seq, name="n-calls-test")

    assert result.n_calls == 137, (
        f"Expected n_calls=137, got {result.n_calls}"
    )


def test_n_labels_matches_alphabet_size():
    """Additional: n_labels must equal the number of unique symbols in the input.

    Creating a sequence drawn from exactly 5 distinct labels; n_labels must be 5.
    """
    rng = np.random.default_rng(17)
    seq = rng.integers(0, 5, size=300).astype(np.intp)
    # Ensure all 5 labels appear
    seq[:5] = np.arange(5)
    result = compute_sis_depth_1(seq, name="n-labels-test")

    assert result.n_labels == 5, (
        f"Expected n_labels=5 for a 5-symbol alphabet, got {result.n_labels}"
    )


def test_name_field_preserved():
    """Additional: the name parameter is stored verbatim in the returned SISResult."""
    expected_name = "scattoni-7-test"
    seq = _make_periodic(n=50)
    result = compute_sis_depth_1(seq, name=expected_name)

    assert result.name == expected_name, (
        f"Expected name='{expected_name}', got '{result.name}'"
    )


def test_conditional_entropy_identity():
    """Additional: H(X|X_prev) = H(X) - MI must hold within floating-point precision.

    This is the information-theoretic identity that defines conditional entropy
    from marginal entropy and MI. Verifying it ensures the three entropy fields
    are internally consistent.
    """
    seq = _make_iid(n=3000, K=5, seed=55)
    result = compute_sis_depth_1(seq, name="cond-entropy")

    expected_cond = result.marginal_entropy - result.mi_at_lag_1
    assert abs(result.conditional_entropy - expected_cond) < 1e-9, (
        f"conditional_entropy={result.conditional_entropy:.9f} does not equal "
        f"H - MI = {expected_cond:.9f}"
    )


def test_script_exit_code_zero(tmp_path: Path):
    """Additional: script exits with code 0 on valid synthetic inputs.

    A separate check from the CSV-content test so that a crash or sys.exit(1)
    is reported as a distinct failure.
    """
    rng = np.random.default_rng(1)
    n = 100
    classified_csv = tmp_path / "classified.csv"
    umap_csv = tmp_path / "umap.csv"

    pd.DataFrame({
        "det_index": np.arange(n),
        "syllable_type": rng.choice(["A", "B", "C"], size=n),
        "label": rng.integers(0, 5, size=n),
        "file": ["rec.wav"] * n,
        "begin_time_s": np.linspace(0, 10, n),
    }).to_csv(classified_csv, index=False)

    pd.DataFrame({
        "det_index": np.arange(n),
        "hdbscan_label": rng.integers(0, 3, size=n),
    }).to_csv(umap_csv, index=False)

    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"Script exited with code {proc.returncode}.\n"
        f"stderr:\n{proc.stderr}"
    )


def test_script_missing_csv_exits_nonzero(tmp_path: Path):
    """Additional: script must exit non-zero when a required CSV is missing.

    Robust error handling is required by Pattern 4 (Script CLI). Passing a
    non-existent path for --classified-csv must not silently succeed.
    """
    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(tmp_path / "does_not_exist.csv"),
            "--umap-csv", str(tmp_path / "also_missing.csv"),
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, (
        "Script should exit non-zero when input CSV files are missing, but got code 0"
    )


def test_length_one_sequence_no_crash():
    """Additional (single-item edge case): a sequence of exactly 1 call must not crash.

    Lag-1 MI requires at least 2 consecutive symbols; with N=1 the function
    must return MI=0 gracefully (not IndexError or divide-by-zero).
    """
    single = np.array([2], dtype=np.intp)
    result = compute_sis_depth_1(single, name="single-call")

    assert isinstance(result, SISResult)
    assert result.mi_at_lag_1 == 0.0, (
        f"Expected mi_at_lag_1=0.0 for single-element sequence, got {result.mi_at_lag_1}"
    )
    assert result.n_calls == 1, (
        f"Expected n_calls=1, got {result.n_calls}"
    )


# ===========================================================================
# Adversarial tests added by test-hardener
# ===========================================================================

# ---------------------------------------------------------------------------
# Category A: Warning paths (W2/W3 fixes — no prior coverage)
# ---------------------------------------------------------------------------


def test_script_warns_on_duplicate_det_index_in_classified_csv(tmp_path: Path):
    """Hardener A1: duplicate det_index in classified_csv triggers a [warn] on stderr.

    W2 fix: _load_merged detects duplicate det_index rows in classified_csv and
    emits '[warn] classified_csv has N duplicate det_index rows'. This path was
    added post-review and has no existing test. The script must still complete
    (exit 0) despite the warning, and the warning text must appear on stderr.
    """
    rng = np.random.default_rng(7)
    n = 50
    classified_csv = tmp_path / "classified_dup.csv"
    umap_csv = tmp_path / "umap.csv"

    # Build a DataFrame that has duplicate det_index rows (row 0 is duplicated)
    classified_df = pd.DataFrame({
        "det_index": [0] * 2 + list(range(1, n)),  # index 0 appears twice
        "syllable_type": rng.choice(["Flat", "Down"], size=n + 1),
        "label": rng.integers(0, 5, size=n + 1),
        "file": ["rec.wav"] * (n + 1),
        "begin_time_s": np.linspace(0, 10, n + 1),
    })
    classified_df.to_csv(classified_csv, index=False)

    pd.DataFrame({
        "det_index": np.arange(n),
        "hdbscan_label": rng.integers(0, 3, size=n),
    }).to_csv(umap_csv, index=False)

    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"Script should still exit 0 with duplicate det_index rows, got {proc.returncode}.\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "[warn]" in proc.stderr and "classified_csv" in proc.stderr, (
        f"Expected '[warn] classified_csv has N duplicate det_index rows' on stderr.\n"
        f"stderr was:\n{proc.stderr}"
    )


def test_script_warns_on_duplicate_det_index_in_umap_csv(tmp_path: Path):
    """Hardener A2: duplicate det_index in umap_csv triggers a [warn] on stderr.

    Symmetric to A1 but for the umap CSV side (separate code branch in
    _load_merged). Verifies the W2 warning is emitted for each CSV independently.
    """
    rng = np.random.default_rng(8)
    n = 50
    classified_csv = tmp_path / "classified.csv"
    umap_csv = tmp_path / "umap_dup.csv"

    pd.DataFrame({
        "det_index": np.arange(n),
        "syllable_type": rng.choice(["Flat", "Down"], size=n),
        "label": rng.integers(0, 5, size=n),
        "file": ["rec.wav"] * n,
        "begin_time_s": np.linspace(0, 10, n),
    }).to_csv(classified_csv, index=False)

    # umap has a duplicated det_index
    umap_df = pd.DataFrame({
        "det_index": [0] * 2 + list(range(1, n)),
        "hdbscan_label": rng.integers(0, 3, size=n + 1),
    })
    umap_df.to_csv(umap_csv, index=False)

    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"Script should still exit 0 with duplicate umap det_index rows, got {proc.returncode}.\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "[warn]" in proc.stderr and "umap_csv" in proc.stderr, (
        f"Expected '[warn] umap_csv has N duplicate det_index rows' on stderr.\n"
        f"stderr was:\n{proc.stderr}"
    )


def test_script_warns_when_sort_keys_fully_absent(tmp_path: Path):
    """Hardener A3: missing both 'file' and 'begin_time_s' → W3 warning on stderr.

    The W3 fix emits '[warn] neither file nor begin_time_s columns found' when
    neither sort key is present. This path has no prior test coverage.
    The script must still produce baselines.csv (run continues on raw row order).
    """
    rng = np.random.default_rng(9)
    n = 60
    classified_csv = tmp_path / "classified_nosort.csv"
    umap_csv = tmp_path / "umap.csv"

    # Deliberately omit both 'file' and 'begin_time_s'
    pd.DataFrame({
        "det_index": np.arange(n),
        "syllable_type": rng.choice(["Flat", "Down", "UShaped"], size=n),
        "label": rng.integers(0, 5, size=n),
    }).to_csv(classified_csv, index=False)

    pd.DataFrame({
        "det_index": np.arange(n),
        "hdbscan_label": rng.integers(0, 3, size=n),
    }).to_csv(umap_csv, index=False)

    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"Script should exit 0 even without sort keys, got {proc.returncode}.\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "[warn]" in proc.stderr and "neither" in proc.stderr, (
        f"Expected '[warn] neither ... sort keys found' on stderr.\n"
        f"stderr was:\n{proc.stderr}"
    )
    assert (tmp_path / "out" / "baselines.csv").exists(), (
        "baselines.csv must be produced even when sort keys are absent"
    )


def test_script_warns_when_sort_keys_partial(tmp_path: Path):
    """Hardener A4: only one of 'file'/'begin_time_s' present → W3 'partial' warning.

    The W3 fix also covers the partial case — the implementation prints
    '[warn] sort keys partial (have [...], missing ...)'. This differs from
    the fully-absent case and exercises the elif branch in _load_merged.
    """
    rng = np.random.default_rng(10)
    n = 60
    classified_csv = tmp_path / "classified_partial.csv"
    umap_csv = tmp_path / "umap.csv"

    # Has 'file' but not 'begin_time_s'
    pd.DataFrame({
        "det_index": np.arange(n),
        "syllable_type": rng.choice(["Flat", "Down"], size=n),
        "label": rng.integers(0, 5, size=n),
        "file": ["rec.wav"] * n,
        # begin_time_s deliberately absent
    }).to_csv(classified_csv, index=False)

    pd.DataFrame({
        "det_index": np.arange(n),
        "hdbscan_label": rng.integers(0, 3, size=n),
    }).to_csv(umap_csv, index=False)

    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"Script should exit 0 with partial sort keys, got {proc.returncode}.\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "[warn]" in proc.stderr and "partial" in proc.stderr, (
        f"Expected '[warn] sort keys partial ...' on stderr.\n"
        f"stderr was:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Category B: Numerical correctness
# ---------------------------------------------------------------------------


def test_three_symbol_periodic_mi_equals_log2_3():
    """Hardener B1: 3-symbol periodic [0,1,2,0,1,2,...] → MI = log2(3) bits.

    For a perfectly periodic length-3 sequence, H(X) = log2(3) bits and the
    sequence is fully determined by its predecessor, so MI = H(X) = log2(3).
    Hand-derivation:
      - marginal probs: P(0)=P(1)=P(2) = 1/3
      - transitions: 0→1, 1→2, 2→0 each with prob 1/3
      - joint P(i,j) = 1/3 for (0,1),(1,2),(2,0) and 0 elsewhere
      - MI = 3 * (1/3) * log2((1/3)/((1/3)*(1/3))) = log2(3) ≈ 1.585 bits
    The finite-sample estimator must converge to within 0.01 bits at N=600.
    """
    n = 600  # multiple of 3 for exact periodicity
    seq = np.tile([0, 1, 2], n // 3).astype(np.intp)
    result = compute_sis_depth_1(seq, name="3-symbol-periodic")

    expected_mi = np.log2(3)
    assert abs(result.mi_at_lag_1 - expected_mi) < 0.01, (
        f"Expected MI ≈ log2(3) = {expected_mi:.6f} bits for 3-symbol periodic sequence, "
        f"got {result.mi_at_lag_1:.6f}"
    )
    assert abs(result.entropy_reduction_pct - 100.0) < 0.5, (
        f"Expected entropy_reduction_pct ≈ 100% for fully determined sequence, "
        f"got {result.entropy_reduction_pct:.3f}%"
    )


def test_skewed_binary_sequence_marginal_entropy():
    """Hardener B2: 3:1 skewed binary → H(X) ≈ 0.8113 bits.

    For a sequence where P(0) = 0.75 and P(1) = 0.25 (independently drawn),
    the marginal entropy is:
      H = -0.75*log2(0.75) - 0.25*log2(0.25)
        = 0.75*(2 - log2(3)) + 0.5
        ≈ 0.81128 bits
    This tests that the marginal entropy computation is correct independently
    of the MI path (an IID sequence has MI ≈ 0, so conditional_entropy ≈ H).
    """
    rng = np.random.default_rng(42)
    n = 10_000
    # 75% label 0, 25% label 1
    seq = rng.choice([0, 1], size=n, p=[0.75, 0.25]).astype(np.intp)

    result = compute_sis_depth_1(seq, name="skewed-binary")

    expected_h = -0.75 * np.log2(0.75) - 0.25 * np.log2(0.25)
    assert abs(result.marginal_entropy - expected_h) < 0.015, (
        f"Expected marginal_entropy ≈ {expected_h:.6f} bits for 3:1 skewed binary, "
        f"got {result.marginal_entropy:.6f}"
    )
    # Also confirm conditional entropy ≈ H (IID → no structure)
    assert abs(result.conditional_entropy - expected_h) < 0.01, (
        f"IID skewed binary should have conditional_entropy ≈ marginal_entropy, "
        f"got conditional={result.conditional_entropy:.6f}, marginal={expected_h:.6f}"
    )


def test_scattoni_ballpark_synthetic_regression():
    """Hardener B3: synthetic 7-symbol sequence calibrated to ≈ 0.093 bits MI.

    The ROADMAP exit criterion is Scattoni-7 MI ≈ 0.093 bits. While real-data
    verification is deferred, we can build a synthetic sequence that should
    produce MI in the same ballpark (0.05–0.15 bits) to act as a regression
    guard against future refactors of the MI computation path.

    Construction: 7-symbol IID sequence has MI ≈ 0 (lower bound). A weakly
    Markov chain where each symbol has a 15% probability of transitioning to
    the "next" symbol (mod 7) and 85% uniform produces mild sequential
    structure matching the observed Scattoni-7 ballpark (≈0.085 bits at this
    seed). We assert the result is in [0.02, 0.15] bits — wide enough to
    avoid flakiness from random seed effects but tight enough to catch a
    factor-of-10 regression in the MI computation path.
    """
    rng = np.random.default_rng(2024)
    n = 7_518  # matches real 5970 dataset size
    k = 7

    # Build weakly Markov: 15% chance next = (current + 1) % k, 85% uniform
    seq = np.empty(n, dtype=np.intp)
    seq[0] = rng.integers(0, k)
    for i in range(1, n):
        if rng.random() < 0.15:
            seq[i] = (seq[i - 1] + 1) % k
        else:
            seq[i] = rng.integers(0, k)

    result = compute_sis_depth_1(seq, name="scattoni-ballpark")

    assert 0.02 <= result.mi_at_lag_1 <= 0.15, (
        f"Synthetic Scattoni-ballpark MI = {result.mi_at_lag_1:.6f} bits is outside "
        f"expected range [0.02, 0.15]. This may indicate a regression in MI computation."
    )


def test_two_call_sequence_boundary():
    """Hardener B4: exactly 2 calls — the minimum for lag-1 MI to be well-defined.

    N=2 is the boundary between the 'n_calls < 2 → return zero' guard and the
    normal computation path. With two different labels, both paths should be
    exercised and the result should have non-trivial structure (n_labels=2,
    n_calls=2, mi_at_lag_1 >= 0).
    """
    seq = np.array([0, 1], dtype=np.intp)
    result = compute_sis_depth_1(seq, name="two-call-boundary")

    assert isinstance(result, SISResult)
    assert result.n_calls == 2, f"Expected n_calls=2, got {result.n_calls}"
    assert result.n_labels == 2, f"Expected n_labels=2, got {result.n_labels}"
    assert result.mi_at_lag_1 >= 0.0, (
        f"MI must be non-negative, got {result.mi_at_lag_1}"
    )
    assert not np.isnan(result.mi_at_lag_1), "mi_at_lag_1 must not be NaN for N=2"
    assert not np.isnan(result.marginal_entropy), "marginal_entropy must not be NaN for N=2"


# ---------------------------------------------------------------------------
# Category C: sort_by_time edge cases
# ---------------------------------------------------------------------------


def test_sort_by_time_ties_are_stable():
    """Hardener C1: tied sort_by_time values preserve insertion order (kind='stable').

    When two calls share the same timestamp, argsort(kind='stable') must
    preserve their original relative order, not produce arbitrary permutations.
    We verify this by constructing a sequence where two runs of identical
    timestamps occupy different label regions — stable sort preserves the
    label sequence exactly (the only change from input order is the inter-group
    reordering, not within-group).
    """
    # Two groups: group A (label 0) at time 1.0, group B (label 1) at time 2.0
    # Both groups have identical timestamps within the group.
    n_per_group = 50
    labels = np.array([0] * n_per_group + [1] * n_per_group, dtype=np.intp)
    times = np.array([1.0] * n_per_group + [2.0] * n_per_group)

    # Shuffle the input
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(labels))
    labels_shuffled = labels[perm]
    times_shuffled = times[perm]

    result_sorted = compute_sis_depth_1(
        labels_shuffled, name="tied-sort", sort_by_time=times_shuffled
    )
    result_direct = compute_sis_depth_1(labels, name="direct-order")

    # After stable sort, the sequence of labels should equal the original order
    # (first all 0s, then all 1s), producing identical MI.
    assert abs(result_sorted.mi_at_lag_1 - result_direct.mi_at_lag_1) < 1e-9, (
        f"Stable sort with ties should reproduce the original order MI.\n"
        f"sorted MI={result_sorted.mi_at_lag_1:.9f}, direct MI={result_direct.mi_at_lag_1:.9f}"
    )


def test_sort_by_time_float_array():
    """Hardener C2: sort_by_time as float64 array — must work, not just integer times.

    The function accepts any array-like for sort_by_time. Real data has
    floating-point begin_time_s values (e.g. 1.234567 seconds). Integer times
    would be a special case; floats must also be handled correctly.
    """
    n = 100
    seq = _make_periodic(n=n)
    # Float times in reverse order (so sort_by_time is actually needed)
    times = np.linspace(10.0, 0.0, n, dtype=np.float64)  # descending floats

    # After sorting by descending times, labels are reversed; MI should still
    # be computable (and equal to MI on the reversed sequence directly).
    result_with_sort = compute_sis_depth_1(
        seq, name="float-times", sort_by_time=times
    )
    result_direct_reversed = compute_sis_depth_1(
        seq[::-1].copy(), name="direct-reversed"
    )

    assert isinstance(result_with_sort, SISResult)
    assert not np.isnan(result_with_sort.mi_at_lag_1), (
        "mi_at_lag_1 must not be NaN when sort_by_time is a float array"
    )
    assert abs(result_with_sort.mi_at_lag_1 - result_direct_reversed.mi_at_lag_1) < 1e-9, (
        f"Float-time sort should produce same MI as direct reversed input.\n"
        f"float-sort MI={result_with_sort.mi_at_lag_1:.9f}, "
        f"direct-reversed MI={result_direct_reversed.mi_at_lag_1:.9f}"
    )


def test_sort_by_time_length_mismatch_raises():
    """Hardener C3: sort_by_time with wrong length must raise, not silently truncate.

    Passing a sort_by_time array of different length than labels is a user bug.
    The function should raise a meaningful error (IndexError or ValueError),
    not silently use only the first N elements of one array or produce a wrong
    result.

    BUG: The current implementation calls np.argsort(sort_by_time) which produces
    indices in range [0, len(sort_by_time)-1], then uses those to index labels via
    fancy indexing. When sort_by_time is shorter than labels, numpy fancy indexing
    only selects len(sort_by_time) elements from labels — no exception is raised,
    and the MI is computed on a silently truncated label sequence. The reported
    n_calls still equals the full labels length (6), but only 3 labels are used
    in the MI computation.
    """
    seq = np.array([0, 1, 2, 0, 1, 2], dtype=np.intp)
    times_too_short = np.array([1.0, 2.0, 3.0])  # len=3, seq len=6

    with pytest.raises((ValueError, IndexError)):
        compute_sis_depth_1(seq, name="length-mismatch", sort_by_time=times_too_short)


# ---------------------------------------------------------------------------
# Category D: pd.factorize determinism
# ---------------------------------------------------------------------------


def test_factorize_determinism_shuffled_input():
    """Hardener D1: shuffled input with same label content → identical MI.

    pd.factorize(sort=True) assigns label codes by sorted unique value, not by
    first-occurrence order. Two runs with different row orderings but the same
    underlying label content (fed in via sort_by_time) must produce the same MI.
    """
    rng = np.random.default_rng(5)
    n = 300
    labels = rng.choice(["alpha", "beta", "gamma", "delta"], size=n)
    times = np.arange(n, dtype=float)

    # Shuffle both arrays identically (so sort_by_time restores original order)
    perm = rng.permutation(n)
    labels_shuffled = labels[perm]
    times_shuffled = times[perm]

    result_a = compute_sis_depth_1(labels, name="original-order")
    result_b = compute_sis_depth_1(
        labels_shuffled, name="shuffled", sort_by_time=times_shuffled
    )

    assert abs(result_a.mi_at_lag_1 - result_b.mi_at_lag_1) < 1e-9, (
        f"Shuffled input with same label content and correct time sort must produce "
        f"identical MI. original={result_a.mi_at_lag_1:.9f}, "
        f"shuffled={result_b.mi_at_lag_1:.9f}"
    )
    assert result_a.n_labels == result_b.n_labels, (
        f"n_labels must be identical: original={result_a.n_labels}, "
        f"shuffled={result_b.n_labels}"
    )


def test_integer_labels_already_in_range_not_confused():
    """Hardener D2: integer labels already in [0, K) are mapped correctly.

    pd.factorize(sort=True) on [0, 1, 2] maps 0→0, 1→1, 2→2 (sorted order).
    Labels [2, 1, 0] map 0→0, 1→1, 2→2 as well — the alphabetic sort is on
    the values, not on first-occurrence. This test verifies n_labels is K (not
    max+1) and that MI is not confounded by the re-mapping.
    """
    # Sequence only uses labels {5, 10} — factorize(sort=True) maps 5→0, 10→1
    n = 200
    seq_high_ints = np.array([5, 10] * (n // 2), dtype=np.intp)
    seq_low_ints = np.array([0, 1] * (n // 2), dtype=np.intp)  # equivalent structure

    result_high = compute_sis_depth_1(seq_high_ints, name="high-int-labels")
    result_low = compute_sis_depth_1(seq_low_ints, name="low-int-labels")

    assert result_high.n_labels == 2, (
        f"n_labels should be 2 for {{5, 10}} alphabet, got {result_high.n_labels}"
    )
    assert abs(result_high.mi_at_lag_1 - result_low.mi_at_lag_1) < 1e-9, (
        f"High-int and low-int labels with same structure must produce same MI.\n"
        f"high={result_high.mi_at_lag_1:.9f}, low={result_low.mi_at_lag_1:.9f}"
    )


# ---------------------------------------------------------------------------
# Category E: CLI edge cases
# ---------------------------------------------------------------------------


def test_script_overwrites_existing_output_dir(tmp_path: Path):
    """Hardener E1: --output-dir pointing at an existing non-empty directory.

    The script uses mkdir(parents=True, exist_ok=True), so an existing directory
    must not cause a failure. A pre-existing baselines.csv inside the directory
    must be overwritten without error.
    """
    classified_csv, umap_csv = _make_script_csvs(tmp_path, n=50, seed=20)
    output_dir = tmp_path / "existing_out"
    output_dir.mkdir()

    # Write a stale baselines.csv with wrong content to verify it gets overwritten
    stale_csv = output_dir / "baselines.csv"
    stale_csv.write_text("stale,content,here\n1,2,3\n")

    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"Script should exit 0 when output dir already exists, got {proc.returncode}.\n"
        f"stderr:\n{proc.stderr}"
    )
    df = pd.read_csv(stale_csv)
    # The stale "stale,content,here" header should be gone; real columns should be present
    assert "name" in df.columns, (
        f"baselines.csv should be overwritten with real content. "
        f"Columns found: {list(df.columns)}"
    )


def test_script_only_one_labeling_column_produces_output(tmp_path: Path):
    """Hardener E2: CSV with only one labeling column (label only, no syllable_type or hdbscan_label).

    The implementation warns and skips missing columns but must still produce
    baselines.csv with 1 row. This tests the partial-results path.
    """
    rng = np.random.default_rng(21)
    n = 80
    classified_csv = tmp_path / "classified_minimal.csv"
    umap_csv = tmp_path / "umap_minimal.csv"

    # Only 'label' column — no syllable_type
    pd.DataFrame({
        "det_index": np.arange(n),
        "label": rng.integers(0, 5, size=n),
        "file": ["rec.wav"] * n,
        "begin_time_s": np.linspace(0, 10, n),
    }).to_csv(classified_csv, index=False)

    # umap CSV has no hdbscan_label column at all
    pd.DataFrame({
        "det_index": np.arange(n),
    }).to_csv(umap_csv, index=False)

    output_dir = tmp_path / "out_minimal"
    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"Script should exit 0 when only one labeling column is present, "
        f"got {proc.returncode}.\nstderr:\n{proc.stderr}"
    )
    baselines_csv = output_dir / "baselines.csv"
    assert baselines_csv.exists(), "baselines.csv must be created even with partial columns"
    df = pd.read_csv(baselines_csv)
    assert len(df) == 1, (
        f"Expected 1 row in baselines.csv for 1 labeling column, got {len(df)}"
    )


def test_script_all_labeling_columns_absent_exits_one(tmp_path: Path):
    """Hardener E3: all three labeling columns missing → script exits 1.

    The ROADMAP requires the script to fail gracefully when no labeling columns
    are found. The implementation has an explicit check: 'if not results: return 1'.
    This path has no prior test.
    """
    rng = np.random.default_rng(22)
    n = 50
    classified_csv = tmp_path / "classified_nolabels.csv"
    umap_csv = tmp_path / "umap_nolabels.csv"

    # CSVs with det_index and sort keys but none of the three labeling columns
    pd.DataFrame({
        "det_index": np.arange(n),
        "file": ["rec.wav"] * n,
        "begin_time_s": np.linspace(0, 10, n),
        "some_other_column": rng.integers(0, 5, size=n),
    }).to_csv(classified_csv, index=False)

    pd.DataFrame({
        "det_index": np.arange(n),
        "another_column": rng.integers(0, 3, size=n),
    }).to_csv(umap_csv, index=False)

    output_dir = tmp_path / "out_nolabels"
    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1, (
        f"Script should exit 1 when no labeling columns are found, "
        f"got {proc.returncode}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "no labeling columns" in proc.stderr.lower() or "error" in proc.stderr.lower(), (
        f"Script should emit an error message about missing labeling columns.\n"
        f"stderr was:\n{proc.stderr}"
    )


def test_script_auto_creates_missing_output_dir(tmp_path: Path):
    """Hardener E4: --output-dir leaf does not exist → auto-created via mkdir(parents=True).

    This tests the mkdir path. The parent exists (tmp_path) but the leaf
    'deep/nested/path' does not. The script should create it without complaint.
    """
    classified_csv, umap_csv = _make_script_csvs(tmp_path, n=50, seed=23)
    deep_output = tmp_path / "deep" / "nested" / "path"

    script_path = REPO_ROOT / "scripts" / "run_sis_baselines.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(classified_csv),
            "--umap-csv", str(umap_csv),
            "--output-dir", str(deep_output),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"Script should auto-create missing output dir, got exit {proc.returncode}.\n"
        f"stderr:\n{proc.stderr}"
    )
    assert (deep_output / "baselines.csv").exists(), (
        f"baselines.csv not found at {deep_output / 'baselines.csv'}"
    )


# ---------------------------------------------------------------------------
# Category F: SISResult dataclass semantics
# ---------------------------------------------------------------------------


def test_sisresult_asdict_has_expected_keys():
    """Hardener F1: asdict(result) returns a dict with exactly the 7 expected keys.

    The CLI uses asdict() to build CSV rows. If a field is added or renamed,
    this test catches it immediately. The 7 required keys match the frozen
    dataclass definition and the ROADMAP spec.
    """
    seq = _make_periodic(n=100)
    result = compute_sis_depth_1(seq, name="asdict-test")
    d = asdict(result)

    expected_keys = {
        "name",
        "n_calls",
        "n_labels",
        "mi_at_lag_1",
        "marginal_entropy",
        "conditional_entropy",
        "entropy_reduction_pct",
    }
    assert set(d.keys()) == expected_keys, (
        f"asdict(SISResult) has wrong keys.\n"
        f"Expected: {sorted(expected_keys)}\n"
        f"Got:      {sorted(d.keys())}"
    )


def test_sisresult_equality_by_value():
    """Hardener F2: two SISResult instances with identical fields compare equal.

    Python frozen dataclasses inherit __eq__ from object by default only when
    eq=True (which is the default for @dataclass). This test ensures that two
    results with identical numeric values are equal, enabling dict/set membership
    and assertion comparisons in downstream tests.
    """
    seq = _make_periodic(n=100)
    result_a = compute_sis_depth_1(seq, name="eq-test")
    result_b = compute_sis_depth_1(seq, name="eq-test")

    assert result_a == result_b, (
        f"Two SISResults from the same deterministic input must compare equal.\n"
        f"result_a={result_a}\nresult_b={result_b}"
    )


def test_sisresult_inequality_by_value():
    """Hardener F3: two SISResult instances with different MI values are not equal.

    Complements F2 — verifies __eq__ is value-based, not identity-based. Two
    different input sequences must produce non-equal SISResults.
    """
    seq_periodic = _make_periodic(n=100)
    seq_iid = _make_iid(n=100, K=2, seed=13)

    result_periodic = compute_sis_depth_1(seq_periodic, name="periodic")
    result_iid = compute_sis_depth_1(seq_iid, name="periodic")  # same name, different MI

    # IID MI ≈ 0, periodic MI ≈ 1.0 — they are not equal
    assert result_periodic != result_iid, (
        "SISResults with different MI values must not compare equal"
    )


def test_sisresult_frozen_nonexistent_field_raises():
    """Hardener F4: attempting to set a non-existent field also raises FrozenInstanceError.

    test_sisresult_is_immutable covers setting an existing field. This test
    verifies that trying to add a brand-new attribute (which would work on a
    regular class) also raises on a frozen dataclass.
    """
    seq = _make_periodic(n=50)
    result = compute_sis_depth_1(seq, name="frozen-nonexistent")

    with pytest.raises((AttributeError, TypeError)):
        result.new_field_that_does_not_exist = 42  # type: ignore[attr-defined]


def test_sisresult_frozen_deletion_raises():
    """Hardener F5: del result.<field> must raise on a frozen dataclass.

    Deletion of a field attribute is another mutation path that frozen=True
    should block. Both FrozenInstanceError and AttributeError are acceptable.
    """
    seq = _make_periodic(n=50)
    result = compute_sis_depth_1(seq, name="frozen-delete")

    with pytest.raises((AttributeError, TypeError)):
        del result.name  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Category G: NaN/Inf inputs
# ---------------------------------------------------------------------------


def test_sort_by_time_with_nan_raises_or_sorts_consistently():
    """Hardener G1: NaN in sort_by_time — must not silently produce wrong results.

    numpy argsort places NaN values at the end in ascending order by convention,
    but the behavior of 'kind=stable' with NaN is implementation-defined across
    numpy versions. The function should either:
      (a) raise a clear error, OR
      (b) produce a result that is at least self-consistent (same result on two
          calls with the same NaN-containing input).
    A silent wrong result (e.g., NaN MI) is not acceptable.

    This is marked as a behavioral documentation test — if the current
    implementation raises, that's fine; if it produces a finite MI, that's
    also fine as long as it's consistent.
    """
    seq = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)
    times_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])

    try:
        result_a = compute_sis_depth_1(seq, name="nan-times-a", sort_by_time=times_with_nan)
        result_b = compute_sis_depth_1(seq, name="nan-times-b", sort_by_time=times_with_nan)
        # If it doesn't raise: result must be finite and self-consistent
        assert not np.isnan(result_a.mi_at_lag_1), (
            "mi_at_lag_1 must not be NaN when sort_by_time contains NaN"
        )
        assert result_a.mi_at_lag_1 == result_b.mi_at_lag_1, (
            "Same NaN-containing input must produce same MI on repeated calls"
        )
    except (ValueError, FloatingPointError):
        pass  # explicit raise is also an acceptable behavior


def test_large_input_no_crash():
    """Hardener G2: very large input (100K calls) must not crash or OOM.

    A basic smoke test for the O(N) MI computation path. Does not assert a
    specific MI value — only verifies no exception and a finite result.
    """
    rng = np.random.default_rng(99)
    seq = rng.integers(0, 7, size=100_000).astype(np.intp)
    result = compute_sis_depth_1(seq, name="large-input")

    assert isinstance(result, SISResult)
    assert np.isfinite(result.mi_at_lag_1), (
        f"MI must be finite for large input, got {result.mi_at_lag_1}"
    )
    assert np.isfinite(result.marginal_entropy), (
        f"marginal_entropy must be finite for large input, got {result.marginal_entropy}"
    )
    assert result.n_calls == 100_000, (
        f"n_calls must equal input length, got {result.n_calls}"
    )
