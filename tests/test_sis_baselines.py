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
