"""Tests for scripts/run_sis_benchmark.py — written by test-architect BEFORE implementation.

This module is the Phase 17.9 driver script that aggregates all per-method SIS labeling
results into a single comparison table, plots, and decision report.

ROADMAP test plan coverage (module 17.9):
  1. Aggregator assembles master table from synthetic per-method CSVs -> test_aggregator_assembles_master_table
  2. Benchmark bar chart includes Hertz reference lines at 0.10, 0.13, 0.22, 0.23 -> test_benchmark_bar_png_includes_hertz_reference_lines
  3. benchmark_by_k.png produced only for methods with multiple K values -> test_benchmark_by_k_produced_only_for_multi_k_methods
  4. Confusion matrix only computed for methods with matching call_ids -> test_confusion_matrix_only_for_matching_call_ids
  5. Report.md includes the winner name and MI value -> test_report_md_includes_winner_and_mi_value
  6. Script runs end-to-end on synthetic results_root with 2 methods -> test_end_to_end_two_methods_all_outputs_created
  7. Missing per-method result -> warn and skip, do not crash -> test_missing_result_warns_and_skips_without_crash

Additional coverage (recurring gap patterns):
  - Empty results_root (no subdirs populated) -> test_empty_results_root_exits_gracefully
  - benchmark.csv column schema validation -> test_benchmark_csv_has_required_columns
  - Single method only -> test_single_method_produces_valid_outputs
  - output-dir is created if it doesn't exist -> test_output_dir_is_created_if_missing
  - benchmark.csv row ordering (descending MI) -> test_benchmark_csv_sorted_descending_by_mi

Total: 12 tests (7 from ROADMAP, 5 additional)

Notes:
  - All tests invoke the script via subprocess.run so failures are ImportError /
    FileNotFoundError / non-zero returncode before the script is written.
  - Synthetic results_root contains the four expected subdirs:
    sis_baselines/, imsa/, cluster_sweep/, sim/
  - Per-method CSVs contain: call_id, file, begin_time_s, label (integer labels)
  - SIS result CSVs (benchmark.csv) have columns:
    [name, family, K, mi_at_lag_1, marginal_entropy, conditional_entropy, entropy_reduction_pct]
"""

from __future__ import annotations

import sys
import subprocess
import textwrap
from pathlib import Path

import pandas as pd
import pytest

# Pattern 8 — import bootstrap
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "run_sis_benchmark.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BENCHMARK_COLUMNS = [
    "name",
    "family",
    "K",
    "mi_at_lag_1",
    "marginal_entropy",
    "conditional_entropy",
    "entropy_reduction_pct",
]

EXPECTED_PNG_OUTPUTS = [
    "benchmark_bar.png",
    "benchmark_by_k.png",
    "label_distribution_grid.png",
    "confusion_top3.png",
]


def _write_labeling_csv(
    path: Path,
    name: str,
    n_calls: int = 20,
    n_labels: int = 4,
    seed: int = 42,
) -> None:
    """Write a synthetic per-method labeling CSV to *path*.

    Columns: call_id, file, begin_time_s, label (integers 0..n_labels-1).
    """
    import numpy as np

    rng = numpy_rng(seed)
    labels = rng.integers(0, n_labels, size=n_calls)
    begin_times = sorted(rng.uniform(0.0, 60.0, size=n_calls))
    rows = {
        "call_id": [f"{name}_{i:04d}" for i in range(n_calls)],
        "file": [f"recording_{i % 3:02d}.wav" for i in range(n_calls)],
        "begin_time_s": begin_times,
        "label": labels.tolist(),
    }
    pd.DataFrame(rows).to_csv(path, index=False)


def numpy_rng(seed: int):
    """Return a numpy default_rng for reproducible synthetic data."""
    import numpy as np

    return np.random.default_rng(seed)


def _write_sis_result_csv(
    path: Path,
    name: str,
    family: str,
    K: int,
    mi_at_lag_1: float,
    marginal_entropy: float = 2.0,
) -> None:
    """Write a synthetic per-method SIS result CSV (already-computed metrics)."""
    conditional_entropy = marginal_entropy - mi_at_lag_1
    entropy_reduction_pct = (mi_at_lag_1 / marginal_entropy) * 100 if marginal_entropy > 0 else 0.0
    row = {
        "name": [name],
        "family": [family],
        "K": [K],
        "mi_at_lag_1": [mi_at_lag_1],
        "marginal_entropy": [marginal_entropy],
        "conditional_entropy": [conditional_entropy],
        "entropy_reduction_pct": [entropy_reduction_pct],
    }
    pd.DataFrame(row).to_csv(path, index=False)


def _build_minimal_results_root(tmp_path: Path, *, seed: int = 0) -> Path:
    """Create a minimal results_root with two methods (one in sis_baselines/, one in imsa/).

    Returns the results_root path.
    """
    results_root = tmp_path / "results_root"
    results_root.mkdir()

    for subdir in ("sis_baselines", "imsa", "cluster_sweep", "sim"):
        (results_root / subdir).mkdir()

    # Method A: scattoni-7 in sis_baselines
    _write_labeling_csv(
        results_root / "sis_baselines" / "scattoni7_labels.csv",
        name="scattoni-7",
        n_calls=30,
        n_labels=7,
        seed=seed,
    )

    # Method B: imsa in imsa/
    _write_labeling_csv(
        results_root / "imsa" / "imsa_labels.csv",
        name="imsa",
        n_calls=30,
        n_labels=5,
        seed=seed + 1,
    )

    return results_root


def _run_script(
    results_root: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Invoke run_sis_benchmark.py via subprocess and return the result."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--results-root", str(results_root),
        "--output-dir", str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=check,
    )


# ---------------------------------------------------------------------------
# ROADMAP Test Plan Tests
# ---------------------------------------------------------------------------


def test_aggregator_assembles_master_table(tmp_path: Path) -> None:
    """Verify ROADMAP item 1: aggregator assembles benchmark.csv with one row per method.

    Spec: Pipeline step 3 produces results/sis_benchmark/benchmark.csv with columns
    [name, family, K, mi_at_lag_1, marginal_entropy, conditional_entropy,
    entropy_reduction_pct]. The table must have exactly one row per recognised method.
    """
    results_root = _build_minimal_results_root(tmp_path)
    output_dir = tmp_path / "benchmark_out"

    proc = _run_script(results_root, output_dir)

    # If the script does not exist yet, returncode will be non-zero (e.g. FileNotFoundError)
    # After implementation this must be 0.
    assert proc.returncode == 0, (
        f"Script exited with code {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    csv_path = output_dir / "benchmark.csv"
    assert csv_path.exists(), f"benchmark.csv not found at {csv_path}"

    df = pd.read_csv(csv_path)
    # Two methods were provided; each must produce exactly one row
    assert len(df) == 2, f"Expected 2 rows (one per method), got {len(df)}"

    # All required columns present
    missing = [c for c in BENCHMARK_COLUMNS if c not in df.columns]
    assert not missing, f"benchmark.csv missing columns: {missing}"


def test_benchmark_bar_png_includes_hertz_reference_lines(tmp_path: Path) -> None:
    """Verify ROADMAP item 2: benchmark_bar.png is produced and the report/CSV encodes
    the four Hertz reference values (0.10, 0.13, 0.22, 0.23 bits).

    Spec: 'benchmark_bar.png: bar chart … with horizontal reference lines at
    Hertz's 0.10/0.13/0.22/0.23 values'.  PNG internals are opaque, so we verify
    (a) the PNG file exists and has non-zero size, and (b) report.md or a companion
    metadata file mentions the four reference values, confirming the script knows about
    them. Full pixel-level content validation is deferred to the test-hardener.
    """
    results_root = _build_minimal_results_root(tmp_path, seed=10)
    output_dir = tmp_path / "benchmark_out"

    proc = _run_script(results_root, output_dir)
    assert proc.returncode == 0, (
        f"Script exited {proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    bar_png = output_dir / "benchmark_bar.png"
    assert bar_png.exists(), "benchmark_bar.png was not created"
    assert bar_png.stat().st_size > 0, "benchmark_bar.png is empty (0 bytes)"

    # The Hertz reference values must appear somewhere in the report so a human
    # and downstream tools know which lines were drawn.
    report_path = output_dir / "report.md"
    assert report_path.exists(), "report.md was not created"
    report_text = report_path.read_text()

    hertz_values = ["0.10", "0.13", "0.22", "0.23"]
    missing_values = [v for v in hertz_values if v not in report_text]
    assert not missing_values, (
        f"report.md does not mention Hertz reference values: {missing_values}\n"
        f"(Expected lines at 0.10, 0.13, 0.22, 0.23 to be cited)"
    )


def test_benchmark_by_k_produced_only_for_multi_k_methods(tmp_path: Path) -> None:
    """Verify ROADMAP item 3: benchmark_by_k.png is created when the results_root
    contains cluster_sweep results with multiple K values (lines per feature source).

    The ROADMAP spec states this plot is for cluster_sweep methods only.  We test
    both that (a) the PNG exists when multi-K cluster_sweep data is present, and
    (b) the script does NOT crash when the cluster_sweep directory is empty.
    """
    results_root = _build_minimal_results_root(tmp_path, seed=20)

    # Add cluster_sweep data with two different K values (K=5 and K=10)
    sweep_dir = results_root / "cluster_sweep"
    _write_labeling_csv(sweep_dir / "sweep_k5_labels.csv", name="sweep-k5", n_calls=30, n_labels=5, seed=21)
    _write_labeling_csv(sweep_dir / "sweep_k10_labels.csv", name="sweep-k10", n_calls=30, n_labels=10, seed=22)

    output_dir = tmp_path / "benchmark_out"
    proc = _run_script(results_root, output_dir)
    assert proc.returncode == 0, (
        f"Script failed with multi-K cluster_sweep data.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    by_k_png = output_dir / "benchmark_by_k.png"
    assert by_k_png.exists(), (
        "benchmark_by_k.png should exist when cluster_sweep has multiple K values"
    )
    assert by_k_png.stat().st_size > 0, "benchmark_by_k.png exists but is empty"

    # --- Variant: empty cluster_sweep directory ---
    results_root2 = tmp_path / "results_root_no_sweep"
    results_root2.mkdir()
    for subdir in ("sis_baselines", "imsa", "cluster_sweep", "sim"):
        (results_root2 / subdir).mkdir()
    _write_labeling_csv(results_root2 / "sis_baselines" / "scattoni7_labels.csv",
                        name="scattoni-7", n_calls=20, n_labels=7, seed=23)
    output_dir2 = tmp_path / "benchmark_out_no_sweep"

    proc2 = _run_script(results_root2, output_dir2)
    assert proc2.returncode == 0, (
        "Script should not crash when cluster_sweep directory is empty; "
        f"got returncode {proc2.returncode}"
    )


def test_confusion_matrix_only_for_methods_with_matching_call_ids(tmp_path: Path) -> None:
    """Verify ROADMAP item 4: confusion/ARI matrix is only computed for method pairs
    that share matching call_ids; methods without overlapping call_ids are skipped
    without crashing.

    We provide three methods:
    - method_a: call_ids 0..19
    - method_b: call_ids 0..19  (overlapping — confusion computable)
    - method_c: call_ids 100..119  (disjoint — no overlap with a or b)
    """
    results_root = tmp_path / "results_root"
    results_root.mkdir()
    for subdir in ("sis_baselines", "imsa", "cluster_sweep", "sim"):
        (results_root / subdir).mkdir()

    # Method A and B share call IDs
    def _write_with_ids(path: Path, name: str, call_ids: list[int], n_labels: int, seed: int) -> None:
        import numpy as np
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, n_labels, size=len(call_ids))
        begin_times = sorted(rng.uniform(0.0, 60.0, size=len(call_ids)))
        pd.DataFrame({
            "call_id": [f"global_{i:04d}" for i in call_ids],
            "file": ["rec.wav"] * len(call_ids),
            "begin_time_s": begin_times,
            "label": labels.tolist(),
        }).to_csv(path, index=False)

    _write_with_ids(results_root / "sis_baselines" / "method_a.csv", "method_a",
                    list(range(20)), n_labels=4, seed=30)
    _write_with_ids(results_root / "imsa" / "method_b.csv", "method_b",
                    list(range(20)), n_labels=5, seed=31)
    _write_with_ids(results_root / "sim" / "method_c.csv", "method_c",
                    list(range(100, 120)), n_labels=3, seed=32)

    output_dir = tmp_path / "benchmark_out"
    proc = _run_script(results_root, output_dir)

    # Script must not crash regardless of which pairs overlap
    assert proc.returncode == 0, (
        f"Script crashed with disjoint call_ids present.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # confusion_top3.png should still be produced (for the overlapping pair)
    confusion_png = output_dir / "confusion_top3.png"
    assert confusion_png.exists(), (
        "confusion_top3.png should be produced for methods that do share call_ids"
    )
    assert confusion_png.stat().st_size > 0, "confusion_top3.png is empty"


def test_report_md_includes_winner_and_mi_value(tmp_path: Path) -> None:
    """Verify ROADMAP item 5: report.md names the winning method and states its MI value.

    Spec: 'report.md with winner + runner-up, MI values, interpretation in one paragraph'.
    We create two methods with known MI values and verify the winner appears by name and
    that a numeric MI value (in floating-point text) is present in the report body.
    """
    results_root = _build_minimal_results_root(tmp_path, seed=40)
    output_dir = tmp_path / "benchmark_out"

    proc = _run_script(results_root, output_dir)
    assert proc.returncode == 0, (
        f"Script failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    report_path = output_dir / "report.md"
    assert report_path.exists(), "report.md not found"

    report_text = report_path.read_text()
    assert len(report_text.strip()) > 0, "report.md is empty"

    # Report must contain at least one floating-point number (the winning MI value)
    import re
    floats_found = re.findall(r"\d+\.\d+", report_text)
    assert floats_found, "report.md contains no floating-point numbers — MI value missing"

    # Report must contain at least one of the method names we provided
    assert ("scattoni" in report_text.lower() or "imsa" in report_text.lower()), (
        "report.md does not mention any of the method names from the synthetic input; "
        "winner name must appear in the report"
    )


def test_end_to_end_two_methods_all_outputs_created(tmp_path: Path) -> None:
    """Verify ROADMAP item 6: script runs end-to-end with 2 methods and creates all
    four PNG outputs, benchmark.csv, and report.md.

    This is the primary smoke test: if this passes, the basic pipeline is wired.
    """
    results_root = _build_minimal_results_root(tmp_path, seed=50)
    output_dir = tmp_path / "benchmark_out"

    proc = _run_script(results_root, output_dir)
    assert proc.returncode == 0, (
        f"Script exited with code {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # benchmark.csv
    csv_path = output_dir / "benchmark.csv"
    assert csv_path.exists(), "benchmark.csv not produced"
    assert csv_path.stat().st_size > 0, "benchmark.csv is empty"

    # report.md
    report_path = output_dir / "report.md"
    assert report_path.exists(), "report.md not produced"
    assert report_path.stat().st_size > 0, "report.md is empty"

    # All four PNGs — note: benchmark_by_k.png may be skipped if no multi-K data,
    # so we only strictly require the other three; by_k is checked for existence
    # in a dedicated test above.
    required_pngs = ["benchmark_bar.png", "label_distribution_grid.png", "confusion_top3.png"]
    for png_name in required_pngs:
        p = output_dir / png_name
        assert p.exists(), f"{png_name} not produced"
        assert p.stat().st_size > 0, f"{png_name} exists but has 0 bytes"


def test_missing_result_warns_and_skips_without_crash(tmp_path: Path) -> None:
    """Verify ROADMAP item 7: when a per-method CSV is malformed or missing, the script
    emits a warning and continues processing remaining methods without crashing.

    We create a results_root where the imsa/ directory exists but contains a corrupt
    CSV (bad columns) and no other CSV, alongside a valid sis_baselines/ method.
    The script must exit 0 and still produce benchmark.csv for the valid method.
    """
    results_root = tmp_path / "results_root"
    results_root.mkdir()
    for subdir in ("sis_baselines", "imsa", "cluster_sweep", "sim"):
        (results_root / subdir).mkdir()

    # Valid method
    _write_labeling_csv(
        results_root / "sis_baselines" / "scattoni7_labels.csv",
        name="scattoni-7",
        n_calls=20,
        n_labels=7,
        seed=60,
    )

    # Corrupt / incompatible CSV in imsa/ (missing required columns)
    corrupt_csv = results_root / "imsa" / "corrupt_labels.csv"
    corrupt_csv.write_text("not_a,valid,csv,schema\n1,2,3,4\n")

    output_dir = tmp_path / "benchmark_out"
    proc = _run_script(results_root, output_dir)

    # Must not crash
    assert proc.returncode == 0, (
        "Script crashed when encountering a corrupt per-method CSV.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # benchmark.csv must still exist (for the valid method)
    csv_path = output_dir / "benchmark.csv"
    assert csv_path.exists(), "benchmark.csv not produced even for the valid method"
    df = pd.read_csv(csv_path)
    assert len(df) >= 1, "benchmark.csv has no rows; valid method was not processed"

    # A warning or notice must have been emitted (stderr or stdout)
    combined_output = (proc.stdout + proc.stderr).lower()
    assert any(kw in combined_output for kw in ("warn", "skip", "error", "missing", "corrupt", "failed")), (
        "Script produced no warning message when a corrupt CSV was encountered; "
        "the spec requires a warning + skip behaviour"
    )


# ---------------------------------------------------------------------------
# Additional gap-pattern tests
# ---------------------------------------------------------------------------


def test_empty_results_root_exits_gracefully(tmp_path: Path) -> None:
    """Additional: when results_root has all four subdirs but no CSV files at all,
    the script should exit 0 and produce benchmark.csv (even if it has 0 data rows).
    It must not raise an unhandled exception.
    """
    results_root = tmp_path / "empty_root"
    results_root.mkdir()
    for subdir in ("sis_baselines", "imsa", "cluster_sweep", "sim"):
        (results_root / subdir).mkdir()

    output_dir = tmp_path / "benchmark_out_empty"
    proc = _run_script(results_root, output_dir)

    # Primary assertion: script must exit 0 — this fails before implementation.
    assert proc.returncode == 0, (
        f"Script exited with code {proc.returncode} on empty results_root.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # Must not raise an unhandled Python exception
    assert "Traceback" not in proc.stderr, (
        "Script raised an unhandled exception on empty results_root.\n"
        f"STDERR:\n{proc.stderr}"
    )

    # benchmark.csv must be produced (column headers at minimum)
    csv_path = output_dir / "benchmark.csv"
    assert csv_path.exists(), (
        "benchmark.csv was not created even for an empty results_root; "
        "the output file should always be initialised"
    )


def test_benchmark_csv_has_required_columns(tmp_path: Path) -> None:
    """Additional: benchmark.csv must contain all seven columns defined in the spec,
    regardless of how many rows are present.

    Spec columns: [name, family, K, mi_at_lag_1, marginal_entropy,
    conditional_entropy, entropy_reduction_pct].
    """
    results_root = _build_minimal_results_root(tmp_path, seed=70)
    output_dir = tmp_path / "benchmark_out"

    proc = _run_script(results_root, output_dir)
    assert proc.returncode == 0, f"Script failed: {proc.stderr}"

    csv_path = output_dir / "benchmark.csv"
    assert csv_path.exists(), "benchmark.csv not found"

    df = pd.read_csv(csv_path)
    for col in BENCHMARK_COLUMNS:
        assert col in df.columns, (
            f"Required column '{col}' missing from benchmark.csv. "
            f"Present columns: {list(df.columns)}"
        )


def test_single_method_produces_valid_outputs(tmp_path: Path) -> None:
    """Additional: with only one valid method, the script should still produce benchmark.csv
    (with 1 row), benchmark_bar.png, and report.md. Confusion matrix and by-K plots
    can be skipped, but must not crash.
    """
    results_root = tmp_path / "single_method_root"
    results_root.mkdir()
    for subdir in ("sis_baselines", "imsa", "cluster_sweep", "sim"):
        (results_root / subdir).mkdir()

    _write_labeling_csv(
        results_root / "sis_baselines" / "scattoni7_labels.csv",
        name="scattoni-7",
        n_calls=25,
        n_labels=7,
        seed=80,
    )

    output_dir = tmp_path / "single_out"
    proc = _run_script(results_root, output_dir)
    assert proc.returncode == 0, (
        f"Script crashed with a single method.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    csv_path = output_dir / "benchmark.csv"
    assert csv_path.exists(), "benchmark.csv not produced for single-method run"
    df = pd.read_csv(csv_path)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"

    bar_png = output_dir / "benchmark_bar.png"
    assert bar_png.exists(), "benchmark_bar.png not produced for single-method run"

    report_path = output_dir / "report.md"
    assert report_path.exists(), "report.md not produced for single-method run"


def test_output_dir_is_created_if_missing(tmp_path: Path) -> None:
    """Additional: --output-dir should be auto-created if it does not exist.

    If the implementer forgets to call output_dir.mkdir(parents=True, exist_ok=True),
    the script will crash with FileNotFoundError.
    """
    results_root = _build_minimal_results_root(tmp_path, seed=90)
    # Deliberately specify a nested path that does not exist
    output_dir = tmp_path / "deep" / "nested" / "benchmark_out"

    assert not output_dir.exists(), "Precondition: output_dir must not exist before the test"

    proc = _run_script(results_root, output_dir)
    assert proc.returncode == 0, (
        f"Script failed when output_dir didn't exist yet.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert output_dir.exists(), "output_dir was not created by the script"


def test_benchmark_csv_sorted_descending_by_mi(tmp_path: Path) -> None:
    """Additional: benchmark.csv rows should be ordered by mi_at_lag_1 descending
    (best method first) so downstream tooling and humans can read the winner from row 0.

    We create three methods with distinct MI levels and verify ordering.
    """
    results_root = tmp_path / "sorted_root"
    results_root.mkdir()
    for subdir in ("sis_baselines", "imsa", "cluster_sweep", "sim"):
        (results_root / subdir).mkdir()

    # Three methods with different label counts (proxy for different MI levels)
    # Deterministic periodic sequences produce distinct MI values.
    import numpy as np

    def _write_periodic(path: Path, name: str, period: int, n_calls: int = 40) -> None:
        labels = [i % period for i in range(n_calls)]
        begin_times = [i * 0.5 for i in range(n_calls)]
        pd.DataFrame({
            "call_id": [f"{name}_{i:04d}" for i in range(n_calls)],
            "file": ["rec.wav"] * n_calls,
            "begin_time_s": begin_times,
            "label": labels,
        }).to_csv(path, index=False)

    # Period 2 (AB pattern) has high MI; period 5 moderate; random low
    _write_periodic(results_root / "sis_baselines" / "period2.csv", "period2", period=2)
    _write_periodic(results_root / "imsa" / "period5.csv", "period5", period=5)

    rng = np.random.default_rng(99)
    pd.DataFrame({
        "call_id": [f"rand_{i:04d}" for i in range(40)],
        "file": ["rec.wav"] * 40,
        "begin_time_s": sorted(rng.uniform(0.0, 20.0, size=40)),
        "label": rng.integers(0, 10, size=40).tolist(),
    }).to_csv(results_root / "sim" / "random_labels.csv", index=False)

    output_dir = tmp_path / "sorted_out"
    proc = _run_script(results_root, output_dir)
    assert proc.returncode == 0, (
        f"Script failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    csv_path = output_dir / "benchmark.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

    mi_values = df["mi_at_lag_1"].tolist()
    assert mi_values == sorted(mi_values, reverse=True), (
        f"benchmark.csv is not sorted descending by mi_at_lag_1: {mi_values}"
    )
