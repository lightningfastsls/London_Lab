"""Tests for ``scripts/cnn_download_vocalmat_sample.py`` (Module 18.2a).

The download script uses dependency injection via ``source_factory`` so
the tests run without any OSF / network access. ``FakeVocalMatSource``
mimics the real :class:`OSFVocalMatSource` enough to exercise
enumeration, sampling, manifest writing, dry-run, and idempotency paths.

The three ROADMAP §18.2a test items are covered by:

- ``--dry-run`` produces expected stdout
  → ``test_dry_run_lists_plan_without_fetching``
- Manifest CSV has correct columns and balanced counts
  → ``test_manifest_columns_and_balance``
- Re-run is idempotent (skips already-downloaded files)
  → ``test_rerun_is_idempotent``
"""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable

import pytest

# Import the script as a module by file path (the script lives in scripts/,
# not in src/, so a plain ``import`` won't work). Doing this once at import
# time keeps each test fast.
REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "cnn_download_vocalmat_sample.py"
_spec = importlib.util.spec_from_file_location("dl_vm_sample", _SCRIPT_PATH)
assert _spec and _spec.loader
dl = importlib.util.module_from_spec(_spec)
sys.modules["dl_vm_sample"] = dl
_spec.loader.exec_module(dl)


# ---------------------------------------------------------------------------
# Fake source — never touches OSF
# ---------------------------------------------------------------------------


def _make_fake_entries(class_name: str, n: int) -> list[dl.FileEntry]:
    """Build synthetic FileEntry rows in the same shape OSF returns."""
    return [
        dl.FileEntry(
            osf_path=f"/fake/{class_name}/{i:04d}",
            name=f"{i}_{class_name}_isolation1_5662F.png",
            size_bytes=1024 + i,
            source_recording="5662F",
        )
        for i in range(n)
    ]


class FakeVocalMatSource:
    """In-memory stand-in for :class:`OSFVocalMatSource`.

    ``class_sizes`` maps class names → total file counts. ``download``
    just writes a small known payload to ``dest`` so the test can verify
    files exist after a run.
    """

    PAYLOAD = b"FAKE PNG\x89PNG"

    def __init__(self, class_sizes: dict[str, int]) -> None:
        self._entries = {
            cls: _make_fake_entries(cls, n) for cls, n in class_sizes.items()
        }
        self.download_calls = 0

    def list_files(self, class_name: str) -> list[dl.FileEntry]:
        return list(self._entries[class_name])

    def download(self, entry: dl.FileEntry, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.PAYLOAD)
        self.download_calls += 1


@pytest.fixture()
def fake_source_full():
    """Class sizes that mirror the real OSF totals — closely enough that
    ``mult_steps`` exercises the "class smaller than cap" branch."""
    sizes = {
        "noise": 1352,
        "step_up": 1814,
        "down_fm": 1775,
        "short": 1713,
        "chevron": 1594,
        "up_fm": 1190,
        "flat": 1133,
        "two_steps": 701,
        "step_down": 389,
        "complex": 350,
        "rev_chevron": 136,
        "mult_steps": 74,
    }
    return FakeVocalMatSource(sizes)


@pytest.fixture()
def fake_source_tiny():
    """All classes have exactly 5 files each — keeps tests fast and lets
    a 200-cap fall under every class so the sample equals the full set."""
    return FakeVocalMatSource({c: 5 for c in dl.GRIMSLEY_OSF_CLASSES})


# ---------------------------------------------------------------------------
# Tests — three ROADMAP-prescribed checks plus a few sanity tests
# ---------------------------------------------------------------------------


def test_parse_source_recording_isolation_pattern():
    """The 'isolation' filename pattern yields the trailing animal ID."""
    assert dl._parse_source_recording("74_7_isolation1_5662F.png") == "5662F"
    assert dl._parse_source_recording("15_3_isolation1_5357M.png") == "5357M"


def test_parse_source_recording_fallback_pattern():
    """Filenames without the isolation pattern fall back to suffix-join."""
    assert dl._parse_source_recording("53_4829_Control_Baseline.png") == \
        "4829_Control_Baseline"


def test_parse_source_recording_unknown():
    """Filenames with no parseable structure yield 'unknown'."""
    assert dl._parse_source_recording("oneword.png") == "unknown"


def test_sample_per_class_caps_at_n(fake_source_full):
    """Sampling caps each class at n_per_class, except classes already
    smaller than the cap (mult_steps with 74 < 200)."""
    entries = {c: fake_source_full.list_files(c) for c in dl.GRIMSLEY_OSF_CLASSES}
    samples = dl.sample_per_class(entries, n_per_class=200, seed=1729)

    assert len(samples["step_up"]) == 200
    assert len(samples["mult_steps"]) == 74  # silent degrade
    assert len(samples["rev_chevron"]) == 136  # also smaller than cap


def test_sample_per_class_deterministic(fake_source_full):
    """Same seed → same files (modulo set equality; sample picks a subset)."""
    entries = {c: fake_source_full.list_files(c) for c in ["step_up"]}
    a = dl.sample_per_class(entries, n_per_class=10, seed=1729)
    b = dl.sample_per_class(entries, n_per_class=10, seed=1729)
    assert [e.name for e in a["step_up"]] == [e.name for e in b["step_up"]]


def test_sample_per_class_full_flag_skips_capping(fake_source_full):
    """When n_per_class is None (the --full path), all files are kept."""
    entries = {c: fake_source_full.list_files(c) for c in ["step_up", "mult_steps"]}
    samples = dl.sample_per_class(entries, n_per_class=None, seed=1729)
    assert len(samples["step_up"]) == 1814
    assert len(samples["mult_steps"]) == 74


# --- ROADMAP §18.2a test 1: --dry-run lists plan without fetching --------


def test_dry_run_lists_plan_without_fetching(tmp_path, fake_source_tiny):
    """``--dry-run`` enumerates and plans but never calls download."""
    out = io.StringIO()
    with redirect_stdout(out):
        rc = dl.main(
            argv=["--output-dir", str(tmp_path), "--n-per-class", "200", "--dry-run"],
            source_factory=lambda: fake_source_tiny,
        )
    text = out.getvalue()

    assert rc == 0
    assert "dry-run" in text.lower()
    # 12 classes × 5 files each = 60 files in the plan
    assert "60 files" in text or "Would fetch 60" in text
    # Per-class table must mention every class name
    for cls in dl.GRIMSLEY_OSF_CLASSES:
        assert cls in text, f"class {cls!r} missing from dry-run output"

    # No actual files written
    assert fake_source_tiny.download_calls == 0
    # output-dir should not contain any PNGs (we only wrote nothing)
    pngs = list(tmp_path.rglob("*.png"))
    assert pngs == []


# --- ROADMAP §18.2a test 2: manifest CSV columns + balance ---------------


def test_manifest_columns_and_balance(tmp_path, fake_source_tiny):
    """Manifest has the documented columns; row counts match the plan."""
    rc = dl.main(
        argv=["--output-dir", str(tmp_path), "--n-per-class", "200"],
        source_factory=lambda: fake_source_tiny,
    )
    assert rc == 0

    manifest_path = tmp_path / dl.MANIFEST_FILENAME
    assert manifest_path.exists()

    with open(manifest_path, newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        rows = list(reader)

    assert rows[0] == list(dl.MANIFEST_COLUMNS)
    body = rows[1:]
    # 12 classes × 5 files each = 60 rows
    assert len(body) == 60

    # Each class shows up exactly 5 times in the manifest (balanced).
    class_col = dl.MANIFEST_COLUMNS.index("class")
    counts: dict[str, int] = {}
    for row in body:
        counts[row[class_col]] = counts.get(row[class_col], 0) + 1
    for cls in dl.GRIMSLEY_OSF_CLASSES:
        assert counts.get(cls) == 5, (
            f"class {cls!r} has {counts.get(cls)} rows, expected 5"
        )

    # osf_path column is populated (not empty) for every row.
    osf_col = dl.MANIFEST_COLUMNS.index("osf_path")
    assert all(r[osf_col].startswith("/fake/") for r in body)


# --- ROADMAP §18.2a test 3: re-run is idempotent -------------------------


def test_rerun_is_idempotent(tmp_path, fake_source_tiny):
    """A second invocation with the same args does not re-download."""
    # First run: download 60 files.
    rc1 = dl.main(
        argv=["--output-dir", str(tmp_path), "--n-per-class", "200"],
        source_factory=lambda: fake_source_tiny,
    )
    assert rc1 == 0
    first_call_count = fake_source_tiny.download_calls
    assert first_call_count == 60

    # Second run: with the SAME tmp_path but a FRESH fake (counters reset).
    # All files are already on disk, so the download method must not be
    # invoked at all.
    fake2 = FakeVocalMatSource({c: 5 for c in dl.GRIMSLEY_OSF_CLASSES})
    out = io.StringIO()
    with redirect_stdout(out):
        rc2 = dl.main(
            argv=["--output-dir", str(tmp_path), "--n-per-class", "200"],
            source_factory=lambda: fake2,
        )
    assert rc2 == 0
    assert fake2.download_calls == 0, (
        "Re-run should not call download(); existing files must be skipped"
    )
    assert "skipped" in out.getvalue().lower()


def test_negative_n_per_class_rejected(tmp_path, fake_source_tiny):
    """``--n-per-class 0`` or negative without --full should exit non-zero."""
    rc = dl.main(
        argv=["--output-dir", str(tmp_path), "--n-per-class", "0"],
        source_factory=lambda: fake_source_tiny,
    )
    assert rc == 2


def test_full_flag_pulls_everything(tmp_path, fake_source_tiny):
    """``--full`` ignores ``--n-per-class`` and pulls every file per class."""
    rc = dl.main(
        argv=["--output-dir", str(tmp_path), "--full"],
        source_factory=lambda: fake_source_tiny,
    )
    assert rc == 0
    # 12 classes × 5 files each = 60 (tiny fixture); a real --full run
    # would be ~12,221 files.
    assert fake_source_tiny.download_calls == 60
