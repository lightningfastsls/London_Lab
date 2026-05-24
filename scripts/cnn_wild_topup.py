"""Wild-cohort top-up: process the 853 WAVs the main prep skipped.

Background: scripts/cnn_prepare_training_data.py uses root.glob("*.wav")
(non-recursive) inside _collect_wav_rows. The 5970 cohort has WAVs at the
top level (12 found), but USV_3452_sample_reviewed/ stores its 853 WAVs
under subdirs (USV_1/.../, USV_2/.../, USV_3/.../, USV_4/.../,
uncertain_usv/.../). Those were silently skipped.

This top-up script imports the prep's existing internals and processes the
5 missing subdirs explicitly. It writes new patches alongside the existing
349 wild patches and rewrites domain_unlabeled.csv to include them.

Does NOT modify any production code. Does NOT touch lab patches. Does
NOT re-run the supervised manifest or train/val/test splits (those are
VocalMat-only).
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from usv_spectrogram.classifier import CleaningConfig  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "cnn_prepare_training_data", SCRIPTS / "cnn_prepare_training_data.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


OUTPUT_DIR = Path("data/lab_cnn_training")
DOMAIN_CSV = OUTPUT_DIR / "domain_unlabeled.csv"

# Direct subdirs that contain the missing 3452 WAVs.
WILD_SUBDIRS_3452 = [
    Path("/home/shachar/projects/mickey_london_lab/USV_3452_sample_reviewed/USV_1"),
    Path("/home/shachar/projects/mickey_london_lab/USV_3452_sample_reviewed/USV_2"),
    Path("/home/shachar/projects/mickey_london_lab/USV_3452_sample_reviewed/USV_3"),
    Path("/home/shachar/projects/mickey_london_lab/USV_3452_sample_reviewed/USV_4"),
    Path("/home/shachar/projects/mickey_london_lab/USV_3452_sample_reviewed/uncertain_usv"),
]


def _read_existing_domain_rows() -> list[dict]:
    """Read the current domain_unlabeled.csv so we can append + rewrite."""
    if not DOMAIN_CSV.exists():
        return []
    with open(DOMAIN_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _read_existing_wild_recordings() -> set[str]:
    """Source-recording stems already represented in patches/wild/."""
    wild_dir = OUTPUT_DIR / "patches" / "wild"
    if not wild_dir.is_dir():
        return set()
    stems: set[str] = set()
    for p in wild_dir.glob("*.png"):
        # Strip _p\d+.png suffix.
        stem = p.stem.rsplit("_p", 1)[0]
        stems.add(stem)
    return stems


def main() -> int:
    cfg = CleaningConfig(baseline_mode="percentile")
    patch_duration_s = 0.22

    existing_domain = _read_existing_domain_rows()
    existing_wild_stems = _read_existing_wild_recordings()
    print(f"  existing domain_unlabeled rows: {len(existing_domain)}")
    print(f"  existing wild source recordings (from disk): {len(existing_wild_stems)}")

    # Discover all WAVs in each subdir (recursive within the subdir, in
    # case those subdirs have their own nesting -- harmless if they don't).
    all_wavs: list[Path] = []
    for subdir in WILD_SUBDIRS_3452:
        if not subdir.is_dir():
            print(f"  WARN: missing subdir {subdir}", file=sys.stderr)
            continue
        wavs = sorted(subdir.rglob("*.wav"))
        all_wavs.extend(wavs)
        print(f"  {subdir.name}: {len(wavs)} WAVs")
    print(f"  total wild WAVs to process: {len(all_wavs)}")

    # Skip WAVs whose stem is already in patches/wild/ (idempotent).
    to_process = [w for w in all_wavs if w.stem not in existing_wild_stems]
    skipped = len(all_wavs) - len(to_process)
    print(f"  skipping {skipped} already-processed (matched by stem)")
    print(f"  to process: {len(to_process)}")

    new_rows: list[dict] = []
    t0 = time.perf_counter()
    for i, wav_path in enumerate(to_process):
        rows = prep._wav_to_patches(
            wav_path, patch_duration_s, OUTPUT_DIR, "wild", cfg
        )
        new_rows.extend(rows)
        if (i + 1) % 50 == 0 or i == len(to_process) - 1:
            dt = time.perf_counter() - t0
            print(f"  [{dt:6.1f}s] processed {i+1}/{len(to_process)} WAVs, {len(new_rows)} new patches so far")

    if not new_rows:
        print("  no new wild patches written -- nothing to update.")
        return 0

    # Append to domain_unlabeled.csv. Preserve existing rows + their order;
    # append the new ones at the end.
    fieldnames = ["path", "cohort", "source_recording", "duration_ms"]
    all_rows = existing_domain + new_rows
    print(f"  rewriting {DOMAIN_CSV} with {len(all_rows)} total rows "
          f"(was {len(existing_domain)}, +{len(new_rows)} new)")
    with open(DOMAIN_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fieldnames})

    # Verify final state.
    final_disk = len(list((OUTPUT_DIR / "patches" / "wild").glob("*.png")))
    print(f"\nDONE. wild patches on disk: {final_disk}; domain_unlabeled rows: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
