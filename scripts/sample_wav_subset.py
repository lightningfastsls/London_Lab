"""Sample 1/5 of WAV files proportionally from USV_1 through USV_5.

Creates a USV_sample/ directory with a proportional random subset of WAV files
from each source folder, preserving directory structure (e.g., USV_1/usv_lmt_034/).

Usage:
    python scripts/sample_wav_subset.py                    # Default: 1/5 sample
    python scripts/sample_wav_subset.py --fraction 0.1     # 10% sample
    python scripts/sample_wav_subset.py --seed 123         # Custom seed
    python scripts/sample_wav_subset.py --dry-run          # Preview without copying
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def find_source_folders(base_dir: Path) -> list[Path]:
    """Find USV_1 through USV_5 folders (excluding _reviewed)."""
    folders = []
    for i in range(1, 6):
        folder = base_dir / f"USV_{i}"
        if folder.is_dir():
            folders.append(folder)
        else:
            print(f"  Warning: {folder} not found, skipping")
    return folders


def collect_wav_files(folder: Path) -> list[Path]:
    """Recursively collect all .wav files under a folder."""
    return sorted(folder.rglob("*.wav"))


def sample_files(
    source_folders: list[Path],
    fraction: float,
    seed: int,
) -> dict[Path, list[Path]]:
    """Sample a fraction of WAV files from each source folder.

    Returns dict mapping source_folder -> list of sampled file paths.
    """
    rng = random.Random(seed)
    sampled = {}

    for folder in source_folders:
        wav_files = collect_wav_files(folder)
        n_sample = max(1, round(len(wav_files) * fraction))
        selected = rng.sample(wav_files, min(n_sample, len(wav_files)))
        selected.sort()
        sampled[folder] = selected

    return sampled


def copy_to_destination(
    sampled: dict[Path, list[Path]],
    base_dir: Path,
    dest_dir: Path,
    dry_run: bool = False,
) -> int:
    """Copy sampled files to destination, preserving relative structure.

    Returns total number of files copied.
    """
    total = 0

    for source_folder, files in sampled.items():
        for wav_path in files:
            # Preserve path relative to base_dir: USV_1/usv_lmt_034/file.wav
            rel_path = wav_path.relative_to(base_dir)
            dest_path = dest_dir / rel_path

            if dry_run:
                print(f"  [dry-run] {rel_path}")
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wav_path, dest_path)

            total += 1

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a fraction of WAV files from USV_1 through USV_5.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing USV_1..USV_5 folders (default: repo root)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: <base-dir>/USV_sample)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="Fraction of files to sample (default: 0.2 = 1/5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be copied without actually copying",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    dest_dir = (args.dest or base_dir / "USV_sample").resolve()

    if not args.dry_run and dest_dir.exists():
        print(f"Error: Destination {dest_dir} already exists.")
        print("Remove it first or choose a different destination with --dest.")
        sys.exit(1)

    print(f"Base directory: {base_dir}")
    print(f"Destination:    {dest_dir}")
    print(f"Fraction:       {args.fraction} ({args.fraction*100:.0f}%)")
    print(f"Seed:           {args.seed}")
    print()

    # Find source folders
    source_folders = find_source_folders(base_dir)
    if not source_folders:
        print("Error: No USV_1..USV_5 folders found.")
        sys.exit(1)

    # Sample files
    sampled = sample_files(source_folders, args.fraction, args.seed)

    # Report
    print("Sampling plan:")
    total_source = 0
    total_sample = 0
    for folder, files in sampled.items():
        all_wavs = len(collect_wav_files(folder))
        n_selected = len(files)
        total_source += all_wavs
        total_sample += n_selected
        print(f"  {folder.name}: {n_selected}/{all_wavs} files ({n_selected/all_wavs*100:.1f}%)")

    print(f"  Total: {total_sample}/{total_source} files ({total_sample/total_source*100:.1f}%)")
    print()

    # Copy
    if args.dry_run:
        print("Dry run - files that would be copied:")
    else:
        print("Copying files...")

    copied = copy_to_destination(sampled, base_dir, dest_dir, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDry run complete. Would copy {copied} files.")
    else:
        print(f"\nDone! Copied {copied} files to {dest_dir}")
        print(f"Point the PyQt6 detection app at: {dest_dir}")


if __name__ == "__main__":
    main()
