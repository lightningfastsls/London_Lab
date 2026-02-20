"""Pick ~600 random WAV files evenly from USV_9252_1 through _8.

Copies selected files into subfolders by source (USV_9252_1/, USV_9252_2/, etc.)
inside the output directory, so you can trace which recording each file came from.
"""

import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOTAL_TARGET = 600
NUM_FOLDERS = 8
SEED = 42
OUTPUT_DIR = REPO_ROOT / "USV_9252_sample"


def main() -> int:
    rng = random.Random(SEED)

    # Clean existing output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleaned existing {OUTPUT_DIR}")

    # Find all folders and their WAV files
    all_folders: list[tuple[str, list[Path]]] = []
    for i in range(1, NUM_FOLDERS + 1):
        name = f"USV_9252_{i}"
        wav_dir = REPO_ROOT / name / "usv_lmt_036"
        if not wav_dir.exists():
            wav_dir = REPO_ROOT / name
        if not wav_dir.exists():
            print(f"  SKIP: {name} not found")
            continue
        wavs = sorted(wav_dir.glob("*.wav"))
        if not wavs:
            print(f"  SKIP: {name} has no WAV files")
            continue
        all_folders.append((name, wavs))
        print(f"  {name}: {len(wavs)} WAV files")

    if not all_folders:
        print("ERROR: No folders found!")
        return 1

    # Calculate per-folder count
    per_folder = TOTAL_TARGET // len(all_folders)
    remainder = TOTAL_TARGET % len(all_folders)
    print(f"\nTarget: {TOTAL_TARGET} files, {per_folder} per folder (+{remainder} extra)")

    total_copied = 0
    for idx, (name, wavs) in enumerate(all_folders):
        target = per_folder + (1 if idx < remainder else 0)
        if len(wavs) <= target:
            chosen = wavs
        else:
            chosen = rng.sample(wavs, target)

        # Create subfolder per source
        sub_dir = OUTPUT_DIR / name
        sub_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in chosen:
            shutil.copy2(wav_path, sub_dir / wav_path.name)
            total_copied += 1

        print(f"  {name}: copied {len(chosen)} files -> {sub_dir}")

    print(f"\nDone! {total_copied} WAV files in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
