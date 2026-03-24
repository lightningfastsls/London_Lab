"""Step 1: Detect USV candidates from sampled WAV files in USV_9252_1 through _8."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.detection.config import DetectionConfig
from usv_spectrogram.detection.energy_detector import EnergyDetector

TOTAL_TARGET = 600
NUM_FOLDERS = 8
PER_FOLDER = TOTAL_TARGET // NUM_FOLDERS
WAV_SAMPLE_PER_FOLDER = 200
SEED = 42

FOLDER_BASE = REPO_ROOT
OUTPUT_CSV = REPO_ROOT / "data" / "candidates_9252_sample.csv"


def save_candidates_csv(candidates, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "candidate_id,source_file,start_ms,end_ms,duration_ms,"
        "context_start_ms,context_end_ms,peak_freq_hz,peak_energy_db,"
        "interference_flag,spectrogram_path"
    )
    lines = [header]
    for c in candidates:
        lines.append(
            f"{c.candidate_id},{c.source_file},{c.start_ms},{c.end_ms},{c.duration_ms},"
            f"{c.context_start_ms},{c.context_end_ms},{c.peak_freq_hz},{c.peak_energy_db},"
            f"{c.interference_flag},"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t0 = time.perf_counter()
    rng = random.Random(SEED)

    print("=" * 60)
    print(f"USV-9252 Detection -- {WAV_SAMPLE_PER_FOLDER} WAVs/folder")
    print("=" * 60)

    wav_dirs = []
    for i in range(1, NUM_FOLDERS + 1):
        folder_name = f"USV_9252_{i}"
        wav_dir = FOLDER_BASE / folder_name / "usv_lmt_036"
        if wav_dir.exists():
            wav_dirs.append((folder_name, wav_dir))
        else:
            print(f"  SKIP: {folder_name}")

    print(f"Found {len(wav_dirs)} folders\n")
    actual_per_folder = TOTAL_TARGET // len(wav_dirs)
    remainder = TOTAL_TARGET % len(wav_dirs)

    config = DetectionConfig(
        sample_rate=300000, auto_sample_rate=True,
        energy_threshold_db=-10.0, energy_mode="peak",
        max_bandwidth_hz=20000, min_duration_ms=10.0, max_duration_ms=500.0,
        merge_gap_ms=3.0, segment_continuity_enabled=True,
        freq_min_hz=25000, freq_max_hz=110000,
    )

    candidates_by_folder = {}

    for folder_name, wav_dir in wav_dirs:
        detector = EnergyDetector(config)
        all_wavs = sorted(wav_dir.glob("*.wav"))
        sample_wavs = rng.sample(all_wavs, min(WAV_SAMPLE_PER_FOLDER, len(all_wavs)))

        print(f"{folder_name}: {len(sample_wavs)} of {len(all_wavs)} WAVs")
        folder_cands = []
        for j, wp in enumerate(sample_wavs, 1):
            try:
                folder_cands.extend(detector.detect(wp))
            except Exception:
                continue
            if j % 100 == 0:
                print(f"  {j}/{len(sample_wavs)}, {len(folder_cands)} cands")

        print(f"  -> {len(folder_cands)} candidates")
        candidates_by_folder[folder_name] = folder_cands

    print("\n--- Sampling ---")
    sampled = []
    for i, fn in enumerate(sorted(candidates_by_folder)):
        cands = candidates_by_folder[fn]
        target = actual_per_folder + (1 if i < remainder else 0)
        if len(cands) <= target:
            sampled.extend(cands)
            print(f"  {fn}: all {len(cands)}")
        else:
            sampled.extend(rng.sample(cands, target))
            print(f"  {fn}: {target} of {len(cands)}")

    print(f"\nTotal: {len(sampled)}")
    save_candidates_csv(sampled, OUTPUT_CSV)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Time: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
