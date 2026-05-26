"""Resample lab WAVs to canonical 300 kHz.

The lab cohort (USV_lab_131204/) was recorded at 250 kHz. The production CNN
and the entire detection/feature pipeline are calibrated for
corpus.SAMPLE_RATE_HZ (300 kHz). The loader in app/core/audio_loader.py
hard-asserts the source sample rate, so we resample to 300 kHz before
detection runs.

Upsample ratio is 300/250 = 6/5. resample_poly uses a polyphase anti-imaging
filter, so the 0..125 kHz band (which covers the full USV range 20..120 kHz)
is preserved exactly. The 125..150 kHz band in the output is spectrally silent,
which is correct: the original recording held no information there.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.corpus import SAMPLE_RATE_HZ

TARGET_RATE_HZ = SAMPLE_RATE_HZ  # 300_000 — canonical CNN / STFT / corpus_facts sample rate


def _resample_one(task: tuple[Path, Path]) -> dict:
    src_path, dst_path = task
    t0 = time.perf_counter()
    data, sr = sf.read(str(src_path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    src_samples = len(data)
    src_duration_s = src_samples / sr

    from math import gcd
    g = gcd(TARGET_RATE_HZ, sr)
    up = TARGET_RATE_HZ // g
    down = sr // g

    resampled = resample_poly(data, up, down).astype(np.float32)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), resampled, TARGET_RATE_HZ, subtype="FLOAT")

    elapsed = time.perf_counter() - t0
    return {
        "src": str(src_path),
        "dst": str(dst_path),
        "src_sample_rate_hz": sr,
        "dst_sample_rate_hz": TARGET_RATE_HZ,
        "up": up,
        "down": down,
        "src_samples": src_samples,
        "dst_samples": len(resampled),
        "src_duration_s": round(src_duration_s, 6),
        "dst_duration_s": round(len(resampled) / TARGET_RATE_HZ, 6),
        "duration_delta_s": round(len(resampled) / TARGET_RATE_HZ - src_duration_s, 6),
        "elapsed_s": round(elapsed, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True, help="Directory of source WAVs.")
    parser.add_argument("--dst-dir", type=Path, required=True, help="Output directory for resampled WAVs.")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for source files (default *.wav).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count (default 4).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files (for testing).")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Write a CSV mapping src->dst with sample rates and durations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.src_dir.is_dir():
        print(f"Error: src-dir not found: {args.src_dir}", file=sys.stderr)
        return 1

    src_files = sorted(args.src_dir.glob(args.pattern))
    if args.limit is not None:
        src_files = src_files[: args.limit]
    if not src_files:
        print(f"No files matching {args.pattern} in {args.src_dir}", file=sys.stderr)
        return 1

    tasks = [(s, args.dst_dir / s.name) for s in src_files]

    print(f"[resample] target_rate_hz={TARGET_RATE_HZ}")
    print(f"[resample] src_dir={args.src_dir}  dst_dir={args.dst_dir}")
    print(f"[resample] files={len(tasks)} workers={args.workers}")

    t0 = time.perf_counter()
    if args.workers <= 1:
        results = [_resample_one(t) for t in tasks]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_resample_one, tasks)
    elapsed = time.perf_counter() - t0

    total_src_s = sum(r["src_duration_s"] for r in results)
    total_dst_s = sum(r["dst_duration_s"] for r in results)
    max_delta = max(abs(r["duration_delta_s"]) for r in results)

    print(f"[resample] done: {len(results)} files in {elapsed:.1f}s")
    print(f"[resample] total src audio: {total_src_s/3600:.3f} h   dst: {total_dst_s/3600:.3f} h")
    print(f"[resample] max abs duration delta: {max_delta*1000:.3f} ms (should be < 1 ms)")

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"[resample] manifest -> {args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
