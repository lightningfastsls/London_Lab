"""Chunk long lab WAVs and resample each chunk to canonical 300 kHz.

The wild detection pipeline was built for trigger-recorded snippets (~2 s each).
The lab cohort (USV_lab_131204/) is continuous 10-min recordings at 250 kHz.
Feeding a 10-min file through the pipeline OOMs the loader (single-pass spectrogram
of a 600 s file at 300 kHz needs ~1.4 GB).

This preprocessor slices each long lab WAV into overlapping chunks and resamples
each chunk to corpus.SAMPLE_RATE_HZ. The pipeline then sees many small files,
identical in shape to wild data.

Chunking is undone downstream by reading the chunk manifest:
   original_begin_time_s = chunk_begin_time_s + start_s_in_original

Overlap-region detections (USVs that fall in the shared region between two
consecutive chunks) are deduped by matching (original_file, original_begin_time_s)
within a small tolerance.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.corpus import SAMPLE_RATE_HZ

TARGET_RATE_HZ = SAMPLE_RATE_HZ  # canonical CNN / STFT / corpus_facts sample rate


def _process_one(task: tuple[Path, Path, float, float, float]) -> list[dict]:
    src_path, dst_dir, chunk_duration_s, overlap_s, min_chunk_s = task

    t0 = time.perf_counter()
    data, src_sr = sf.read(str(src_path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)
    total_samples = len(data)
    src_duration_s = total_samples / src_sr

    g = gcd(TARGET_RATE_HZ, src_sr)
    up = TARGET_RATE_HZ // g
    down = src_sr // g

    chunk_size_samples = int(round(chunk_duration_s * src_sr))
    step_samples = int(round((chunk_duration_s - overlap_s) * src_sr))
    min_chunk_samples = int(round(min_chunk_s * src_sr))

    dst_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    chunk_index = 0
    cursor = 0
    while cursor < total_samples:
        end = min(cursor + chunk_size_samples, total_samples)
        chunk_samples = end - cursor
        if chunk_samples < min_chunk_samples:
            break

        chunk = data[cursor:end]
        resampled = resample_poly(chunk, up, down).astype(np.float32)

        chunk_filename = f"{src_path.stem}_chunk_{chunk_index:03d}.wav"
        chunk_path = dst_dir / chunk_filename
        sf.write(str(chunk_path), resampled, TARGET_RATE_HZ, subtype="FLOAT")

        rows.append({
            "chunk_filename": chunk_filename,
            "original_filename": src_path.name,
            "chunk_index": chunk_index,
            "start_s_in_original": round(cursor / src_sr, 6),
            "end_s_in_original": round(end / src_sr, 6),
            "src_chunk_duration_s": round(chunk_samples / src_sr, 6),
            "dst_chunk_duration_s": round(len(resampled) / TARGET_RATE_HZ, 6),
            "src_sample_rate_hz": src_sr,
            "dst_sample_rate_hz": TARGET_RATE_HZ,
            "up": up,
            "down": down,
            "overlap_s": overlap_s,
            "src_samples_in_chunk": chunk_samples,
            "dst_samples_in_chunk": len(resampled),
            "src_total_duration_s": round(src_duration_s, 6),
        })

        chunk_index += 1
        cursor += step_samples

    elapsed = time.perf_counter() - t0
    print(f"[chunk] {src_path.name}: {len(rows)} chunks in {elapsed:.1f}s", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Output CSV: chunk_filename -> original_filename + start_s_in_original.")
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--chunk-duration-s", type=float, default=60.0)
    parser.add_argument("--overlap-s", type=float, default=0.5)
    parser.add_argument("--min-chunk-s", type=float, default=1.0,
                        help="Discard a final partial chunk shorter than this (default 1 s).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N source files (smoke testing).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.src_dir.is_dir():
        print(f"Error: src-dir not found: {args.src_dir}", file=sys.stderr)
        return 1
    if args.overlap_s >= args.chunk_duration_s:
        print(f"Error: overlap ({args.overlap_s}s) must be < chunk duration ({args.chunk_duration_s}s)",
              file=sys.stderr)
        return 1

    src_files = sorted(args.src_dir.glob(args.pattern))
    if args.limit is not None:
        src_files = src_files[: args.limit]
    if not src_files:
        print(f"No files matching {args.pattern} in {args.src_dir}", file=sys.stderr)
        return 1

    tasks = [
        (s, args.dst_dir, args.chunk_duration_s, args.overlap_s, args.min_chunk_s)
        for s in src_files
    ]

    print(f"[chunk] target_rate_hz={TARGET_RATE_HZ}")
    print(f"[chunk] src_dir={args.src_dir}  dst_dir={args.dst_dir}")
    print(f"[chunk] chunk_duration_s={args.chunk_duration_s}  overlap_s={args.overlap_s}  min_chunk_s={args.min_chunk_s}")
    print(f"[chunk] src_files={len(tasks)}  workers={args.workers}")

    t0 = time.perf_counter()
    if args.workers <= 1:
        results = [_process_one(t) for t in tasks]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_process_one, tasks)
    elapsed = time.perf_counter() - t0

    flat = [row for rows in results for row in rows]
    n_chunks = len(flat)
    total_dst_audio_s = sum(r["dst_chunk_duration_s"] for r in flat)
    print(f"[chunk] done: {len(tasks)} src files -> {n_chunks} chunks in {elapsed:.1f}s")
    print(f"[chunk] total dst audio: {total_dst_audio_s/3600:.3f} h "
          f"(includes {args.overlap_s*1000:.0f}ms × (n_chunks - n_src) overlap)")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as fh:
        if not flat:
            print("[chunk] WARNING: no chunks produced", file=sys.stderr)
            return 1
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(f"[chunk] manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
