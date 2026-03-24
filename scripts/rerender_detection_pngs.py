"""Re-render saved detection PNGs with correct MAD normalization.

Walks detection directories, loads the WAV segment for each detection,
recomputes the spectrogram using the same STFT pipeline as the PyQt6 app,
and overwrites the PNG with properly normalized magma colormap rendering.

Usage:
    # Preview one detection (dry run):
    python scripts/rerender_detection_pngs.py --single <detection_dir>/<detection>.json

    # Re-render all detections in a directory:
    python scripts/rerender_detection_pngs.py --dir USV_Detections/

    # Dry run (count only, no overwrite):
    python scripts/rerender_detection_pngs.py --dir USV_Detections/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# STFT parameters matching the app (ExtractionConfig defaults)
SAMPLE_RATE = 300_000
N_FFT = 512
HOP_LENGTH = 128
WINDOW = "hann"
FREQ_MIN_HZ = 20_000
FREQ_MAX_HZ = 120_000
EPS = 1e-10

# MAD normalization scales (same as spectrogram_view.py and extraction_config.py)
MAD_VMIN_SCALE = 2.0
MAD_VMAX_SCALE = 4.0


def find_wav_file(detection_dir: Path) -> Path | None:
    """Find the WAV file corresponding to a detection folder.

    Searches multiple known locations:
    1. Same parent directory (e.g., USV_Detections/../5970 USV/)
    2. 5970 USV/ in repo root
    3. Nearby reviewed sample directories
    """
    folder_name = detection_dir.name  # e.g., "2024-09-30_11-18-17_0000001"
    wav_name = f"{folder_name}.wav"

    # Search paths (ordered by priority)
    repo_root = Path(__file__).resolve().parents[1]
    search_dirs = [
        repo_root / "5970 USV",
    ]

    # Check simple search dirs first
    for search_dir in search_dirs:
        candidate = search_dir / wav_name
        if candidate.exists():
            return candidate

    # Search inside reviewed sample dirs (USV_3452_sample_reviewed, etc.)
    for reviewed_dir in repo_root.glob("USV_*_sample_reviewed"):
        for usv_subdir in reviewed_dir.glob("USV_*"):
            for lmt_dir in usv_subdir.glob("usv_lmt_*"):
                candidate = lmt_dir / wav_name
                if candidate.exists():
                    return candidate

    # Search inside unreviewed sample dirs (USV_3452_sample, etc.)
    for sample_dir in repo_root.glob("USV_*_sample"):
        for usv_subdir in sample_dir.glob("USV_*"):
            for lmt_dir in usv_subdir.glob("usv_lmt_*"):
                candidate = lmt_dir / wav_name
                if candidate.exists():
                    return candidate

    # Search archived 5970 recordings (reviewed + unreviewed)
    archive_5970 = repo_root / "archive" / "5970"
    if archive_5970.exists():
        for status_dir in archive_5970.iterdir():  # reviewed/, unreviewed/
            if not status_dir.is_dir():
                continue
            for usv_subdir in status_dir.glob("USV_*"):
                for lmt_dir in usv_subdir.glob("usv_lmt_*"):
                    candidate = lmt_dir / wav_name
                    if candidate.exists():
                        return candidate

    # Search downloaded WAV batches (Google Drive chunks)
    for batch_dir in repo_root.glob("usv_lmt_034-*"):
        for lmt_dir in batch_dir.glob("usv_lmt_*"):
            candidate = lmt_dir / wav_name
            if candidate.exists():
                return candidate

    return None


def compute_spectrogram_segment(
    wav_path: Path,
    start_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load WAV segment and compute spectrogram matching the app pipeline.

    Returns:
        (spec_db, freqs_hz, times_s) matching AudioLoader output.
    """
    import soundfile as sf

    # Load segment
    with sf.SoundFile(str(wav_path)) as wav:
        sr = int(wav.samplerate)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz")
        start_frame = int(round(start_s * sr))
        if start_frame < 0:
            start_frame = 0
        wav.seek(start_frame)
        n_frames = int(round(duration_s * sr))
        samples = wav.read(n_frames, dtype="float32", always_2d=True)

    if samples.size == 0:
        raise ValueError("Empty audio segment")
    if samples.shape[1] > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples[:, 0]

    # Build frequency bins and band mask
    freqs_hz = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    band_mask = (freqs_hz >= FREQ_MIN_HZ) & (freqs_hz <= FREQ_MAX_HZ)

    if len(samples) < N_FFT:
        raise ValueError(f"Audio too short ({len(samples)} samples < {N_FFT} n_fft)")

    # Extract frames
    n_stft_frames = 1 + (len(samples) - N_FFT) // HOP_LENGTH
    frame_starts = np.arange(n_stft_frames) * HOP_LENGTH
    frames = np.stack(
        [samples[start:start + N_FFT] for start in frame_starts],
        axis=0,
    )

    # Window + FFT
    window = signal.get_window(WINDOW, N_FFT, fftbins=True)
    windowed = frames * window
    stft = np.fft.rfft(windowed, n=N_FFT, axis=1)
    magnitude = np.abs(stft)

    # Normalize magnitude (matches app: normalize_magnitude=True)
    magnitude = magnitude / (np.max(magnitude) + EPS)

    # Convert to dB
    spec_db = 20.0 * np.log10(magnitude + EPS)
    spec_db = spec_db[:, band_mask].T  # Shape: (n_freq_bins, n_frames)

    # Time bins (frame centers)
    times_s = (frame_starts + N_FFT / 2.0) / SAMPLE_RATE + start_s

    return spec_db, freqs_hz[band_mask], times_s


def render_detection_png(
    spec_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    detection_start_s: float,
    detection_end_s: float,
    detection_index: int,
    max_prob: float,
    mean_prob: float,
    output_path: Path,
) -> None:
    """Render annotated spectrogram PNG with MAD normalization."""
    fig, ax = plt.subplots(figsize=(12, 6))

    freqs_khz = freqs_hz / 1000.0

    # MAD-based normalization (matches on-screen display)
    median = np.median(spec_db)
    mad = np.median(np.abs(spec_db - median))
    vmin = median - MAD_VMIN_SCALE * mad
    vmax = median + MAD_VMAX_SCALE * mad

    im = ax.imshow(
        spec_db,
        aspect='auto',
        origin='lower',
        extent=[times_s[0], times_s[-1], freqs_khz[0], freqs_khz[-1]],
        cmap='magma',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax,
    )

    # Detection boundary lines
    ax.axvline(detection_start_s, color='cyan', linestyle='--',
               linewidth=2, label='Detection start')
    ax.axvline(detection_end_s, color='lime', linestyle='--',
               linewidth=2, label='Detection end')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)

    duration_ms = (detection_end_s - detection_start_s) * 1000
    ax.set_title(
        f'Detection {detection_start_s:.3f}s - {detection_end_s:.3f}s '
        f'({duration_ms:.1f} ms)\n'
        f'Max prob: {max_prob:.3f}, Mean prob: {mean_prob:.3f}',
        fontsize=11,
    )

    plt.colorbar(im, ax=ax, label='Power (dB)')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def rerender_single(json_path: Path, wav_path: Path | None = None, dry_run: bool = False) -> bool:
    """Re-render a single detection PNG from its JSON metadata.

    Returns True if successful.
    """
    # Load metadata
    try:
        with open(json_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [SKIP] Corrupt JSON: {json_path.name} ({e})")
        return False

    try:
        core = meta["core_time"]
        saved = meta["saved_region"]
        probs = meta["probabilities"]
    except KeyError as e:
        print(f"  [SKIP] Missing key in {json_path.name}: {e}")
        return False

    png_path = json_path.with_suffix(".png").absolute()

    if dry_run:
        print(f"  [DRY RUN] Would re-render: {png_path.name}")
        return True

    # Find WAV file
    if wav_path is None:
        wav_path = find_wav_file(json_path.parent)
    if wav_path is None:
        print(f"  [SKIP] No WAV found for {json_path.parent.name}")
        return False

    # Compute spectrogram for the saved region
    start_s = saved["start_s"]
    end_s = saved["end_s"]
    duration_s = end_s - start_s

    try:
        spec_db, freqs_hz, times_s = compute_spectrogram_segment(
            wav_path, start_s, duration_s
        )
    except Exception as e:
        print(f"  [ERROR] {json_path.name}: {e}")
        return False

    # Render with correct normalization
    try:
        render_detection_png(
            spec_db, freqs_hz, times_s,
            detection_start_s=core["start_s"],
            detection_end_s=core["end_s"],
            detection_index=meta["detection_index"],
            max_prob=probs["max"],
            mean_prob=probs["mean"],
            output_path=png_path,
        )
    except Exception as e:
        print(f"  [ERROR] Render failed {json_path.name}: {e}")
        return False

    print(f"  [OK] {png_path.name}")
    return True


def rerender_directory(base_dir: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Re-render all detection PNGs under a directory.

    Returns (success_count, skip_count, error_count).
    """
    success = skip = errors = 0

    # Find all detection JSON files
    json_files = sorted(base_dir.rglob("detection_*.json"))

    if not json_files:
        print(f"No detection JSON files found under {base_dir}")
        return 0, 0, 0

    print(f"Found {len(json_files)} detection(s) under {base_dir}")

    # Group by parent directory (WAV file) for efficiency
    from collections import defaultdict
    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for jf in json_files:
        by_parent[jf.parent].append(jf)

    for parent_dir, jsons in sorted(by_parent.items()):
        wav_path = find_wav_file(parent_dir)
        folder_name = parent_dir.name

        if wav_path is None and not dry_run:
            print(f"\n[{folder_name}] WAV not found — skipping {len(jsons)} detection(s)")
            skip += len(jsons)
            continue

        print(f"\n[{folder_name}] {len(jsons)} detection(s)" +
              (f" — WAV: {wav_path.name}" if wav_path else ""))

        for jf in jsons:
            ok = rerender_single(jf, wav_path=wav_path, dry_run=dry_run)
            if ok:
                success += 1
            else:
                errors += 1

    return success, skip, errors


def main():
    parser = argparse.ArgumentParser(
        description="Re-render saved detection PNGs with correct MAD normalization."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", type=Path, help="Path to a single detection JSON file")
    group.add_argument("--dir", type=Path, help="Base directory to scan for detection JSONs")
    parser.add_argument("--dry-run", action="store_true", help="Count files only, don't overwrite")

    args = parser.parse_args()

    if args.single:
        if not args.single.exists():
            print(f"File not found: {args.single}")
            sys.exit(1)
        ok = rerender_single(args.single, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)
    else:
        if not args.dir.exists():
            print(f"Directory not found: {args.dir}")
            sys.exit(1)
        success, skip, errors = rerender_directory(args.dir, dry_run=args.dry_run)
        print(f"\nDone: {success} re-rendered, {skip} skipped, {errors} errors")
        sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
