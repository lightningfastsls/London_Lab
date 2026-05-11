#!/usr/bin/env python3
"""Build a per-rig tonal library by sampling WAV chunks from a rig directory.

Samples ``--sample-size`` chunks uniformly at random from ``--wav-dir``,
runs :func:`usv_spectrogram.app.core.notch.discover_tonals` on each, then
clusters detections across chunks by frequency proximity. Clusters whose
``detection_rate >= --min-detection-rate`` are promoted to
:class:`~usv_spectrogram.app.core.notch.LibraryEntry` records and written
to ``--output`` as a JSON-serialized :class:`TonalLibrary`.

This is the canonical generator for the Layer-2 corpus fact at
``data/lab_tonal_lines/<rig_id>.json``. Re-run when:

- Equipment changes (new rig, replaced microphone, gain adjustment).
- ``run_batch_detection.py`` fires the stale-library warning.
- ``scripts/audit_corpus.py`` flags the file as > 365 days old.

Usage::

    python scripts/calibrate_lab_tonal_lines.py \\
        --wav-dir USV_lab_131204_chunked_2s_hot/ \\
        --rig-id lab_131204 \\
        --sample-size 50 \\
        --min-detection-rate 0.5 \\
        --output data/lab_tonal_lines/lab_131204.json

Rig-id heuristic
----------------
When ``--rig-id`` is omitted, the script derives it from the WAV directory
name by stripping a leading ``USV_`` and a trailing ``_chunked_*`` suffix.
For example, ``USV_lab_131204_chunked_2s_hot/`` -> ``lab_131204``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Path bootstrap so notch.py is importable when running as a script
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.notch import (  # noqa: E402
    DetectedTonal,
    LibraryEntry,
    TonalLibrary,
    discover_tonals,
)


def derive_rig_id(wav_dir: Path) -> str:
    """Strip ``USV_`` prefix and ``_chunked_*`` suffix from the directory name.

    Example: ``USV_lab_131204_chunked_2s_hot/`` -> ``lab_131204``.
    """
    name = wav_dir.name
    if name.startswith("USV_"):
        name = name[len("USV_"):]
    idx = name.find("_chunked_")
    if idx >= 0:
        name = name[:idx]
    return name


def _load_audio_mono(path: Path) -> tuple[np.ndarray, float]:
    """Return ``(audio_1d, fs_hz)``. Multi-channel files are channel-0'd."""
    audio, fs = sf.read(str(path), dtype="float64", always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    return audio, float(fs)


def _cluster_detections(
    per_chunk_tonals: list[list[DetectedTonal]],
    cluster_tolerance_hz: float,
) -> list[dict]:
    """Cluster per-chunk detections by center-frequency proximity.

    Each cluster tracks the detection records (for mean/stdev computation)
    and the distinct set of chunk indices in which the cluster was seen
    (for ``detection_rate``). At most one detection from a given chunk can
    contribute to a single cluster — duplicate-frequency detections within
    the same chunk are silently merged into the closest existing cluster
    or, failing that, start a new cluster.

    Cluster anchor uses the running mean center, so the cluster center is
    not pinned to the first detection's frequency.
    """
    clusters: list[dict] = []
    for chunk_idx, tonals in enumerate(per_chunk_tonals):
        for t in tonals:
            # Find the closest existing cluster within tolerance.
            best = -1
            best_dist = cluster_tolerance_hz + 1
            for i, c in enumerate(clusters):
                mean_center = sum(c["centers"]) / len(c["centers"])
                dist = abs(t.center_hz - mean_center)
                if dist <= cluster_tolerance_hz and dist < best_dist:
                    best_dist = dist
                    best = i
            if best >= 0:
                c = clusters[best]
                c["centers"].append(t.center_hz)
                c["widths"].append(t.width_hz)
                c["aboves"].append(t.above_median_db)
                c["chunks_seen"].add(chunk_idx)
            else:
                clusters.append({
                    "centers": [t.center_hz],
                    "widths": [t.width_hz],
                    "aboves": [t.above_median_db],
                    "chunks_seen": {chunk_idx},
                })
    return clusters


def calibrate(
    wav_dir: Path,
    rig_id: str,
    *,
    sample_size: int = 50,
    min_detection_rate: float = 0.5,
    cluster_tolerance_hz: float = 200.0,
    random_seed: int = 0,
    discovery_threshold_db: float = 10.0,
    median_window_hz: float = 4_000.0,
    nperseg: int = 8192,
) -> TonalLibrary:
    """Sample WAVs, discover tonals across the sample, cluster, and filter.

    Parameters mirror the spec defaults. The pure-Python entry point used
    by ``tests/test_notch.py`` tests 14 and 15. The CLI ``main()`` is a thin
    wrapper around this.

    Returns the calibrated :class:`TonalLibrary` (not yet written to disk).
    """
    wav_dir = Path(wav_dir)
    if not wav_dir.is_dir():
        raise ValueError(f"--wav-dir is not a directory: {wav_dir}")

    # Exclude artefacts of prior soft-notch / band-stop experiments — a WAV
    # that has already been filtered is not representative of the raw rig
    # signal and would bias the library stats.
    wav_paths = sorted(
        p for p in wav_dir.glob("*.wav")
        if "_notch" not in p.stem and "_filtered" not in p.stem
    )
    if not wav_paths:
        raise ValueError(f"No .wav files found in {wav_dir}")

    rng = np.random.default_rng(random_seed)
    n_to_sample = min(int(sample_size), len(wav_paths))
    sampled_indices = sorted(rng.choice(len(wav_paths), size=n_to_sample, replace=False))
    sampled_paths = [wav_paths[i] for i in sampled_indices]

    per_chunk_tonals: list[list[DetectedTonal]] = []
    successfully_loaded: list[Path] = []
    for path in sampled_paths:
        try:
            audio, fs = _load_audio_mono(path)
        except Exception as exc:  # noqa: BLE001 — tolerate any soundfile error
            print(f"[warn] failed to load {path}: {exc}", file=sys.stderr)
            continue
        tonals = discover_tonals(
            audio, fs,
            discovery_threshold_db=discovery_threshold_db,
            median_window_hz=median_window_hz,
            nperseg=nperseg,
        )
        per_chunk_tonals.append(tonals)
        successfully_loaded.append(path)

    n_effective = len(successfully_loaded)
    if n_effective == 0:
        raise RuntimeError(
            f"None of the {n_to_sample} sampled WAVs could be loaded from {wav_dir}"
        )

    clusters = _cluster_detections(per_chunk_tonals, cluster_tolerance_hz)

    entries: list[LibraryEntry] = []
    for c in clusters:
        n_chunks_seen = len(c["chunks_seen"])
        detection_rate = n_chunks_seen / n_effective
        if detection_rate < min_detection_rate:
            continue
        centers = np.asarray(c["centers"], dtype=np.float64)
        widths = np.asarray(c["widths"], dtype=np.float64)
        aboves = np.asarray(c["aboves"], dtype=np.float64)
        entries.append(LibraryEntry(
            center_hz=float(centers.mean()),
            width_hz=float(widths.mean()),
            mean_above_median_db=float(aboves.mean()),
            stdev_above_median_db=float(aboves.std(ddof=0)),
            n_chunks_seen=int(n_chunks_seen),
            detection_rate=float(detection_rate),
        ))

    entries.sort(key=lambda e: e.center_hz)

    return TonalLibrary(
        rig_id=rig_id,
        calibrated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_chunks_sampled=n_effective,
        sample_files=[p.name for p in successfully_loaded],
        entries=entries,
    )


def _print_summary(library: TonalLibrary, wav_dir: Path) -> None:
    print("=" * 66)
    print(f"calibrate_lab_tonal_lines — rig={library.rig_id}")
    print("=" * 66)
    print(f"  wav_dir            : {wav_dir}")
    print(f"  calibrated_at (UTC): {library.calibrated_at}")
    print(f"  n_chunks_sampled   : {library.n_chunks_sampled}")
    print(f"  n_entries          : {len(library.entries)}")
    if library.entries:
        print()
        print(f"  {'center_hz':>12}  {'width_hz':>10}  {'mean_above':>11}  "
              f"{'stdev_above':>12}  {'n_chunks':>9}  {'rate':>6}")
        # Top 5 by detection_rate (then by mean_above_median_db as tiebreaker)
        ranked = sorted(
            library.entries,
            key=lambda e: (-e.detection_rate, -e.mean_above_median_db),
        )
        for e in ranked[:5]:
            print(f"  {e.center_hz:>12.2f}  {e.width_hz:>10.2f}  "
                  f"{e.mean_above_median_db:>11.2f}  "
                  f"{e.stdev_above_median_db:>12.2f}  "
                  f"{e.n_chunks_seen:>9d}  {e.detection_rate:>6.3f}")
    print("=" * 66)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--wav-dir", type=Path, required=True,
                   help="Directory of WAV chunks for this rig.")
    p.add_argument("--rig-id", type=str, default=None,
                   help="Rig identifier (default: derived from --wav-dir name).")
    p.add_argument("--sample-size", type=int, default=50,
                   help="Number of chunks to sample uniformly at random.")
    p.add_argument("--min-detection-rate", type=float, default=0.5,
                   help="Detection-rate threshold to promote a cluster to a library entry.")
    p.add_argument("--cluster-tolerance-hz", type=float, default=200.0,
                   help="Cross-chunk clustering tolerance for tonal centers (Hz).")
    p.add_argument("--discovery-threshold-db", type=float, default=10.0,
                   help="Per-chunk discovery threshold (peak - local median, dB).")
    p.add_argument("--median-window-hz", type=float, default=4_000.0,
                   help="Local-median window for the noise floor estimate (Hz).")
    p.add_argument("--nperseg", type=int, default=8192,
                   help="Welch segment length for PSD estimation.")
    p.add_argument("--random-seed", type=int, default=0,
                   help="Seed for the chunk-sampling RNG (reproducibility).")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSON path (e.g. data/lab_tonal_lines/<rig>.json).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    rig_id = args.rig_id or derive_rig_id(args.wav_dir)

    library = calibrate(
        wav_dir=args.wav_dir,
        rig_id=rig_id,
        sample_size=args.sample_size,
        min_detection_rate=args.min_detection_rate,
        cluster_tolerance_hz=args.cluster_tolerance_hz,
        random_seed=args.random_seed,
        discovery_threshold_db=args.discovery_threshold_db,
        median_window_hz=args.median_window_hz,
        nperseg=args.nperseg,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    library.save(args.output)

    _print_summary(library, args.wav_dir)
    print(f"[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
