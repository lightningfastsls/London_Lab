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
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap so notch.py is importable when running as a script
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.notch import TonalLibrary  # noqa: E402


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
    raise NotImplementedError("calibrate not yet implemented")


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
    raise NotImplementedError("main not yet implemented")


if __name__ == "__main__":
    sys.exit(main())
