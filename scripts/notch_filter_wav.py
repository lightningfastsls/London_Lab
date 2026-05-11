#!/usr/bin/env python3
"""Adaptive soft-notch filter for a single WAV file.

Thin CLI wrapper around :func:`usv_spectrogram.app.core.notch.auto_soft_notch`.
Replaces the legacy hand-tuned hard band-stop (``--center``/``--width``)
with a data-driven adaptive cut whose frequency, width, and depth all come
from the audio itself (audit mode) or a per-rig calibrated library
(library mode).

Modes
-----
- **Auto-detect** (default, no ``--library``): runs Welch PSD + local-median
  peak discovery on the input WAV; for each detected tonal, applies a
  finite-depth soft-notch whose cut equals the tonal's elevation above the
  local median. Useful for single-file experiments and probe runs.

- **Library** (``--library data/lab_tonal_lines/<rig>.json``): applies the
  library's calibrated entries. Cut depth is still measured per-chunk
  locally; only frequency and width come from the library. Useful for
  testing whether a library is suitable for a new WAV from the same rig.

- **Probe** (``--probe``): discover + print + plot, but do NOT write a WAV.
  Lists every detected tonal with its center, width, and elevation.

- **Manual override** (``--manual-band``, repeatable): force a band into
  the filter set, on top of (or instead of) the auto-detector's findings.
  Format: ``center_hz,width_hz`` (e.g. ``50000,300``). Use when the
  auto-detector misses a known issue.

Examples
--------
::

    # Audit mode: see what's there
    python scripts/notch_filter_wav.py --input recording.wav --probe

    # Audit mode: clean and write recording_notch.wav
    python scripts/notch_filter_wav.py --input recording.wav --plot

    # Library mode: apply lab_131204 library, also plot before/after
    python scripts/notch_filter_wav.py --input recording.wav \\
        --library data/lab_tonal_lines/lab_131204.json --plot

    # Manual override on top of auto-detect
    python scripts/notch_filter_wav.py --input recording.wav \\
        --manual-band 60000,500 --manual-band 73000,400
"""

from __future__ import annotations

import argparse
import sys
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
    apply_soft_notches,
    auto_soft_notch,
    discover_tonals,
)


def _read_audio(path: Path) -> tuple[np.ndarray, int, str]:
    """Return ``(audio, fs_hz, subtype)``. Subtype is preserved on write."""
    info = sf.info(str(path))
    audio, fs_hz = sf.read(str(path), dtype="float64", always_2d=False)
    return audio, int(fs_hz), info.subtype


def _float_to_native(audio_f64: np.ndarray, subtype: str) -> np.ndarray:
    """Convert filtered float64 audio back to the input file's native dtype."""
    if subtype in ("FLOAT", "DOUBLE"):
        return audio_f64.astype(np.float32 if subtype == "FLOAT" else np.float64)
    if subtype == "PCM_16":
        clipped = np.clip(audio_f64, -1.0, 1.0)
        return (clipped * 32767.0).round().astype(np.int16)
    if subtype in ("PCM_24", "PCM_32"):
        clipped = np.clip(audio_f64, -1.0, 1.0)
        return (clipped * (2**31 - 1)).round().astype(np.int32)
    # Unknown subtype — preserve as float32.
    return audio_f64.astype(np.float32)


def _parse_manual_band(spec: str) -> tuple[float, float]:
    """Parse ``center,width`` (Hz) for ``--manual-band``."""
    parts = [s.strip() for s in spec.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--manual-band expects 'center_hz,width_hz'; got {spec!r}"
        )
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--manual-band parse error in {spec!r}: {exc}"
        ) from exc


def _manual_to_detected_tonal(
    audio: np.ndarray, fs_hz: float, center_hz: float, width_hz: float,
    median_window_hz: float, nperseg: int,
) -> DetectedTonal:
    """Build a :class:`DetectedTonal` for a manually-specified band.

    Cut depth is measured from the local PSD (peak in the band minus median
    in the surrounding ``median_window_hz``), so manual bands attenuate by
    the right amount without the caller having to guess ``cut_depth_db``.
    """
    from usv_spectrogram.app.core.notch import _measure_band_psd, _select_channel, _welch_psd_db
    a = _select_channel(audio)
    f, pxx_db = _welch_psd_db(a, fs_hz, nperseg)
    peak_db, local_median_db = _measure_band_psd(
        f, pxx_db,
        band_lo_hz=center_hz - width_hz / 2.0,
        band_hi_hz=center_hz + width_hz / 2.0,
        median_window_hz=median_window_hz,
    )
    above = peak_db - local_median_db if peak_db == peak_db else 0.0  # NaN guard
    return DetectedTonal(
        center_hz=center_hz,
        width_hz=width_hz,
        peak_db=float(peak_db if peak_db == peak_db else 0.0),
        local_median_db=float(local_median_db if local_median_db == local_median_db else 0.0),
        above_median_db=float(above if above > 0 else 0.0),
    )


def _format_tonal_line(t, source: str) -> str:
    # LibraryEntry uses mean_above_median_db; DetectedTonal uses above_median_db.
    above_db = getattr(t, "above_median_db", None)
    if above_db is None:
        above_db = getattr(t, "mean_above_median_db", float("nan"))
    return (f"  {source:>8}  center={t.center_hz/1000:7.2f} kHz  "
            f"width={t.width_hz:6.1f} Hz  above_median={above_db:6.2f} dB")


def _plot_psd(
    audio_in: np.ndarray, audio_out: np.ndarray, fs_hz: float,
    library_entries: list[LibraryEntry], detected: list[DetectedTonal],
    save_path: Path,
) -> None:
    """Save a Welch PSD before/after plot. Library bands blue; audit red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    a_in = audio_in[:, 0] if audio_in.ndim == 2 else audio_in
    a_out = audio_out[:, 0] if audio_out.ndim == 2 else audio_out
    nperseg = min(8192, len(a_in))
    f, pxx_in = welch(a_in, fs=fs_hz, nperseg=nperseg)
    _, pxx_out = welch(a_out, fs=fs_hz, nperseg=nperseg)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogy(f / 1000.0, pxx_in, label="before", alpha=0.7)
    ax.semilogy(f / 1000.0, pxx_out, label="after", alpha=0.9)
    for e in library_entries:
        ax.axvspan((e.center_hz - e.width_hz/2)/1000.0,
                   (e.center_hz + e.width_hz/2)/1000.0,
                   color="blue", alpha=0.15)
    for d in detected:
        ax.axvspan((d.center_hz - d.width_hz/2)/1000.0,
                   (d.center_hz + d.width_hz/2)/1000.0,
                   color="red", alpha=0.15)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.set_title(f"Adaptive soft-notch — fs={fs_hz/1000:.0f} kHz "
                 f"(blue=library, red=audit)")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Path to input WAV.")
    p.add_argument("--library", type=Path, default=None,
                   help="Path to a TonalLibrary JSON. If omitted, runs pure auto-detect.")
    p.add_argument("--probe", action="store_true",
                   help="Discover and print tonals without writing a WAV.")
    p.add_argument("--plot", action="store_true",
                   help="Save a before/after Welch PSD next to the output WAV.")
    p.add_argument("--manual-band", action="append", default=[], type=_parse_manual_band,
                   help="Force a band into the filter set. Format: 'center_hz,width_hz'. "
                        "Repeatable. Cut depth is measured from the local PSD.")
    p.add_argument("--discovery-threshold-db", type=float, default=10.0,
                   help="Per-chunk discovery threshold for the audit detector.")
    p.add_argument("--median-window-hz", type=float, default=4_000.0,
                   help="Local-median window for the noise-floor estimate (Hz).")
    p.add_argument("--safety-margin-db", type=float, default=0.0,
                   help="Extra dB applied on top of the measured cut depth.")
    p.add_argument("--width-safety-factor", type=float, default=2.0,
                   help="Multiplier on measured tonal width when applying the filter.")
    p.add_argument("--nperseg", type=int, default=8192,
                   help="Welch segment length for PSD estimation.")
    p.add_argument("--suffix", default="_notch",
                   help="Suffix appended to the output filename stem.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    print(f"Reading {args.input}")
    audio, fs_hz, subtype = _read_audio(args.input)
    n_samples = audio.shape[0]
    n_ch = audio.shape[1] if audio.ndim == 2 else 1
    duration_s = n_samples / fs_hz
    print(f"  fs={fs_hz} Hz, channels={n_ch}, samples={n_samples} "
          f"({duration_s:.2f} s), subtype={subtype}")

    library: Optional[TonalLibrary] = None
    if args.library is not None:
        library = TonalLibrary.load(args.library)
        print(f"Loaded library rig={library.rig_id} entries={len(library.entries)}")

    # ----- Probe mode -----
    if args.probe:
        detected = discover_tonals(
            audio, fs_hz,
            discovery_threshold_db=args.discovery_threshold_db,
            median_window_hz=args.median_window_hz,
            nperseg=args.nperseg,
        )
        print(f"Probe mode — {len(detected)} tonals discovered:")
        if library is not None:
            for e in library.entries:
                print(_format_tonal_line(e, "library"))
        for d in detected:
            print(_format_tonal_line(d, "audit"))
        return 0

    # ----- Engine: auto_soft_notch -----
    cleaned, recon = auto_soft_notch(
        audio, fs_hz, library=library,
        discovery_threshold_db=args.discovery_threshold_db,
        median_window_hz=args.median_window_hz,
        nperseg=args.nperseg,
        safety_margin_db=args.safety_margin_db,
        width_safety_factor=args.width_safety_factor,
    )

    # ----- Manual band overrides -----
    if args.manual_band:
        manual_tonals = [
            _manual_to_detected_tonal(
                audio, fs_hz, c_hz, w_hz,
                median_window_hz=args.median_window_hz,
                nperseg=args.nperseg,
            )
            for c_hz, w_hz in args.manual_band
        ]
        cleaned = apply_soft_notches(
            cleaned, fs_hz, manual_tonals,
            width_safety_factor=1.0,  # manual width is already explicit; don't double it
            safety_margin_db=args.safety_margin_db,
        )

    # ----- Logging -----
    if library is not None:
        for entry, det in recon.matched:
            print(_format_tonal_line(entry, "library"))
            print(f"     measured center={det.center_hz/1000:.2f} kHz  "
                  f"above_median={det.above_median_db:.2f} dB")
        for d in recon.unmatched_detections:
            print(_format_tonal_line(d, "DRIFT"))
            print("     (audit-only; not filtered by library)")
        for entry, sigma in recon.intensity_drifts:
            print(f"  WARN intensity-drift at center={entry.center_hz/1000:.2f} kHz: "
                  f"|measured - library mean| = {sigma:.2f} sigma")
    else:
        for d in recon.unmatched_detections:
            print(_format_tonal_line(d, "audit"))
    if args.manual_band:
        for c_hz, w_hz in args.manual_band:
            print(f"  manual    center={c_hz/1000:.2f} kHz  width={w_hz:.1f} Hz")

    # ----- Write output -----
    out_path = args.input.with_name(args.input.stem + args.suffix + args.input.suffix)
    out_native = _float_to_native(cleaned, subtype)
    print(f"Writing {out_path} (subtype={subtype})")
    sf.write(str(out_path), out_native, int(fs_hz), subtype=subtype)

    if args.plot:
        plot_path = out_path.with_suffix(".png")
        print(f"Saving PSD plot -> {plot_path}")
        lib_entries = list(library.entries) if library is not None else []
        all_detected = list(recon.unmatched_detections) + [m[1] for m in recon.matched]
        _plot_psd(audio, cleaned, fs_hz, lib_entries, all_detected, plot_path)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
