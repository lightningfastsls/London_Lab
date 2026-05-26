"""Preview spectral background subtraction (Boll 1979) on lab chunks.

For each chunk we render a side-by-side comparison of the original spectrogram
versus the spectrogram after subtracting a per-frequency-bin temporal baseline.

The principle: a stationary tonal noise (e.g. equipment hum at 50 kHz that lasts
through the entire chunk) has a high temporal baseline in its frequency bin.
A transient USV at the same frequency has a low temporal baseline (it's only
loud for ~50 ms out of 2,000 ms). Subtracting the baseline removes the
stationary component while preserving transients.

Implementation notes:
- We subtract in *linear magnitude*, not in dB, so the math is correct.
- Baseline is the 10th percentile of magnitude over time per frequency bin.
  Robust to occasional bright USV bursts; not corrupted unless a single freq
  bin is occupied >90% of the chunk by signal (essentially never for mice).
- Cleaned spectrogram is converted back to dB for plotting & comparison.
- The CNN's input pipeline is unchanged; this script only previews what
  spectral subtraction *would* do if we applied it as a preprocessing step.

Outputs:
    results/batch_lab_131204_full/spectral_subtraction_preview/{stem}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from usv_spectrogram.app.core.audio_loader import AudioLoader  # noqa: E402
from usv_spectrogram.app.core.denoise import (  # noqa: E402
    DEFAULT_BASELINE_PERCENTILE,
    subtract_temporal_baseline,
)


def subtract_stationary_baseline(
    spec_db: np.ndarray, baseline_percentile: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cleaned_spec_db, baseline_db_per_bin).

    Thin dB↔magnitude wrapper around the canonical
    ``denoise.subtract_temporal_baseline``, retained here so the preview
    can render the per-bin baseline alongside cleaned spectrograms.
    """
    eps = 1e-10
    magnitude = np.power(10.0, spec_db / 20.0)
    baseline_mag = np.percentile(
        magnitude, baseline_percentile, axis=1, keepdims=True
    )
    cleaned_mag = subtract_temporal_baseline(
        magnitude, percentile=baseline_percentile, epsilon=eps
    )
    cleaned_db = 20.0 * np.log10(cleaned_mag)
    baseline_db = 20.0 * np.log10(baseline_mag.squeeze() + eps)
    return cleaned_db, baseline_db


def render_comparison(
    chunk_wav: Path,
    events: pd.DataFrame,
    out_path: Path,
    baseline_percentile: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader = AudioLoader()
    audio = loader.load(chunk_wav)
    spec_db = audio.spectrogram_db
    spec_times = audio.times
    freqs_hz = audio.frequencies

    cleaned_db, baseline_db = subtract_stationary_baseline(spec_db, baseline_percentile)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=140, sharex=True)

    extent = [spec_times[0], spec_times[-1], freqs_hz[0] / 1000.0, freqs_hz[-1] / 1000.0]
    vmin = float(np.percentile(spec_db, 5))
    vmax = float(np.percentile(spec_db, 99))

    axes[0].imshow(
        spec_db, aspect="auto", origin="lower", cmap="magma",
        extent=extent, vmin=vmin, vmax=vmax,
    )
    axes[0].set_title(
        f"ORIGINAL  {chunk_wav.stem}  ({len(events)} CNN detections)", fontsize=10
    )
    axes[0].set_ylabel("Freq (kHz)")

    vmin_c = float(np.percentile(cleaned_db, 5))
    vmax_c = float(np.percentile(cleaned_db, 99))
    axes[1].imshow(
        cleaned_db, aspect="auto", origin="lower", cmap="magma",
        extent=extent, vmin=vmin_c, vmax=vmax_c,
    )
    axes[1].set_title(
        f"CLEANED  (per-bin {baseline_percentile:.0f}th-percentile baseline subtracted)",
        fontsize=10,
    )
    axes[1].set_xlabel("Time in chunk (s)")
    axes[1].set_ylabel("Freq (kHz)")

    for _, ev in events.iterrows():
        start_t = ev["original_begin_time_s"] - ev["start_s_in_original"]
        end_t = ev["original_end_time_s"] - ev["start_s_in_original"]
        for ax in axes:
            ax.axvspan(start_t, end_t, alpha=0.25, color="cyan", linewidth=0)
            ax.axvline(start_t, color="cyan", linewidth=0.5, alpha=0.7)
            ax.axvline(end_t, color="cyan", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--events",
        type=Path,
        default=REPO_ROOT / "results/batch_lab_131204_full/merged_events_full.parquet",
    )
    ap.add_argument(
        "--chunks-dir",
        type=Path,
        default=REPO_ROOT / "USV_lab_131204_chunked_2s_full",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "results/batch_lab_131204_full/spectral_subtraction_preview",
    )
    ap.add_argument("--baseline-percentile", type=float, default=DEFAULT_BASELINE_PERCENTILE)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    events = pd.read_parquet(args.events)
    print(f"[load] {len(events)} events")

    targets: list[str] = [
        "131204_1400_m4fm4_chunk_194",
        "131208_1000_m2fm2_chunk_124",
        "131205_1000_m1fm1_chunk_107",
        "131211_1800_m4fm2_chunk_283",
        "131205_1800_m6fm6_chunk_165",
    ]

    summary = pd.read_parquet(
        REPO_ROOT / "results/batch_lab_131204_full/merged_summary_full.parquet"
    )
    rng = np.random.default_rng(seed=0)
    for tier, n in [("auto_accept", 2), ("manual_review", 2), ("auto_reject", 1)]:
        pool = summary[summary["tier"] == tier]
        sampled = rng.choice(pool["chunk_stem"].values, size=n, replace=False)
        targets.extend(sampled.tolist())

    print(f"[targets] {len(targets)} chunks selected")
    print(
        f"[params] baseline_percentile={args.baseline_percentile}  "
        f"(per-freq-bin temporal baseline subtracted in linear-magnitude domain)"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stem in targets:
        chunk_wav = args.chunks_dir / f"{stem}.wav"
        if not chunk_wav.exists():
            print(f"[skip] {stem} (no WAV)")
            continue
        evs = events[events["chunk_stem"] == stem]
        out_path = args.out_dir / f"{stem}.png"
        render_comparison(chunk_wav, evs, out_path, args.baseline_percentile)
        print(f"[render] {stem}  ({len(evs)} events) -> {out_path.name}")

    print(f"\n[done] {len(targets)} comparison PNGs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
