"""Render PNGs to visually validate the stationary-band noise filter.

Two output folders:
- flagged/         — events with stationary_energy_fraction > reject_fraction
                     (what the filter says to remove)
- near_threshold/  — top-N events below the cutoff but above a near-threshold
                     floor (what the filter *almost* flagged)

Each PNG shows:
- The host chunk's spectrogram (magma).
- The event's time bounds as cyan vertical lines / shaded span.
- The stationary frequency bins (filter's targets) as red ticks on the
  right edge of the spectrogram, so you can see the lines the filter is
  reacting to.
- A title printing: chunk_stem, event_idx, SEF, max_probability, tier,
  n_stationary_bins, and the parameter source.

Validation question we're answering:
- For flagged events: are these actually noise-driven false positives?
- For near-threshold events: do these look like noise the filter
  *should have* caught? If yes → reject_fraction cutoff too high.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from usv_spectrogram.app.core.audio_loader import AudioLoader  # noqa: E402

DEFAULT_CORPUS_FACTS = REPO_ROOT / "data/corpus_facts/lab_131204.json"


def find_stationary_bands(spec_db: np.ndarray, contrast_db: float) -> np.ndarray:
    bin_floor_db = np.percentile(spec_db, 10, axis=1)
    reference_floor_db = float(np.median(bin_floor_db))
    return bin_floor_db - reference_floor_db > contrast_db


def render_event(
    chunk_wav: Path,
    event_row: pd.Series,
    out_path: Path,
    contrast_db: float,
    label_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader = AudioLoader()
    audio = loader.load(chunk_wav)
    spec_db = audio.spectrogram_db
    spec_times = audio.times
    freqs_hz = audio.frequencies

    stationary_mask = find_stationary_bands(spec_db, contrast_db)
    stationary_freqs_khz = freqs_hz[stationary_mask] / 1000.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=140)
    extent = [
        spec_times[0],
        spec_times[-1],
        freqs_hz[0] / 1000.0,
        freqs_hz[-1] / 1000.0,
    ]
    vmin = float(np.percentile(spec_db, 5))
    vmax = float(np.percentile(spec_db, 99))
    ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    start_t = event_row["original_begin_time_s"] - event_row["start_s_in_original"]
    end_t = event_row["original_end_time_s"] - event_row["start_s_in_original"]
    ax.axvspan(start_t, end_t, alpha=0.20, color="cyan", linewidth=0)
    ax.axvline(start_t, color="cyan", linewidth=1.0, alpha=0.9)
    ax.axvline(end_t, color="cyan", linewidth=1.0, alpha=0.9)

    for fk in stationary_freqs_khz:
        ax.plot(
            [spec_times[-1] - 0.02, spec_times[-1]],
            [fk, fk],
            color="red",
            linewidth=2.0,
            alpha=0.9,
            solid_capstyle="butt",
        )

    title = (
        f"[{label_prefix}] {event_row['chunk_stem']}  ev_idx={int(event_row['chunk_detection_idx'])}\n"
        f"SEF={event_row['stationary_energy_fraction']:.3f}  "
        f"max_prob={event_row['max_probability']:.3f}  "
        f"tier={event_row['tier']}  "
        f"n_stat_bins={int(event_row['n_stationary_bins'])}  "
        f"dur={event_row['duration_s']*1000:.0f}ms"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time in chunk (s)")
    ax.set_ylabel("Freq (kHz)  (red ticks at right = stationary bins)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--events-with-filter",
        type=Path,
        default=REPO_ROOT
        / "results/batch_lab_131204_full/merged_events_with_filter.parquet",
    )
    ap.add_argument(
        "--chunks-dir",
        type=Path,
        default=REPO_ROOT / "USV_lab_131204_chunked_2s_full",
    )
    ap.add_argument(
        "--corpus-facts",
        type=Path,
        default=DEFAULT_CORPUS_FACTS,
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "results/batch_lab_131204_full/filter_validation",
    )
    ap.add_argument(
        "--n-near-threshold",
        type=int,
        default=15,
        help="Number of near-threshold events (top SEF below the cutoff) to render.",
    )
    ap.add_argument(
        "--near-threshold-floor",
        type=float,
        default=0.20,
        help="Lower bound for near-threshold SEF — events with SEF in [floor, cutoff) are candidates.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.corpus_facts) as fh:
        facts = json.load(fh)
    nf = facts["noise_filter"]
    contrast_db = float(nf["contrast_db"])
    reject_fraction = float(nf["reject_fraction"])

    print(f"[params] algorithm={nf['algorithm']}  contrast_db={contrast_db}  "
          f"reject_fraction={reject_fraction}  [source: {args.corpus_facts.name}]")

    df = pd.read_parquet(args.events_with_filter)
    print(f"[load] {len(df):,} events from {args.events_with_filter.name}")

    flagged = df[df["noise_band_overlap"]].copy()
    flagged = flagged.sort_values("stationary_energy_fraction", ascending=False)

    near = df[
        (~df["noise_band_overlap"])
        & (df["stationary_energy_fraction"] >= args.near_threshold_floor)
    ].copy()
    near = near.sort_values("stationary_energy_fraction", ascending=False).head(
        args.n_near_threshold
    )

    print(f"[group:flagged]         {len(flagged):>4} events  "
          f"(SEF > {reject_fraction})")
    print(f"[group:near_threshold]  {len(near):>4} events  "
          f"(SEF in [{args.near_threshold_floor}, {reject_fraction}); "
          f"top {args.n_near_threshold} by SEF)")

    flagged_dir = args.out_dir / "flagged"
    near_dir = args.out_dir / "near_threshold"
    flagged_dir.mkdir(parents=True, exist_ok=True)
    near_dir.mkdir(parents=True, exist_ok=True)

    def emit(group_df: pd.DataFrame, out_subdir: Path, label: str) -> int:
        n_rendered = 0
        for _, ev in group_df.iterrows():
            stem = ev["chunk_stem"]
            chunk_wav = args.chunks_dir / f"{stem}.wav"
            if not chunk_wav.exists():
                print(f"  [skip] {stem} — WAV not found")
                continue
            sef = ev["stationary_energy_fraction"]
            ev_idx = int(ev["chunk_detection_idx"])
            fname = f"{label}_sef{sef:.3f}_{stem}_ev{ev_idx:03d}.png"
            out_path = out_subdir / fname
            render_event(chunk_wav, ev, out_path, contrast_db, label)
            n_rendered += 1
            print(f"  [render] {label} {stem}  ev{ev_idx}  SEF={sef:.3f}  →  {out_path.name}")
        return n_rendered

    n_flagged = emit(flagged, flagged_dir, "FLAGGED")
    n_near = emit(near, near_dir, "NEAR")

    print(f"\n[done] flagged: {n_flagged} PNGs in {flagged_dir}")
    print(f"[done] near_threshold: {n_near} PNGs in {near_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
