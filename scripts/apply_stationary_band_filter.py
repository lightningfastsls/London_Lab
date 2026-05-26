"""Post-hoc stationary-band noise filter for lab batch detection results.

Idea: detect frequency bins that are 'always on' within each chunk's
spectrogram (the flat horizontal lines characteristic of HVAC hum, ultrasonic
monitors, fluorescent ballast harmonics — equipment noise that the
wild-trained CNN misclassifies as USVs on lab data). For each detected event,
compute the fraction of its energy concentrated in those stationary bins;
events above a threshold are flagged as ``noise_band_overlap``.

The filter is a *diagnostic*: this script does not delete events, it adds
flags so we can see the precision/recall tradeoff before committing to apply
the filter for real.

Tuning knobs (NOT corpus parameters; these are post-hoc filter knobs):

    --stationary-fraction-threshold  fraction of frames a freq bin must be
                                     'active' to be considered stationary.
                                     Higher = stricter (fewer bins flagged).
    --contrast-db                    how much louder than the chunk's global
                                     median the bin must be to count as 'active'.
                                     Higher = stricter (only loud lines flagged).
    --reject-fraction                fraction of an event's energy that must
                                     fall in stationary bins for the event
                                     to be flagged. Higher = stricter (only
                                     events dominated by noise bands flagged).

Outputs:
    results/batch_lab_131204_full/merged_events_with_filter.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from usv_spectrogram.app.core.audio_loader import AudioLoader  # noqa: E402

DEFAULT_STATIONARY_FRACTION = 0.5
DEFAULT_CORPUS_FACTS = REPO_ROOT / "data/corpus_facts/lab_131204.json"


def find_stationary_bands(
    spec_db: np.ndarray, fraction_thr: float, contrast_db: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask_of_stationary_bins, per_bin_floor_excess_db).

    Algorithm v2 (per-bin floor reference):
        bin_floor_db        = temporal 10th percentile of each freq bin
        reference_floor_db  = median across bins of bin_floor_db
        bin is stationary  <=>  bin_floor_db > reference_floor_db + contrast_db

    A tonal narrow-band noise source has an *elevated noise floor* in its
    bin — even when not transiently bright, the bin's quietest level sits
    above the spectrum's typical floor. A USV bin has a normal floor and a
    high p99; only the floor distinguishes the two.

    The ``fraction_thr`` argument is accepted for CLI back-compat but is
    not used in v2. Algorithm v1 used a global-median reference that missed
    the lab data's tonal bands (they were only 1-2 dB above the global
    median; v1 required +6 dB across >50% of frames).
    """
    bin_floor_db = np.percentile(spec_db, 10, axis=1)
    reference_floor_db = float(np.median(bin_floor_db))
    excess_db = bin_floor_db - reference_floor_db
    mask = excess_db > contrast_db
    return mask, excess_db


def event_stationary_fraction(
    spec_db: np.ndarray,
    start_col: int,
    end_col: int,
    stationary_mask: np.ndarray,
) -> float:
    """Fraction of the event's in-band power that falls in stationary bins.

    Uses linear power (10**(dB/10)) so quiet noise floor doesn't dominate
    the fraction. start/end_col are inclusive spectrogram column indices.
    """
    if start_col > end_col or end_col >= spec_db.shape[1]:
        return 0.0
    event_spec_db = spec_db[:, start_col : end_col + 1]
    if event_spec_db.size == 0:
        return 0.0
    event_power = np.power(10.0, event_spec_db / 10.0)
    total = float(event_power.sum())
    if total <= 0.0:
        return 0.0
    stationary = float(event_power[stationary_mask].sum())
    return stationary / total


def time_to_col(t_s: float, spec_times: np.ndarray) -> int:
    if spec_times.size == 0:
        return 0
    idx = int(np.searchsorted(spec_times, t_s))
    return int(np.clip(idx, 0, spec_times.size - 1))


def _process_chunk(task: tuple) -> list[dict]:
    chunk_wav, events, fraction_thr, contrast_db = task
    loader = AudioLoader()
    audio_data = loader.load(chunk_wav)
    spec_db = audio_data.spectrogram_db
    spec_times = audio_data.times

    stationary_mask, fraction_active = find_stationary_bands(
        spec_db, fraction_thr, contrast_db
    )
    n_stationary = int(stationary_mask.sum())

    rows = []
    for ev in events:
        start_t = ev["original_begin_time_s"] - ev["start_s_in_original"]
        end_t = ev["original_end_time_s"] - ev["start_s_in_original"]
        c0 = time_to_col(start_t, spec_times)
        c1 = time_to_col(end_t, spec_times)
        if c1 < c0:
            c0, c1 = c1, c0
        sf = event_stationary_fraction(spec_db, c0, c1, stationary_mask)
        rows.append(
            {
                "chunk_stem": ev["chunk_stem"],
                "chunk_detection_idx": int(ev["chunk_detection_idx"]),
                "stationary_energy_fraction": sf,
                "n_stationary_bins": n_stationary,
            }
        )
    return rows


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
        "--out",
        type=Path,
        default=REPO_ROOT
        / "results/batch_lab_131204_full/merged_events_with_filter.parquet",
    )
    ap.add_argument(
        "--corpus-facts",
        type=Path,
        default=DEFAULT_CORPUS_FACTS,
        help="Layer-2 corpus_facts JSON. Provides contrast_db, reject_fraction, "
        "and algorithm name. CLI flags below override the JSON values.",
    )
    ap.add_argument(
        "--stationary-fraction-threshold",
        type=float,
        default=DEFAULT_STATIONARY_FRACTION,
        help="Vestigial in algorithm v2 (kept for CLI back-compat).",
    )
    ap.add_argument(
        "--contrast-db",
        type=float,
        default=None,
        help="Override corpus_facts.noise_filter.contrast_db.",
    )
    ap.add_argument(
        "--reject-fraction",
        type=float,
        default=None,
        help="Override corpus_facts.noise_filter.reject_fraction.",
    )
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.corpus_facts) as fh:
        facts = json.load(fh)
    nf = facts["noise_filter"]
    expected_alg = "v2_per_bin_floor_excess"
    if nf["algorithm"] != expected_alg:
        raise RuntimeError(
            f"corpus_facts.noise_filter.algorithm={nf['algorithm']!r} but this "
            f"script implements {expected_alg!r}. Refusing to run with mismatched "
            f"algorithm/parameter pair."
        )
    contrast_db = args.contrast_db if args.contrast_db is not None else float(nf["contrast_db"])
    reject_fraction = (
        args.reject_fraction if args.reject_fraction is not None else float(nf["reject_fraction"])
    )
    contrast_src = "CLI override" if args.contrast_db is not None else f"corpus_facts ({args.corpus_facts.name})"
    reject_src = "CLI override" if args.reject_fraction is not None else f"corpus_facts ({args.corpus_facts.name})"

    events = pd.read_parquet(args.events)
    print(f"[load] {len(events)} events from {args.events}")

    grouped = events.groupby("chunk_stem")
    print(f"  spans {grouped.ngroups} unique chunks (chunks with no events skipped)")

    tasks = []
    for stem, grp in grouped:
        chunk_wav = args.chunks_dir / f"{stem}.wav"
        if not chunk_wav.exists():
            continue
        ev_list = grp.to_dict("records")
        tasks.append(
            (
                str(chunk_wav),
                ev_list,
                args.stationary_fraction_threshold,
                contrast_db,
            )
        )

    print(f"[params] algorithm={expected_alg}")
    print(f"[params] contrast_db={contrast_db}  [source: {contrast_src}]")
    print(f"[params] reject_fraction={reject_fraction}  [source: {reject_src}]")
    print(
        f"[note] --stationary-fraction-threshold={args.stationary_fraction_threshold} "
        f"is accepted but unused in v2 (kept for CLI back-compat)"
    )
    print(f"[run] processing {len(tasks)} chunks with {args.workers} workers")

    all_rows: list[dict] = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, rows in enumerate(
                pool.imap_unordered(_process_chunk, tasks, chunksize=20), start=1
            ):
                all_rows.extend(rows)
                if i % 1000 == 0:
                    print(f"  [{i}/{len(tasks)}] chunks processed")
    else:
        for i, t in enumerate(tasks, start=1):
            all_rows.extend(_process_chunk(t))
            if i % 1000 == 0:
                print(f"  [{i}/{len(tasks)}]")

    flag_df = pd.DataFrame(all_rows)
    merged = events.merge(
        flag_df, on=["chunk_stem", "chunk_detection_idx"], how="left"
    )
    merged["noise_band_overlap"] = (
        merged["stationary_energy_fraction"] > reject_fraction
    )

    print("\n=== FILTER IMPACT ===")
    total = len(merged)
    flagged = int(merged["noise_band_overlap"].sum())
    print(f"Total events:                {total}")
    print(f"Flagged (would be rejected): {flagged}  ({100 * flagged / total:.2f}%)")

    print("\nBy tier (chunk-level triage):")
    by_tier = (
        merged.groupby("tier")
        .agg(total=("noise_band_overlap", "size"), flagged=("noise_band_overlap", "sum"))
        .assign(flagged_pct=lambda d: 100.0 * d["flagged"] / d["total"])
    )
    print(by_tier.to_string())

    print("\nBy max_probability bin:")
    bins = [0.0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]
    merged["conf_bin"] = pd.cut(merged["max_probability"], bins=bins, include_lowest=True)
    by_conf = (
        merged.groupby("conf_bin", observed=True)
        .agg(total=("noise_band_overlap", "size"), flagged=("noise_band_overlap", "sum"))
        .assign(flagged_pct=lambda d: 100.0 * d["flagged"] / d["total"])
    )
    print(by_conf.to_string())

    print("\nCross-tab (tier x flagged):")
    print(
        pd.crosstab(
            merged["tier"], merged["noise_band_overlap"], margins=True
        ).to_string()
    )

    print("\nDistribution of stationary_energy_fraction (all events):")
    desc = merged["stationary_energy_fraction"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    print(desc.to_string())

    print("\nPer-chunk n_stationary_bins distribution:")
    per_chunk_stats = merged.groupby("chunk_stem")["n_stationary_bins"].first()
    print(
        per_chunk_stats.describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        ).to_string()
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.drop(columns=["conf_bin"]).to_parquet(args.out, index=False)
    print(f"\n[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
