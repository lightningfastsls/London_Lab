"""Render a stratified audit sample of lab `auto_accept` detections.

Reads `merged_events_with_filter.parquet` (post-stationary-band-filter
diagnostic output) and emits 5 buckets of PNGs the user can eyeball:

  cleanest/                SEF == 0.0,  50ms <= dur <= 250ms      (n<=15)
  typical/                 SEF in (0.05, 0.15],  50-250ms          (n<=15)
  borderline/              SEF in (0.20, 0.50]                     (n<=10)
  high_overlap_survivors/  SEF > 0.50  (would-be filter targets)   (all 22)
  long_event_survivors/    dur > 600ms (May-6 max_duration target) (n<=10)

Stratified by `original_filename` so no single recording dominates.

Reuses `render_event` + `find_stationary_bands` from
`scripts/render_filter_validation_pngs.py` so the visual style matches the
existing filter validation PNGs (cyan event bounds, red stationary-band
ticks, SEF/prob/dur title).

Output dir defaults to:
    results/batch_lab_131204_full/audit_2026-05-08/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from render_filter_validation_pngs import (  # noqa: E402
    DEFAULT_CORPUS_FACTS,
    render_event,
)
from usv_spectrogram.postprocessing.hysteresis import HysteresisConfig  # noqa: E402

# Long-event cutoff tracks the production gate (May 2026 lab-noise guard).
# Source of truth: HysteresisConfig.max_duration_ms in postprocessing/hysteresis.py.
LONG_EVENT_THRESHOLD_S = HysteresisConfig().max_duration_ms / 1000.0

# Ad-hoc bucket cutoffs for the audit only — NOT corpus parameters.
# Reference USV duration range is 10-300 ms (per docs/handoffs/2026-05-06_lab-detection-long-event-qc.md);
# we stratify "cleanest"/"typical" to the 50-250 ms core where most real calls sit.
TYPICAL_DUR_MIN_S = 0.05
TYPICAL_DUR_MAX_S = 0.25


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    by: str,
    seed: int,
) -> pd.DataFrame:
    """Round-robin sample across `by` groups until n rows collected."""
    if df.empty or n <= 0:
        return df.head(0)
    if len(df) <= n:
        return df.copy()
    rng_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    groups = {k: g.reset_index(drop=True) for k, g in rng_df.groupby(by, sort=False)}
    keys = list(groups.keys())
    picked = []
    cursor = {k: 0 for k in keys}
    while len(picked) < n:
        progressed = False
        for k in keys:
            if cursor[k] < len(groups[k]):
                picked.append(groups[k].iloc[cursor[k]])
                cursor[k] += 1
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    return pd.DataFrame(picked).reset_index(drop=True)


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
        / "results/batch_lab_131204_full/audit_2026-05-08",
    )
    ap.add_argument("--n-cleanest", type=int, default=15)
    ap.add_argument("--n-typical", type=int, default=15)
    ap.add_argument("--n-borderline", type=int, default=10)
    ap.add_argument("--n-long", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.corpus_facts) as fh:
        facts = json.load(fh)
    nf = facts["noise_filter"]
    contrast_db = float(nf["contrast_db"])
    reject_fraction = float(nf["reject_fraction"])
    print(
        f"[params] algorithm={nf['algorithm']}  contrast_db={contrast_db}  "
        f"reject_fraction={reject_fraction}  [source: {args.corpus_facts.name}]"
    )

    df = pd.read_parquet(args.events_with_filter)
    print(f"[load] {len(df):,} events from {args.events_with_filter.name}")
    aa = df[df["tier"] == "auto_accept"].copy()
    print(f"[filter] {len(aa):,} auto_accept events")

    sef = aa["stationary_energy_fraction"]
    dur = aa["duration_s"]

    cleanest = aa[
        (sef == 0.0) & (dur >= TYPICAL_DUR_MIN_S) & (dur <= TYPICAL_DUR_MAX_S)
    ]
    typical = aa[
        (sef > 0.05) & (sef <= 0.15)
        & (dur >= TYPICAL_DUR_MIN_S) & (dur <= TYPICAL_DUR_MAX_S)
    ]
    borderline = aa[(sef > 0.20) & (sef <= 0.50)]
    high_overlap = aa[sef > 0.50]
    long_event = aa[dur > LONG_EVENT_THRESHOLD_S]

    print(
        "[buckets] population sizes  "
        f"cleanest={len(cleanest)}  typical={len(typical)}  "
        f"borderline={len(borderline)}  high_overlap={len(high_overlap)}  "
        f"long_event={len(long_event)}"
    )

    samples = {
        "cleanest": stratified_sample(
            cleanest, args.n_cleanest, "original_filename", args.seed
        ),
        "typical": stratified_sample(
            typical, args.n_typical, "original_filename", args.seed
        ),
        "borderline": stratified_sample(
            borderline, args.n_borderline, "original_filename", args.seed
        ),
        "high_overlap_survivors": high_overlap.copy(),
        "long_event_survivors": stratified_sample(
            long_event, args.n_long, "original_filename", args.seed
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for label, sub in samples.items():
        out_subdir = args.out_dir / label
        out_subdir.mkdir(parents=True, exist_ok=True)
        if sub.empty:
            print(f"[render] {label}: 0 events (empty bucket)")
            continue
        print(
            f"[render] {label}: {len(sub)} events  "
            f"({sub['original_filename'].nunique()} distinct files)"
        )
        for _, ev in sub.iterrows():
            stem = ev["chunk_stem"]
            chunk_wav = args.chunks_dir / f"{stem}.wav"
            if not chunk_wav.exists():
                print(f"  [skip] {stem} — WAV not found")
                continue
            sef_val = float(ev["stationary_energy_fraction"])
            ev_idx = int(ev["chunk_detection_idx"])
            dur_ms = int(round(ev["duration_s"] * 1000))
            fname = (
                f"{label}_sef{sef_val:.3f}_dur{dur_ms:04d}ms_"
                f"{stem}_ev{ev_idx:03d}.png"
            )
            out_path = out_subdir / fname
            render_event(
                chunk_wav, ev, out_path, contrast_db, label.upper()
            )
            total += 1
            print(
                f"  [{label}] {stem}  ev{ev_idx}  "
                f"SEF={sef_val:.3f}  dur={dur_ms}ms  →  {out_path.name}"
            )

    print(f"\n[done] {total} PNGs written to {args.out_dir}")
    print(
        "Buckets:\n"
        "  cleanest/                — should look like clean USVs\n"
        "  typical/                 — common-case lab events\n"
        "  borderline/              — filter not sure (20-50% SEF)\n"
        "  high_overlap_survivors/  — filter says 'noise', currently auto_accept\n"
        "  long_event_survivors/    — May-6 max_duration_ms=600 would cut these"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
