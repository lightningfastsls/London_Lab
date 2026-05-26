"""Interactive labeling-session driver for the lab batch eyeball pass.

Four subcommands:

    pick <N>             Pick 50 events for batch N, render PNGs to
                         to_label/ with numeric prefixes 001_..050_.
    record <N> <STRING>  Parse a 50-char label string (N=noise, U=USV,
                         X=unsure) positionally against the position map,
                         append to labels.csv, archive PNGs.
    record <N>           (Legacy) Read to_label/{noise,real,unsure}/
                         subfolders for drag-drop sorting.
    summary              Print running tally by stratum.

Sample size is fixed at 50 events per batch. Stratification:
  25 events at 250-299 ms (the high-noise contested band)
  25 events at 200-249 ms (the borderline contested band)

Exclusions:
  - Stems already in labels.csv (no re-labeling).
  - Stems from earlier ad-hoc rounds (eyeball_picks*.parquet).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_long_event_spectrograms import _render, WAV_DIR  # noqa: E402

BATCH = Path("results/batch_lab_full_softnotch_20260513_1538")
LAB = BATCH / "eyeball_labeling"
TO_LABEL = LAB / "to_label"
DONE = LAB / "done"
LABELS_CSV = LAB / "labels.csv"

PER_BATCH = 50
STRATA = [
    ("dur_250_299", 25, 250.0, 300.0),
    ("dur_200_249", 25, 200.0, 250.0),
]


def cmd_pick(batch_n: int) -> None:
    events = pd.read_parquet(BATCH / "all_events_with_unmatched_flag.parquet")
    events["couple"] = events["stem"].str.extract(r"(m\d+fm\d+)")[0]
    events["n_tonals"] = events["unmatched_centers_khz"].apply(len)
    events["has_loud_616"] = events["unmatched_centers_khz"].apply(
        lambda lst: any(60.5 < c < 62.5 for c in lst)
    )

    exclude = _already_labeled_or_seen()
    avail = events[~events["stem"].isin(exclude)].copy()

    picks = []
    for stratum, n, lo, hi in STRATA:
        sub = avail[
            (avail["duration_ms"] >= lo)
            & (avail["duration_ms"] < hi)
            & (avail["max_prob"] > 0.9)
        ].copy()
        sub = _stratified_sample(sub, n)
        sub["stratum"] = stratum
        picks.append(sub)
        avail = avail[~avail["stem"].isin(sub["stem"])]

    batch_picks = pd.concat(picks).reset_index(drop=True)
    _render_to(batch_picks, TO_LABEL, batch_n)
    out_parquet = LAB / f"batch_{batch_n:02d}_picks.parquet"
    batch_picks[
        ["stratum", "stem", "duration_ms", "start_s", "end_s", "max_prob",
         "unmatched_centers_khz", "tier", "n_tonals", "has_loud_616"]
    ].to_parquet(out_parquet)
    print(f"Rendered {len(batch_picks)} PNGs into {TO_LABEL}/")
    print(f"Picks parquet: {out_parquet}")
    print()
    print(f"Next: scripts/labeling_batch.py record {batch_n} <50-char-string>")
    print("(N=noise, U=USV, X=unsure; positional, matches 001-050 prefix)")


def cmd_record(batch_n: int, label_string: str | None = None) -> None:
    pick_parquet = LAB / f"batch_{batch_n:02d}_picks.parquet"
    if not pick_parquet.exists():
        sys.exit(f"No picks parquet for batch {batch_n} at {pick_parquet}")
    picks = pd.read_parquet(pick_parquet)
    picks["label"] = ""

    if label_string is not None:
        picks = _label_from_string(picks, label_string, batch_n)
    else:
        # Legacy subfolder mode
        for label in ("noise", "real", "unsure"):
            sub_dir = TO_LABEL / label
            if not sub_dir.exists():
                continue
            for png in sub_dir.glob("*.png"):
                stem = png.stem.rsplit("_", 1)[0]
                stem = stem.split("_", 1)[1] if stem[:3].isdigit() else stem
                picks.loc[picks["stem"] == stem, "label"] = label

    unlabeled = picks[picks["label"] == ""]
    if len(unlabeled):
        print(f"WARNING: {len(unlabeled)} events unlabeled.")

    picks["batch"] = batch_n
    _append_labels(picks[picks["label"] != ""])
    _archive(batch_n)
    print(f"Recorded {(picks['label'] != '').sum()} labels for batch {batch_n}")
    print(f"Archived PNGs to {DONE / f'batch_{batch_n:02d}'}/")
    _print_summary()


def _label_from_string(picks: pd.DataFrame, label_string: str,
                       batch_n: int) -> pd.DataFrame:
    """Map a positional label string (N=noise, U=USV/real, X=unsure) to stems
    using batch_NN_position_map.csv."""
    pos_csv = LAB / f"batch_{batch_n:02d}_position_map.csv"
    if not pos_csv.exists():
        sys.exit(f"Missing position map: {pos_csv}")
    pos_map = pd.read_csv(pos_csv).set_index("position")

    clean = "".join(c for c in label_string.upper() if c in "NUX")
    if len(clean) != len(pos_map):
        print(f"WARNING: cleaned string is {len(clean)} chars, "
              f"expected {len(pos_map)}. Using what we have.")

    code_to_label = {"N": "noise", "U": "real", "X": "unsure"}
    for i, c in enumerate(clean, start=1):
        if i not in pos_map.index:
            break
        stem = pos_map.loc[i, "stem"]
        picks.loc[picks["stem"] == stem, "label"] = code_to_label[c]
    return picks


def cmd_summary() -> None:
    _print_summary()


def _already_labeled_or_seen() -> set:
    seen: set = set()
    if LABELS_CSV.exists():
        seen.update(pd.read_csv(LABELS_CSV)["stem"].tolist())
    for parquet in BATCH.glob("eyeball_picks*.parquet"):
        seen.update(pd.read_parquet(parquet)["stem"].tolist())
    seen |= {
        "131208_1000_m1fm1_chunk_302", "131209_1000_m4fm4_chunk_041",
        "131209_1000_m3fm3_chunk_281", "131209_1000_m6fm6_chunk_005",
        "131209_1000_m3fm3_chunk_189",
    }
    return seen


def _stratified_sample(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    """Round-robin across couples for cross-mouse variety."""
    if len(sub) <= n:
        return sub
    by_couple = {c: g.sample(frac=1, random_state=42)
                 for c, g in sub.groupby("couple")}
    rows: list = []
    couples = list(by_couple.keys())
    i = 0
    while len(rows) < n and any(len(g) > 0 for g in by_couple.values()):
        c = couples[i % len(couples)]
        if len(by_couple[c]) > 0:
            rows.append(by_couple[c].iloc[0])
            by_couple[c] = by_couple[c].iloc[1:]
        i += 1
    return pd.DataFrame(rows).reset_index(drop=True)


def _render_to(picks: pd.DataFrame, out_dir: Path, batch_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    # Sort by stem so the prefix order is deterministic and matches what
    # the file manager will show alphabetically inside each rendered batch.
    sorted_picks = picks.sort_values("stem").reset_index(drop=True)
    for i, ev in enumerate(sorted_picks.itertuples(), start=1):
        wav_path = WAV_DIR / f"{ev.stem}.wav"
        if not wav_path.exists():
            print(f"MISSING: {wav_path}")
            continue
        base_name = f"{ev.stem}_{int(ev.duration_ms)}ms"
        prefixed = f"{i:03d}_{base_name}.png"
        out_path = out_dir / prefixed
        ev_for_render = pd.Series({
            "duration_ms": float(ev.duration_ms),
            "start_s": float(ev.start_s),
            "end_s": float(ev.end_s),
            "max_prob": float(ev.max_prob),
            "unmatched_centers_khz": list(ev.unmatched_centers_khz),
        })
        _render(wav_path, ev_for_render, out_path)
        mapping.append({"position": i, "filename": prefixed, "stem": ev.stem})
    pd.DataFrame(mapping).to_csv(LAB / f"batch_{batch_n:02d}_position_map.csv",
                                  index=False)


def _append_labels(picks_with_labels: pd.DataFrame) -> None:
    cols = ["batch", "stratum", "stem", "duration_ms", "max_prob", "n_tonals",
            "has_loud_616", "tier", "label"]
    df = picks_with_labels[cols].copy()
    header = not LABELS_CSV.exists()
    df.to_csv(LABELS_CSV, mode="a", header=header, index=False)


def _archive(batch_n: int) -> None:
    target = DONE / f"batch_{batch_n:02d}"
    target.mkdir(parents=True, exist_ok=True)
    for label in ("noise", "real", "unsure"):
        for png in (TO_LABEL / label).glob("*.png"):
            shutil.move(str(png), str(target / png.name))
    # Sweep any unlabeled stragglers too.
    for png in TO_LABEL.glob("*.png"):
        shutil.move(str(png), str(target / png.name))


def _print_summary() -> None:
    if not LABELS_CSV.exists():
        print("No labels recorded yet.")
        return
    df = pd.read_csv(LABELS_CSV)
    print(f"=== Labeling summary ({len(df)} events recorded) ===")
    print()
    pivot = pd.crosstab(df["stratum"], df["label"], margins=True)
    print(pivot.to_string())
    print()
    for stratum in sorted(df["stratum"].unique()):
        sub = df[df["stratum"] == stratum]
        n = len(sub)
        noise_rate = (sub["label"] == "noise").mean() * 100
        print(f"  {stratum}: n={n}, noise_rate={noise_rate:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "pick":
        cmd_pick(int(sys.argv[2]))
    elif cmd == "record":
        batch_n = int(sys.argv[2])
        label_string = sys.argv[3] if len(sys.argv) > 3 else None
        cmd_record(batch_n, label_string)
    elif cmd == "summary":
        cmd_summary()
    else:
        sys.exit(f"Unknown command: {cmd}")
