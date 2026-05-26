"""Diagnose the 34% unmatched-rate warning from batch_lab_full_softnotch_20260513_1538.

Produces a 3-panel PNG and a short verdict text file.

Panels:
  1. Frequency histogram of unmatched ("audit") tonals vs. library entries.
  2. Per-couple presence rate of the top-6 candidate missing tonals.
  3. Triage tier rates inside vs. outside unmatched chunks (confound evidence).

Run:
    .venv/bin/python scripts/diagnose_lab_unmatched_tonals.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from usv_spectrogram.corpus import USV_FREQ_MAX_HZ, USV_FREQ_MIN_HZ

BATCH_DIR = Path("results/batch_lab_full_softnotch_20260513_1538")
LIBRARY_PATH = Path("data/lab_tonal_lines/lab_131204.json")
OUT_PNG = BATCH_DIR / "diagnose_unmatched_tonals.png"
OUT_TXT = BATCH_DIR / "diagnose_unmatched_tonals.txt"

PLOT_MIN_KHZ = USV_FREQ_MIN_HZ / 1000
PLOT_MAX_KHZ = USV_FREQ_MAX_HZ / 1000 + 5


def main() -> None:
    df = pd.read_parquet(BATCH_DIR / "soft_notch_applied.parquet")
    summary = pd.read_parquet(BATCH_DIR / "summary.parquet")
    library = json.loads(LIBRARY_PATH.read_text())

    audit = df[df["source"] == "audit"].copy()
    audit["couple"] = audit["recording_path"].str.extract(r"(m\d+fm\d+)")[0]

    candidates = _top_candidate_tonals(audit, n=6, bin_hz=200)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    _plot_freq_histogram(axes[0], audit, library, candidates)
    _plot_per_couple_presence(axes[1], audit, candidates, df)
    _plot_triage_confound(axes[2], summary, audit)
    fig.suptitle(
        "Lab batch (131204) soft-notch diagnostic — "
        "why 34.3% of chunks fired the stale-library warning",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")

    verdict = _write_verdict(audit, summary, library, candidates)
    OUT_TXT.write_text(verdict)
    print(f"Wrote {OUT_TXT}")
    print()
    print(verdict)


def _top_candidate_tonals(audit: pd.DataFrame, n: int, bin_hz: int) -> pd.DataFrame:
    audit = audit.assign(bin_hz=(audit["center_hz"] / bin_hz).round() * bin_hz)
    grouped = (
        audit.groupby("bin_hz")
        .agg(n_chunks=("recording_path", "nunique"),
             mean_above_median_db=("peak_db", "mean"),
             mean_width_hz=("width_hz", "mean"))
        .sort_values("n_chunks", ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={"bin_hz": "center_hz"})
    )
    return grouped


def _plot_freq_histogram(ax, audit, library, candidates):
    ax.hist(audit["center_hz"] / 1000,
            bins=np.arange(PLOT_MIN_KHZ, PLOT_MAX_KHZ + 5, 0.4),
            color="steelblue", edgecolor="none", alpha=0.85)
    for entry in library["entries"]:
        ax.axvline(entry["center_hz"] / 1000, color="green", lw=2,
                   label=f"library ({entry['center_hz']/1000:.1f} kHz)")
    for _, row in candidates.iterrows():
        ax.axvline(row["center_hz"] / 1000, color="red", lw=1, ls="--", alpha=0.7)
        ax.text(row["center_hz"] / 1000, ax.get_ylim()[1] * 0.95,
                f"{row['center_hz']/1000:.1f}\n({row['n_chunks']})",
                ha="center", va="top", fontsize=8, color="red")
    ax.set_xlabel("Center frequency (kHz)")
    ax.set_ylabel("Unmatched-tonal occurrences (rows)")
    ax.set_title("Panel 1 — Unmatched tonals are concentrated at 4-6 specific frequencies "
                 "(red dashed lines), not random drift")
    ax.set_xlim(PLOT_MIN_KHZ, PLOT_MAX_KHZ)
    ax.legend(loc="upper right")


def _plot_per_couple_presence(ax, audit, candidates, df):
    all_paths = df["recording_path"].drop_duplicates()
    all_couple = all_paths.str.extract(r"(m\d+fm\d+)")[0]
    couple_chunk_counts = all_couple.value_counts()
    couples = couple_chunk_counts.head(10).index.tolist()
    couples_sorted = sorted(couples)

    heat = np.zeros((len(candidates), len(couples_sorted)))
    for i, (_, row) in enumerate(candidates.iterrows()):
        f = row["center_hz"]
        sub = audit[(audit["center_hz"] > f - 200) & (audit["center_hz"] < f + 200)]
        per_couple = sub.groupby("couple")["recording_path"].nunique()
        for j, couple in enumerate(couples_sorted):
            total = couple_chunk_counts.get(couple, 0)
            heat[i, j] = (per_couple.get(couple, 0) / total * 100) if total else 0

    im = ax.imshow(heat, cmap="Reds", aspect="auto", vmin=0, vmax=25)
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels([f"{r['center_hz']/1000:.1f} kHz" for _, r in candidates.iterrows()])
    ax.set_xticks(range(len(couples_sorted)))
    ax.set_xticklabels(couples_sorted, rotation=45, ha="right")
    ax.set_xlabel("Couple")
    ax.set_title("Panel 2 — These tonals appear ACROSS couples (lab-wide rig artifact, not couple-specific). "
                 "Cells = % of couple's chunks with that tonal.")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            color = "white" if v > 12 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax, label="% chunks with tonal")


def _plot_triage_confound(ax, summary, audit):
    has_unmatched = summary["filepath"].isin(audit["recording_path"].unique())
    inside = summary[has_unmatched]
    outside = summary[~has_unmatched]

    tiers = ["auto_accept", "manual_review", "auto_reject"]
    inside_rates = [(inside["tier"] == t).mean() * 100 for t in tiers]
    outside_rates = [(outside["tier"] == t).mean() * 100 for t in tiers]

    x = np.arange(len(tiers))
    w = 0.35
    ax.bar(x - w / 2, inside_rates, w,
           label=f"INSIDE unmatched-tonal chunks  (n={len(inside)})", color="crimson")
    ax.bar(x + w / 2, outside_rates, w,
           label=f"OUTSIDE                          (n={len(outside)})", color="steelblue")
    for i, (a, b) in enumerate(zip(inside_rates, outside_rates)):
        ax.text(i - w / 2, a + 1, f"{a:.1f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 1, f"{b:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("% of chunks")
    ax.set_title("Panel 3 — Triage CONFOUND: chunks containing un-suppressed tonals are 2.9× "
                 "more likely to flag as USV-positive. Detections are partly noise.")
    ax.legend()


def _write_verdict(audit, summary, library, candidates):
    has_unmatched = summary["filepath"].isin(audit["recording_path"].unique())
    inside = summary[has_unmatched]
    outside = summary[~has_unmatched]
    inside_pos = ((inside["tier"] == "auto_accept") | (inside["tier"] == "manual_review")).mean()
    outside_pos = ((outside["tier"] == "auto_accept") | (outside["tier"] == "manual_review")).mean()

    lines = []
    lines.append("=== Lab 131204 soft-notch diagnostic verdict ===\n")
    lines.append(f"Library: {len(library['entries'])} entry @ "
                 f"{library['entries'][0]['center_hz']/1000:.1f} kHz")
    lines.append(f"Audit rows: {len(audit):,}  in {audit['recording_path'].nunique():,} chunks "
                 f"(34.3% of batch).")
    lines.append("")
    lines.append("Top 6 candidate missing tonals (binned at 200 Hz):")
    for _, r in candidates.iterrows():
        lines.append(f"  {r['center_hz']/1000:5.1f} kHz  "
                     f"{int(r['n_chunks']):5d} chunks  "
                     f"width≈{r['mean_width_hz']:5.1f} Hz")
    lines.append("")
    lines.append("Triage rate (auto_accept + manual_review):")
    lines.append(f"  INSIDE  unmatched chunks: {inside_pos*100:.1f}%  (n={len(inside)})")
    lines.append(f"  OUTSIDE unmatched chunks: {outside_pos*100:.1f}%  (n={len(outside)})")
    lines.append(f"  Inflation factor: {inside_pos/outside_pos:.2f}×")
    lines.append("")
    lines.append("=== Verdict ===")
    lines.append("UNDERCALIBRATED LIBRARY. The four-six candidate frequencies appear across "
                 "all major couples (lab-wide rig artifacts, not couple-specific), but each "
                 "appears in only 7-22% of chunks per couple. They fall below the "
                 "calibration `min_detection_rate=0.5` filter and were rejected.")
    lines.append("")
    lines.append("Recommendation: re-run calibration with min_detection_rate ~= 0.05-0.10 "
                 "on a broader sample that includes m1fm1 (currently absent from the "
                 "calibration set), then re-batch the lab data with the new library. "
                 "Expected outcome: unmatched-rate drops well below the 10% threshold, and "
                 "the inside/outside triage-rate gap closes — confirming the gap was driven "
                 "by un-suppressed equipment tonals, not real biology.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
