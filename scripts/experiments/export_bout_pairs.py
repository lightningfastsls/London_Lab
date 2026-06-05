"""WS-B Step 1 — build the within-bout adjacent-pair stream for grammar TE.

Joins elastic-FPCA shape coordinates (amp_pc1, amp_pc2) to per-call pitch /
duration / timing features, orders calls within each wav_stem by absolute time,
segments into bouts (silence-bounded runs), and emits within-bout adjacent
(call_t, call_t+1) pairs plus full per-call ordered series for TE.

JOIN (verified, the flagged gotcha):
    identity = (wav_stem, call_id - 1) == (wav_stem, det_index)
    - det_index is 0-based per wav_stem (max 32); call_id is 1-based (max 33).
    - Do NOT join on `id` (1-based DeepSqueak call number, max 92 — wrong).

DEDUPE RULES (documented, deterministic):
    - FPCA side: a call_id may carry >1 ridge fragment (DeepSqueak focus-STFT can
      split one call into multiple contour segments). Keep ONE ridge per
      (wav_stem, call_id): the fragment with the largest |amp_pc1| (most
      energetic primary shape), ties broken by lowest amp_pc2 then row order.
    - Detection side: (wav_stem, det_index) can repeat (279 in 5970). Keep the
      row with the highest det_prob_max, ties by lowest begin_time_s.
    A many-to-one merge would silently duplicate features — both sides are made
    key-unique BEFORE merging.

PITCH/DURATION/TIMING (canonical):
    pitch    = principal_freq_hz
    duration = call_length_s          (NOT det_duration_ms — differ up to 10x)
    gap      = begin_time_s[t+1] - end_time_s[t]   (silent inter-call interval)
    mean_power_db / tonality are CAGE artifacts — never used.

Ordering: within each wav_stem, sort by begin_time_s. Pairs are formed ONLY
within a bout and ONLY within a wav_stem (never across files).

Bouts: usv_language.analysis.sequence_analysis.segment_into_bouts, run at BOTH
0.25 s (Stream-5 MI plateau) and 0.6 s (corpus_facts 3x median IOI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from usv_language.analysis.sequence_analysis import segment_into_bouts  # noqa: E402

FPCA_PATH = REPO / "models/shape_fpca/elastic_fpca_scores.parquet"
CSV_BY_COHORT = {
    "5970": REPO / "classified_detections_full.csv",
    "lab_131204": REPO / "classified_detections_lab_131204_clean.csv",
    "3452": REPO / "classified_detections_3452.csv",
    "9252": REPO / "classified_detections_9252.csv",
}
SHAPE_COLS = ["amp_pc1", "amp_pc2"]
PITCH_COL = "principal_freq_hz"
DUR_COL = "call_length_s"
BOUT_THRESHOLDS = (0.25, 0.6)


def dedupe_fpca(fp_cohort: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One ridge per (wav_stem, call_id): largest |amp_pc1|, ties low amp_pc2."""
    df = fp_cohort.copy()
    df["_abs_pc1"] = df["amp_pc1"].abs()
    df = df.sort_values(
        ["wav_stem", "call_id", "_abs_pc1", "amp_pc2"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    n_before = len(df)
    df = df.drop_duplicates(["wav_stem", "call_id"], keep="first").drop(columns="_abs_pc1")
    return df, n_before - len(df)


def dedupe_dets(d: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per (wav_stem, det_index): highest det_prob_max, ties low begin_time."""
    df = d.copy()
    prob = df["det_prob_max"] if "det_prob_max" in df.columns else pd.Series(0.0, index=df.index)
    df = df.assign(_prob=prob.fillna(-1.0))
    df = df.sort_values(
        ["wav_stem", "det_index", "_prob", "begin_time_s"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    n_before = len(df)
    df = df.drop_duplicates(["wav_stem", "det_index"], keep="first").drop(columns="_prob")
    return df, n_before - len(df)


def build_cohort_calls(cohort: str, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return the joined per-call table (one row per matched call) + coverage stats."""
    fp = pd.read_parquet(FPCA_PATH)
    fpc = fp[fp.cohort == cohort].copy()
    fpc["wav_stem"] = fpc["wav_stem"].astype(str)
    fpc["det_index"] = (fpc["call_id"] - 1).astype(int)
    n_ridges = len(fpc)
    fpc, n_fp_dropped = dedupe_fpca(fpc)

    d = pd.read_csv(CSV_BY_COHORT[cohort])
    d = d.dropna(subset=["file", "det_index"]).copy()
    d["wav_stem"] = d["wav_stem"].astype(str)
    d["det_index"] = d["det_index"].astype(int)
    n_dets = len(d)
    d, n_det_dropped = dedupe_dets(d)

    keep = ["wav_stem", "det_index", PITCH_COL, DUR_COL, "begin_time_s", "end_time_s"]
    merged = fpc.merge(d[keep], on=["wav_stem", "det_index"], how="inner")
    # require non-null pitch/duration/shape
    need = SHAPE_COLS + [PITCH_COL, DUR_COL, "begin_time_s", "end_time_s"]
    merged = merged.dropna(subset=need).reset_index(drop=True)

    n_joined = len(merged)
    stats = dict(
        cohort=cohort,
        n_ridges=n_ridges,
        n_fpca_dedup_dropped=n_fp_dropped,
        n_detections=n_dets,
        n_det_dedup_dropped=n_det_dropped,
        n_joined=n_joined,
        coverage_pct=round(100.0 * n_joined / max(1, len(fpc)), 1),
    )
    if verbose:
        print(
            f"  [{cohort}] ridges={n_ridges} (fpca-dedup drop {n_fp_dropped}) | "
            f"detections={n_dets} (det-dedup drop {n_det_dropped}) | "
            f"joined={n_joined} | coverage={stats['coverage_pct']}% of unique ridges"
        )
    return merged, stats


def build_bout_series(calls: pd.DataFrame, bout_threshold_s: float) -> list[dict]:
    """Per wav_stem: order by begin_time, segment into bouts, return per-bout series.

    Each returned dict has aligned arrays (one entry per call in the bout) for:
    amp_pc1, amp_pc2, pitch, duration, gap_to_next, plus wav_stem & bout_id.
    Gaps array has length len(bout) with the last gap = NaN (no successor).
    """
    bouts: list[dict] = []
    bid = 0
    for stem, g in calls.groupby("wav_stem", sort=True):
        g = g.sort_values("begin_time_s", kind="mergesort").reset_index(drop=True)
        if len(g) == 0:
            continue
        starts = g["begin_time_s"].to_numpy(float)
        ends = g["end_time_s"].to_numpy(float)
        # silent inter-call gap = start(next) - end(current)
        ici = starts[1:] - ends[:-1] if len(g) > 1 else np.array([])
        codes = np.arange(len(g))  # placeholder codes; we only need the segmentation
        segs = segment_into_bouts(codes, ici, bout_threshold_s)
        cursor = 0
        for seg in segs:
            m = len(seg)
            idx = slice(cursor, cursor + m)
            sub = g.iloc[idx]
            within_gap = starts[cursor + 1: cursor + m] - ends[cursor: cursor + m - 1]
            gap_arr = np.append(within_gap, np.nan) if m >= 1 else np.array([])
            bouts.append(dict(
                wav_stem=stem,
                bout_id=bid,
                amp_pc1=sub["amp_pc1"].to_numpy(float),
                amp_pc2=sub["amp_pc2"].to_numpy(float),
                pitch=sub[PITCH_COL].to_numpy(float),
                duration=sub[DUR_COL].to_numpy(float),
                gap=gap_arr,
            ))
            bid += 1
            cursor += m
    return bouts


def count_pairs(bouts: list[dict]) -> int:
    return int(sum(max(0, len(b["pitch"]) - 1) for b in bouts))


if __name__ == "__main__":
    print("=== WS-B Step 1: within-bout pair export ===")
    print(f"FPCA: {FPCA_PATH}")
    print(f"shape cols={SHAPE_COLS} pitch={PITCH_COL} duration={DUR_COL}")
    print(f"bout thresholds (s) = {BOUT_THRESHOLDS}")
    for coh in CSV_BY_COHORT:
        calls, st = build_cohort_calls(coh)
        for bt in BOUT_THRESHOLDS:
            bouts = build_bout_series(calls, bt)
            multi = [b for b in bouts if len(b["pitch"]) >= 2]
            print(
                f"     thr={bt}s: bouts={len(bouts)} multi-call bouts={len(multi)} "
                f"within-bout pairs={count_pairs(bouts)}"
            )
