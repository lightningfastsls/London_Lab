"""Verified FPCA-scores ↔ classified_detections merge (shared by WS-D and WS-E).

This module exists so the inherited join/dedup gotchas from
``docs/handoffs/2026-06-04_ws-de-archetypes-confound-comparison.md`` §1 are
implemented in exactly one place. Both ``shape_archetypes.py`` (WS-D) and
``harmonize_and_compare.py`` (WS-E) import :func:`load_merged_fpca`.

Gotchas encoded here (empirically verified 2026-06-05, Phase 0):

1. **Join key.** FPCA ``(wav_stem, call_id - 1)`` == classified_detections
   ``(wav_stem, det_index)``. The −1 offset is real (det_index is 0-based;
   call_id is the 1-based DeepSqueak call number). Joining on ``id`` (also
   1-based, but a *different* numbering, max 92 vs det_index max 32) gives a
   100%-match-rate but attaches the WRONG detection's metadata to each call —
   match rate does NOT expose the error.

2. **Non-unique FPCA key.** ``(wav_stem, call_id)`` is non-unique within a
   cohort (multiple ridge fragments per call; 5970: 12,098 rows / 7,064 unique).
   We dedupe to one row per call, keeping the largest ``|amp_pc1|`` fragment
   (the WS-B convention). The scores parquet is row-for-row aligned with the
   soft-DTW letters parquet, so positional attach elsewhere stays safe.

3. **Column semantics.** pitch = ``principal_freq_hz``; duration =
   ``call_length_s`` (NOT ``det_duration_ms``). ``mean_power_db`` / ``tonality``
   are cage artifacts — callers must not report them as biology without
   cross-cage calibration.

Cohort sizes (FPCA rows): lab_131204=54,399, 5970=12,098, 9252=506, 3452=334.
3452/9252 are tiny and cage-confounded → wild-vs-wild is a noise floor.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo root = two levels up from this file (scripts/experiments/_fpca_merge.py).
_ROOT = Path(__file__).resolve().parents[2]

FPCA_SCORES_PATH = _ROOT / "models" / "shape_fpca" / "elastic_fpca_scores.parquet"

# cohort -> classified_detections CSV (read-only consumables, per handoff §4).
CLASSIFIED_CSV = {
    "5970": _ROOT / "classified_detections_full.csv",
    "3452": _ROOT / "classified_detections_3452.csv",
    "9252": _ROOT / "classified_detections_9252.csv",
    "lab_131204": _ROOT / "classified_detections_lab_131204_clean.csv",
}

AMP_PCS = ["amp_pc1", "amp_pc2", "amp_pc3", "amp_pc4", "amp_pc5"]
PHASE_PCS = ["phase_pc1", "phase_pc2", "phase_pc3"]
FPCA_FEATURES = AMP_PCS + PHASE_PCS

# Metadata columns we attach from classified_detections (biology-safe set).
PITCH_COL = "principal_freq_hz"
DURATION_COL = "call_length_s"
LABEL_COL = "label"
_META_COLS = [PITCH_COL, DURATION_COL, LABEL_COL]


def load_fpca_scores(dedupe: bool = True) -> pd.DataFrame:
    """Load the elastic-FPCA score table.

    Parameters
    ----------
    dedupe
        If True (default), collapse the non-unique ``(wav_stem, call_id)`` key
        to one row per call, keeping the largest ``|amp_pc1|`` fragment.

    Returns
    -------
    DataFrame with columns ``wav_stem, call_id, cohort, det_index`` +
    :data:`FPCA_FEATURES`. ``det_index`` is added as ``call_id - 1``.
    """
    df = pd.read_parquet(FPCA_SCORES_PATH)
    df["det_index"] = df["call_id"].astype("int64") - 1
    if dedupe:
        df = dedupe_fpca(df)
    return df.reset_index(drop=True)


def dedupe_fpca(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (wav_stem, call_id): the largest ``|amp_pc1|`` fragment."""
    order = df["amp_pc1"].abs()
    return (
        df.assign(_abs_pc1=order)
        .sort_values("_abs_pc1", ascending=False)
        .drop_duplicates(["wav_stem", "call_id"])
        .drop(columns="_abs_pc1")
    )


def load_merged_fpca(
    cohorts: list[str] | None = None,
    dedupe: bool = True,
    require_meta: bool = False,
) -> pd.DataFrame:
    """FPCA scores joined to biology-safe detection metadata, per cohort.

    The join is the verified ``(wav_stem, det_index)`` key (NOT ``id``).

    Parameters
    ----------
    cohorts
        Subset of {'5970','3452','9252','lab_131204'}. None = all four.
    dedupe
        Collapse non-unique FPCA key (default True).
    require_meta
        If True, drop calls whose metadata join missed (default False — keep
        all FPCA rows, leaving metadata NaN so the caller sees the gap).

    Returns
    -------
    DataFrame: FPCA features + ``cohort`` + ``principal_freq_hz``,
    ``call_length_s``, ``label``.
    """
    fp = load_fpca_scores(dedupe=dedupe)
    if cohorts is not None:
        fp = fp[fp["cohort"].isin(cohorts)].copy()

    out = []
    for cohort, g in fp.groupby("cohort"):
        csv = CLASSIFIED_CSV.get(cohort)
        if csv is None or not csv.exists():
            raise FileNotFoundError(f"No classified_detections CSV for cohort {cohort!r}")
        cd = pd.read_csv(csv, low_memory=False)
        cd = cd.dropna(subset=["det_index"]).copy()
        cd["det_index"] = cd["det_index"].astype("int64")
        meta = cd[["wav_stem", "det_index", *_META_COLS]].drop_duplicates(
            ["wav_stem", "det_index"]
        )
        merged = g.merge(meta, on=["wav_stem", "det_index"], how="left", validate="m:1")
        out.append(merged)

    res = pd.concat(out, ignore_index=True)
    if require_meta:
        res = res.dropna(subset=_META_COLS).reset_index(drop=True)
    return res


if __name__ == "__main__":  # pragma: no cover - manual smoke
    df = load_merged_fpca()
    print(f"merged rows: {len(df)}")
    print(df.groupby("cohort").size().to_dict())
    print(
        "meta-missing per cohort:",
        df[df[PITCH_COL].isna()].groupby("cohort").size().to_dict(),
    )
