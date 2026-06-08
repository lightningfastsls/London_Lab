"""Loader for the shape-invariance benchmark.

`load_labeled()` returns arrays aligned to the 611 human-labeled rows (join via
the SPEC `build_join`, offset -1, dropping shape_label=='unclear').
`load_full()` returns the same per-call fields for the whole 67,337-call corpus
(minus human labels) for optional full-corpus encoding.

DATA PROVENANCE NOTE (verified 2026-06-07, deviates from the handoff text):
  The handoff claims "ALL 611 labels are cohort lab_131204; wild UNLABELED, so
  cross-cohort stratification is VACUOUS." This is STALE. The actual
  `data/manual_shape_labels.csv` is the Phase-2-EXPANDED gold set: 758 rows
  spanning lab (17 m*fm* pairings) AND three wild cohorts. After join+drop the
  611 labeled rows are cohort {lab_131204:182, 5970:204, 9252:140, 3452:85}.
  Cross-cohort stratification is therefore REAL, not vacuous. `cohort` (4 levels)
  is the meaningful cage/stratum axis; `pairing` (parsed from wav_stem) is the
  finer lab-internal proxy but is degenerate for wild (mostly singleton stems).
  The harness uses `cohort` as the within-stratum field by default.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Reuse the SPEC join/family functions (never reimplement).
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/experiments
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from eval_shape_human_anchored import build_join, group_family  # noqa: E402

# Stable absolute data paths (the harness default points at an ephemeral job dir).
META_NPZ = "/home/shachar/.claude/jobs/b619c2bb/tmp/shape_data/true_registered_ridges_meta.npz"
LAB_NPZ = "/home/shachar/.claude/jobs/b619c2bb/tmp/shape_data/true_registered_ridges.npz"
HUMAN_CSV = "data/manual_shape_labels.csv"


def _parse_pairing(wav_stem: str) -> str:
    """Lab stems look like '131204_1400_m1fm1_chunk_033' -> 'm1fm1'.

    Wild stems are numeric ('0005418_...') with no pairing token -> return the
    leading numeric id as a (mostly-singleton) pseudo-pairing. `cohort` is the
    field to actually stratify on; pairing is a finer lab-internal proxy.
    """
    parts = str(wav_stem).split("_")
    if len(parts) >= 3 and parts[2].startswith("m") and "fm" in parts[2]:
        return parts[2]
    return parts[0]


def _resample128(contour50: np.ndarray) -> np.ndarray:
    """Cubic-spline resample (N,50)->(N,128) on a normalized [0,1] parameter."""
    x = np.linspace(0.0, 1.0, contour50.shape[1])
    xq = np.linspace(0.0, 1.0, 128)
    f = interp1d(x, contour50, kind="cubic", axis=1, assume_sorted=True)
    return f(xq).astype(np.float32)


def _side_channels(shapes: np.ndarray, duration_ms: np.ndarray):
    """Per-call scalar side-channels (handoff rule 2). Returns (N,3) raw:
    [duration_ms, freq_range = max-min, freq_std]. z-scoring happens in harness.
    """
    freq_range = shapes.max(1) - shapes.min(1)
    freq_std = shapes.std(1)
    side = np.column_stack([duration_ms, freq_range, freq_std]).astype(np.float64)
    return side, freq_range.astype(np.float64), freq_std.astype(np.float64)


def load_full(meta_npz: str = META_NPZ, lab_npz: str = LAB_NPZ, verbose: bool = True):
    """Whole 67,337-call corpus, per-call fields (no human labels).

    Returns a dict: rows(=arange), contour50, contour128, duration_ms,
    freq_range, freq_std, cohort, pairing, side, lab_shape.
    """
    m = np.load(meta_npz, allow_pickle=True)
    L = np.load(lab_npz, allow_pickle=True)
    shapes = m["shapes"].astype(np.float64)
    assert L["shapes"].shape == shapes.shape, "meta/lab row mismatch"
    ws = m["wav_stem"].astype(str)
    cohort = m["cohort"].astype(str)
    duration_ms = L["duration"].astype(np.float64)
    side, freq_range, freq_std = _side_channels(shapes, duration_ms)
    pairing = np.array([_parse_pairing(s) for s in ws])
    out = {
        "rows": np.arange(len(shapes)),
        "contour50": shapes,
        "contour128": _resample128(shapes),
        "duration_ms": duration_ms,
        "freq_range": freq_range,
        "freq_std": freq_std,
        "cohort": cohort,
        "pairing": pairing,
        "wav_stem": ws,
        "side": side,
        "lab_shape": L["lab_shape"],
    }
    if verbose:
        print(f"[loader.load_full] corpus = {shapes.shape}; "
              f"cohorts = {dict(zip(*np.unique(cohort, return_counts=True)))}")
    return out


def load_labeled(meta_npz: str = META_NPZ, lab_npz: str = LAB_NPZ,
                 human_csv: str = HUMAN_CSV, verbose: bool = True):
    """Arrays aligned to the human-labeled rows (join offset -1, drop 'unclear').

    Returns a dict with keys: rows, contour50 (N,50), contour128 (N,128),
    duration_ms, freq_range, freq_std, cohort, pairing, family, shape_label
    (raw), lab_shape, side (N,3), and `stratum` (== cohort; the harness's
    within-stratum field).
    """
    m = np.load(meta_npz, allow_pickle=True)
    L = np.load(lab_npz, allow_pickle=True)
    shapes_all = m["shapes"].astype(np.float64)
    assert L["shapes"].shape == shapes_all.shape, "meta/lab row mismatch"
    ws = m["wav_stem"].astype(str)
    cid = m["call_id"]
    cohort_all = m["cohort"].astype(str)
    duration_all = L["duration"].astype(np.float64)

    h = pd.read_csv(human_csv)
    rows, joined = build_join(ws, cid, h, offset=-1)
    y_raw = joined["shape_label"].to_numpy()
    keep = ~np.isin(y_raw, ["unclear"])
    rows = rows[keep]
    shape_label = y_raw[keep]
    family = np.array([group_family(v) for v in shape_label])

    shapes = shapes_all[rows]
    duration_ms = duration_all[rows]
    side, freq_range, freq_std = _side_channels(shapes, duration_ms)
    cohort = cohort_all[rows]
    pairing = np.array([_parse_pairing(ws[r]) for r in rows])

    out = {
        "rows": rows,
        "contour50": shapes,
        "contour128": _resample128(shapes),
        "duration_ms": duration_ms,
        "freq_range": freq_range,
        "freq_std": freq_std,
        "cohort": cohort,
        "pairing": pairing,
        "stratum": cohort,  # harness within-stratum field (cage axis)
        "family": family,
        "shape_label": shape_label,
        "lab_shape": L["lab_shape"][rows],
        "side": side,
    }
    if verbose:
        fam_counts = {k: int(v) for k, v in pd.Series(family).value_counts().items()}
        coh_counts = {str(k): int(v) for k, v in zip(*np.unique(cohort, return_counts=True))}
        print(f"[loader.load_labeled] matched {len(joined)}/{len(h)} human labels (offset -1); "
              f"N after drop 'unclear' = {len(family)}")
        print(f"[loader.load_labeled] family counts = {fam_counts}")
        print(f"[loader.load_labeled] cohort (stratum) counts = {coh_counts}")
        print(f"[loader.load_labeled] DEVIATION from handoff: labels span 4 cohorts, "
              f"NOT lab-only -> cross-cohort stratification is REAL.")
        n_pair = len(np.unique(pairing))
        print(f"[loader.load_labeled] distinct pairings = {n_pair} (lab m*fm* + wild singleton stems)")
    return out
