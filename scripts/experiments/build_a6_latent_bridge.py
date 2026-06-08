"""α₃-C A6 ID-bridge: contour-VAE latents → oracle/γ call_ids.

Problem (two namespaces that do not share a string key):
  * contour-VAE `latents.parquet` (rig) is keyed by **DeepSqueak call_id** —
    an integer (1, 2, …) scoped per (wav_stem, window_idx). It carries the
    learned z_* latents but NO detection-index and NO absolute timing.
  * oracle labels (`labels_vocalmat_v1_on_131204.csv`) + γ hand-labels are
    keyed by `{wav_stem}__det{det_index}` — a **CNN detection index**. A
    different numbering scheme; DeepSqueak call #1 is NOT guaranteed to be
    CNN det0.

The only quantity both namespaces agree on is **time**. So we bridge in two
hops, both validated empirically (see __main__ report):

  1. latents ⋈ patches_manifest  ON (wav_stem, call_id, window_idx)
     → attaches abs_time_start_s / abs_time_end_s to every latent.
     (NOT on `patch_idx`: the combined latents file uses a GLOBAL patch_idx
      while each cohort manifest uses a LOCAL one — joining on it silently
      mixes cohorts. Verified: wav_stem agreement 0.0 on a patch_idx join.)

  2. latents-with-time ⋈ classified_detections  ON wav_stem + max time-overlap
     → assigns each latent the `{wav_stem}__det{det_index}` of the detection
     whose [det_start_s, det_end_s] best overlaps the latent's
     [abs_time_start_s, abs_time_end_s] window. Many-to-one is expected (a
     wide focus window can contain several detections; we take the argmax-
     overlap detection and record the overlap fraction so downstream code can
     threshold ambiguous matches).

Output: a bridge table (one row per lab latent) with z_* + matched call_id +
overlap diagnostics. This is label-independent — it can be built before the
γ hand-labels exist, so A6 becomes a one-shot join once they land.

Eval-validity note: the bridge uses only timing + ids, never the F(t) ridge
that built the encoder input, so it does not contaminate the substrate-
independence rule (handoff "Eval-validity rules" #1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LATENT_COLS = [f"z_{i}" for i in range(32)]
JOIN_KEY = ["wav_stem", "call_id", "window_idx"]


def _interval_overlap(a_start, a_end, b_start, b_end):
    """Overlap length of [a_start,a_end] ∩ [b_start,b_end] (0 if disjoint)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def attach_timing(latents: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Hop 1: latents ⋈ manifest on the natural key → adds abs_time_*."""
    cols = JOIN_KEY + ["abs_time_start_s", "abs_time_end_s"]
    merged = latents.merge(manifest[cols], on=JOIN_KEY, how="inner")
    return merged


def bridge_to_detections(
    latents_t: pd.DataFrame, det: pd.DataFrame
) -> pd.DataFrame:
    """Hop 2: assign each latent the max-overlap detection's call_id.

    Returns the input frame plus columns:
      matched_det_index, matched_call_id, overlap_s, overlap_frac,
      n_overlapping_dets.
    overlap_frac is overlap_s / latent-window-length (how much of the latent
    window the detection explains).
    """
    det = det.copy()
    det["det_index"] = det["det_index"].astype(int)
    det_by_stem = {stem: g for stem, g in det.groupby("wav_stem")}

    out_idx, out_call, out_ov, out_frac, out_n = [], [], [], [], []
    for row in latents_t.itertuples(index=False):
        stem = row.wav_stem
        ls, le = row.abs_time_start_s, row.abs_time_end_s
        win_len = max(le - ls, 1e-9)
        g = det_by_stem.get(stem)
        if g is None:
            out_idx.append(-1); out_call.append(None)
            out_ov.append(0.0); out_frac.append(0.0); out_n.append(0)
            continue
        overlaps = [
            _interval_overlap(ls, le, ds, de)
            for ds, de in zip(g["det_start_s"].values, g["det_end_s"].values)
        ]
        overlaps = np.asarray(overlaps)
        n_ov = int((overlaps > 0).sum())
        best = int(overlaps.argmax())
        best_ov = float(overlaps[best])
        if best_ov <= 0.0:
            out_idx.append(-1); out_call.append(None)
            out_ov.append(0.0); out_frac.append(0.0); out_n.append(0)
            continue
        di = int(g["det_index"].values[best])
        out_idx.append(di)
        out_call.append(f"{stem}__det{di}")
        out_ov.append(best_ov)
        out_frac.append(best_ov / win_len)
        out_n.append(n_ov)

    res = latents_t.copy()
    res["matched_det_index"] = out_idx
    res["matched_call_id"] = out_call
    res["overlap_s"] = out_ov
    res["overlap_frac"] = out_frac
    res["n_overlapping_dets"] = out_n
    return res


def build_bridge(
    latents_path: Path,
    manifest_path: Path,
    detections_path: Path,
    cohort: str = "lab_131204",
) -> pd.DataFrame:
    lat = pd.read_parquet(latents_path)
    lat = lat[lat["cohort"] == cohort].copy()
    man = pd.read_parquet(manifest_path)
    det = pd.read_csv(
        detections_path,
        usecols=["wav_stem", "det_start_s", "det_end_s", "det_index"],
    )
    lat_t = attach_timing(lat, man)
    bridged = bridge_to_detections(lat_t, det)
    keep = JOIN_KEY + [
        "abs_time_start_s", "abs_time_end_s",
        "matched_det_index", "matched_call_id",
        "overlap_s", "overlap_frac", "n_overlapping_dets",
    ] + [c for c in LATENT_COLS if c in bridged.columns]
    return bridged[keep]


def _report(bridge: pd.DataFrame, oracle_path: Path | None) -> str:
    n = len(bridge)
    matched = bridge["matched_call_id"].notna().sum()
    lines = []
    lines.append(f"lab latents (rows)             : {n}")
    lines.append(f"matched to a detection         : {matched} ({matched/n:.1%})")
    lines.append(f"unmatched (no time-overlap)    : {n - matched}")
    m = bridge[bridge["matched_call_id"].notna()]
    lines.append(f"median overlap_frac            : {m['overlap_frac'].median():.3f}")
    lines.append(f"latents w/ ≥2 overlapping dets  : {(m['n_overlapping_dets']>=2).sum()} "
                 f"({(m['n_overlapping_dets']>=2).mean():.1%})")
    # many-to-one: distinct call_ids vs matched rows
    n_call = m["matched_call_id"].nunique()
    lines.append(f"distinct matched call_ids      : {n_call} "
                 f"(many-to-one factor {matched/max(n_call,1):.2f} latents/call)")
    if oracle_path is not None and Path(oracle_path).exists():
        oracle = pd.read_csv(oracle_path, usecols=["call_id"])
        oset = set(oracle["call_id"])
        cset = set(m["matched_call_id"])
        inter = oset & cset
        lines.append("")
        lines.append(f"oracle call_ids (total)        : {len(oset)}")
        lines.append(f"oracle call_ids w/ a latent    : {len(inter)} "
                     f"({len(inter)/len(oset):.1%} oracle coverage)")
        lines.append(f"bridge call_ids not in oracle  : {len(cset - oset)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents", required=True, type=Path,
                    help="contour-VAE latents.parquet (combined, has 'cohort').")
    ap.add_argument("--manifest", required=True, type=Path,
                    help="patches_manifest.parquet for the cohort (has abs_time_*).")
    ap.add_argument("--detections", required=True, type=Path,
                    help="classified_detections_*_clean.csv (wav_stem, det_*).")
    ap.add_argument("--cohort", default="lab_131204")
    ap.add_argument("--oracle-labels", type=Path, default=None,
                    help="optional: oracle CSV to report call_id coverage.")
    ap.add_argument("--out", required=True, type=Path,
                    help="output bridge table (parquet).")
    args = ap.parse_args()

    bridge = build_bridge(args.latents, args.manifest, args.detections, args.cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_parquet(args.out, index=False)

    print(f"=== A6 latent bridge — cohort={args.cohort} ===")
    print(f"wrote {args.out}  ({len(bridge)} rows)")
    print()
    print(_report(bridge, args.oracle_labels))


if __name__ == "__main__":
    main()
