#!/usr/bin/env python3
"""Build structured labels from manual review annotations.

Maps Shachar's per-detection annotations to the raw candidate CSV
(results/batch_5970/manual_review_all_detections.csv) and outputs
a labeled CSV for hard negative mining and CNN retraining.

Outputs:
  data/manual_review_labels.csv
  data/manual_review_labels.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation rules
# Format: suffix -> (rule_type, rule_data)
#
# "all_noise"         — every detection is noise
# "all_usv"           — every detection is USV
# "usv_at"            — set of 0-based indices that are USV, rest noise
# "first_n_noise"     — first N are noise, rest USV
# "last_n_noise"      — last N are noise, rest USV
# "skip"              — exclude from dataset
# "special"           — needs manual handling (noted in comments)
# ---------------------------------------------------------------------------

ANNOTATIONS = {
    # 491: "all 4 detection at the start are noise the 2 at end are USVs"
    # 6 detections: 0,1,2,3 = noise; 4,5 = USV
    "0000491": ("first_n_noise", 4),

    # 494: "all detections except 2 and 3 are noise"
    # 7 detections: 2,3 = USV; rest = noise
    "0000494": ("usv_at", {2, 3}),

    # 551: "noise"
    "0000551": ("all_noise", None),

    # 1628: "all detections but 4 5 are noise"
    # 8 detections: 4,5 = USV; rest = noise
    "0001628": ("usv_at", {4, 5}),

    # 1700: "all noise"
    "0001700": ("all_noise", None),

    # 1845: "everything but detection 7 I think is noise"
    # 10 detections: 7 = USV; rest = noise
    "0001845": ("usv_at", {7}),

    # 1847: "last detection is noise"
    # 3 detections: 0,1 = USV; 2 = noise
    "0001847": ("last_n_noise", 1),

    # 1930: "all noise"
    "0001930": ("all_noise", None),

    # 1960: "all noise"
    "0001960": ("all_noise", None),

    # 2096: "10 11 are usvs all the others are noise"
    # 25 detections: 10,11 = USV; rest = noise
    "0002096": ("usv_at", {10, 11}),

    # 2273: "noise"
    "0002273": ("all_noise", None),

    # 2361: "all noise"
    "0002361": ("all_noise", None),

    # 2431: "noise"
    "0002431": ("all_noise", None),

    # 2522: "noise"
    "0002522": ("all_noise", None),

    # 3189: "2 3 4 are USVs the others are noise"
    # 5 detections: 2,3,4 = USV; rest = noise
    "0003189": ("usv_at", {2, 3, 4}),

    # 3493: "noise"
    "0003493": ("all_noise", None),

    # 3502: "noise"
    "0003502": ("all_noise", None),

    # 3503: "noise"
    "0003503": ("all_noise", None),

    # 3655: "noise"
    "0003655": ("all_noise", None),

    # 3781: "noise"
    "0003781": ("all_noise", None),

    # 3794: "noise"
    "0003794": ("all_noise", None),

    # 4134: special — "there were 11 USVs there and there is only 4 in the png
    #   so add those to database and have a look at the rejected detection"
    # 4 detections in CSV. User says there should be 11 USVs.
    # For now: mark all 4 as USV, flag for further investigation
    "0004134": ("all_usv", None),  # TODO: investigate missing USVs + rejected detections

    # 4494: "all noise"
    "0004494": ("all_noise", None),

    # 4521: "lets skip"
    "0004521": ("skip", None),

    # 4783: "all detections are USVs"
    "0004783": ("all_usv", None),

    # 4954: "3 usvs 8 11 and 15 all other detections are noise"
    # 23 detections: 3,8,11,15 = USV; rest = noise
    "0004954": ("usv_at", {3, 8, 11, 15}),

    # 5107: "all noise"
    "0005107": ("all_noise", None),

    # 5656: "all detections are noise"
    "0005656": ("all_noise", None),

    # 5843: "all noise"
    "0005843": ("all_noise", None),

    # 5882: "3 and 4 are usvs all other detection are noise"
    # 5 detections: 3,4 = USV; rest = noise
    "0005882": ("usv_at", {3, 4}),

    # 6086: "all noise"
    "0006086": ("all_noise", None),

    # 6113: "first detection is noise the second is usv"
    # 2 detections: 0 = noise; 1 = USV
    "0006113": ("usv_at", {1}),

    # 6394: "4 is usv all others are noise"
    # 12 detections: 4 = USV; rest = noise
    "0006394": ("usv_at", {4}),

    # 6395: "4 is usv all others are noise"
    # 5 detections: 4 = USV; rest = noise
    "0006395": ("usv_at", {4}),
}


def apply_labels(n_events: int, rule_type: str, rule_data) -> dict[int, str]:
    """Apply annotation rule to produce per-event labels."""
    if rule_type == "all_noise":
        return {i: "noise" for i in range(n_events)}
    elif rule_type == "all_usv":
        return {i: "usv" for i in range(n_events)}
    elif rule_type == "usv_at":
        return {i: ("usv" if i in rule_data else "noise") for i in range(n_events)}
    elif rule_type == "first_n_noise":
        return {i: ("noise" if i < rule_data else "usv") for i in range(n_events)}
    elif rule_type == "last_n_noise":
        return {i: ("noise" if i >= (n_events - rule_data) else "usv") for i in range(n_events)}
    elif rule_type == "skip":
        return {}
    else:
        raise ValueError(f"Unknown rule: {rule_type}")


def main():
    csv_path = REPO_ROOT / "results" / "batch_5970" / "manual_review_all_detections.csv"
    df = pd.read_csv(csv_path)
    df["suffix"] = df["stem"].str[-7:]

    output_dir = REPO_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    issues = []

    for suffix, (rule_type, rule_data) in sorted(ANNOTATIONS.items()):
        sub = df[df["suffix"] == suffix].sort_values("detection_idx")

        if sub.empty:
            if rule_type != "skip":
                issues.append(f"{suffix}: not found in CSV")
            continue

        n_events = len(sub)
        labels = apply_labels(n_events, rule_type, rule_data)

        if not labels:  # skip
            continue

        # Validate USV indices in range
        if rule_type == "usv_at" and rule_data:
            max_idx = max(rule_data)
            if max_idx >= n_events:
                issues.append(f"{suffix}: USV index {max_idx} >= n_events {n_events}")

        n_noise = sum(1 for v in labels.values() if v == "noise")
        n_usv = sum(1 for v in labels.values() if v == "usv")
        log.info("%-7s  events=%2d  noise=%2d  usv=%2d", suffix, n_events, n_noise, n_usv)

        for _, row in sub.iterrows():
            idx = row["detection_idx"]
            if idx in labels:
                all_rows.append({
                    "stem": row["stem"],
                    "suffix": suffix,
                    "detection_idx": idx,
                    "label": labels[idx],
                    "start_time_s": row["start_time_s"],
                    "end_time_s": row["end_time_s"],
                    "duration_s": row["duration_s"],
                    "max_probability": row["max_probability"],
                    "mean_probability": row["mean_probability"],
                    "window_count": row["window_count"],
                    "start_window": row["start_window"],
                    "end_window": row["end_window"],
                })

    if issues:
        log.warning("\n=== ISSUES ===")
        for issue in issues:
            log.warning("  %s", issue)

    result = pd.DataFrame(all_rows)
    out_csv = output_dir / "manual_review_labels.csv"
    out_json = output_dir / "manual_review_labels.json"
    result.to_csv(out_csv, index=False)
    result.to_json(out_json, orient="records", indent=2)

    n_noise = (result["label"] == "noise").sum()
    n_usv = (result["label"] == "usv").sum()
    n_files = result["stem"].nunique()

    log.info("\n=== SUMMARY ===")
    log.info("Files labeled: %d", n_files)
    log.info("Total detections labeled: %d", len(result))
    log.info("  Noise: %d", n_noise)
    log.info("  USV:   %d", n_usv)
    log.info("Saved: %s", out_csv)

    # Flag 4134 for investigation
    if "0004134" in {r["suffix"] for r in all_rows}:
        log.info("\n⚠ File 4134: marked 4 detections as USV but user reports 11 USVs.")
        log.info("  TODO: investigate rejected detections and missing USVs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
