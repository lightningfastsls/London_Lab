#!/usr/bin/env python3
"""Collate evaluation results across lab fine-tune experiment variants.

Reads per-variant JSONs (produced by eval_model_on_lab_holdout.py and the
post-hoc filter's holdout_predictions.csv) plus the Phase 5a regression
parquets, and prints a single comparison table.

Usage:
    .venv/bin/python scripts/compare_lab_finetune_variants.py \
        --eval-dir results/lab_finetune_v1_evals/ \
        --regression-dir results/  \
        --output results/lab_finetune_v1_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def fmt_pct(v: float) -> str:
    return f"{100*v:.2f}%"


def load_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("eval")


def load_regression(prod_parquet: Path, new_parquet: Path) -> dict | None:
    if not new_parquet.exists():
        return None
    prod = pd.read_parquet(prod_parquet)
    new = pd.read_parquet(new_parquet)

    def key(df):
        return df["filepath"].str.rsplit("/", n=1).str[-1].str.replace(".wav", "", regex=False)
    prod["stem"] = key(prod)
    new["stem"] = key(new)
    common = set(prod["stem"]) & set(new["stem"])
    prod = prod[prod["stem"].isin(common)]
    new = new[new["stem"].isin(common)]

    tier_prod = prod["tier"].value_counts().to_dict()
    tier_new = new["tier"].value_counts().to_dict()
    delta_events = int(new["n_events"].sum() - prod["n_events"].sum())
    pct_events = (
        100.0 * (new["n_events"].sum() - prod["n_events"].sum())
        / max(1, prod["n_events"].sum())
    )
    return {
        "prod_total_events": int(prod["n_events"].sum()),
        "new_total_events": int(new["n_events"].sum()),
        "delta_events": delta_events,
        "delta_events_pct": pct_events,
        "prod_tiers": tier_prod,
        "new_tiers": tier_new,
        "n_wavs": len(prod),
        "n_wavs_with_disagreement": int(
            (prod.set_index("stem")["n_events"] != new.set_index("stem")["n_events"]).sum()
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="Dir containing <variant>_holdout.json files")
    ap.add_argument("--regression-dir", type=Path, required=True,
                    help="Parent dir of batch_5970_*_regression/ directories")
    ap.add_argument("--prod-summary", type=Path,
                    default=Path("results/batch_5970_v2_full/summary.parquet"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    # Variants we'll look for
    variants = [
        ("run1_pathB_w3.0",  "lab_finetune_v1_run1",                "batch_5970_run1_regression"),
        ("run2_pathB_w1.5",  "lab_finetune_v1_run2",                "batch_5970_finetune_v1_regression"),
        ("run3_pathA",       "lab_finetune_v1_run3_pathA",          "batch_5970_run3_regression"),
        ("run4_pathC_all",   "lab_finetune_v1_run4_pathC_all",      "batch_5970_run4_regression"),
        ("run5_pathC_partial","lab_finetune_v1_run5_pathC_partial", "batch_5970_run5_regression"),
        ("run6_filter",      "lab_finetune_v1_run6_filter",         None),  # Filter has no Phase 5a — uses prod model
    ]

    rows = []
    for label, eval_name, regression_name in variants:
        holdout = load_eval(args.eval_dir / f"{eval_name}_holdout.json")
        regression = (
            load_regression(args.prod_summary,
                            args.regression_dir / regression_name / "summary.parquet")
            if regression_name else None
        )
        rows.append({
            "variant": label,
            "holdout": holdout,
            "regression": regression,
        })

    # Print comparison
    print("=" * 100)
    print("LAB FINE-TUNE VARIANT COMPARISON")
    print("=" * 100)
    print()
    print(f"{'variant':<22} | {'lab-holdout':^30} | {'wild regression':^30}")
    print(f"{'':<22} | {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6}  "
          f"|  {'Δevents':>8} {'Δ%':>7}  {'tiers':<14}")
    print("-" * 100)
    for r in rows:
        v = r["variant"]
        h = r["holdout"]
        reg = r["regression"]

        if h is None:
            holdout_str = " (not run)                  "
        else:
            holdout_str = (
                f"{h['accuracy']:>6.3f} {h['precision']:>6.3f} "
                f"{h['recall']:>6.3f} {h['f1']:>6.3f}"
            )
        if reg is None:
            reg_str = " (n/a — see notes)            "
        else:
            tier_d = ""
            try:
                p = reg["prod_tiers"]; n = reg["new_tiers"]
                tier_d = (
                    f"AA:{n.get('auto_accept',0)}/{p.get('auto_accept',0)} "
                    f"MR:{n.get('manual_review',0)}/{p.get('manual_review',0)}"
                )
            except Exception:
                tier_d = "—"
            reg_str = (
                f"  {reg['delta_events']:>+8} {reg['delta_events_pct']:>+6.2f}%  {tier_d:<14}"
            )

        print(f"{v:<22} | {holdout_str}  | {reg_str}")

    print()
    print("Notes:")
    print("  - lab-holdout: model predictions on 99 lab events from sessions 131209_1000 + 131217_1400")
    print("  - wild regression: lab_finetune_vX events vs production on the 49 wild WAVs")
    print("  - Δevents: new minus prod (negative = our model finds fewer USVs than prod)")
    print("  - tiers AA/MR: auto_accept / manual_review counts (new / prod)")
    print("  - run6_filter: same model as production; filter operates on production's events. ")
    print("    Wild regression for that variant is trivially identical to production.")

    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
