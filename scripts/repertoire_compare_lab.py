#!/usr/bin/env python3
"""Tier-aware and couple-aware repertoire comparisons for the lab batch.

Produces per-cohort Scattoni-7 syllable proportions, chi-square tests of
independence, and stacked-bar visualizations for the three Phase 2B guard
variants from the 2026-05-14 post-labeling handoff:

  1. tier            — auto_accept vs manual_review (handoff finding 2)
  2. couple_keep_set — 13 retained vs 4 noise-prone {m1fm1, m1fm2, m1fm4,
                       m3fm3} (handoff finding 3)
  3. couple          — all 17 couples (descriptive baseline)

Bypasses ``analyze_repertoire.py`` because its PERMANOVA test assumes a 1:1
animal_id↔population mapping which the per-event ``tier`` column violates.

Inputs : results/traditional_taxonomy_lab_131204/classified_traditional.csv
Outputs: results/repertoire_lab_131204/
           by_tier.{csv,png}
           by_couple_keep_set.{csv,png}
           by_couple.{csv,png}
           summary.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SYLLABLE_ORDER = ["Short", "Flat", "Up", "Down", "Chevron", "Complex", "Frequency_Jump"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/traditional_taxonomy_lab_131204/classified_traditional.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/repertoire_lab_131204"),
    )
    return parser.parse_args()


def compare(df: pd.DataFrame, group_col: str, name: str, out_dir: Path) -> dict:
    """Cross-tab + chi-square + stacked bar for one grouping variable."""

    log.info("=== Variant: %s (group_col=%s) ===", name, group_col)
    ct = pd.crosstab(df[group_col], df["syllable_type"])
    # Reindex columns to canonical Scattoni-7 order
    ct = ct.reindex(columns=SYLLABLE_ORDER, fill_value=0)
    proportions = ct.div(ct.sum(axis=1), axis=0)

    chi2, p, dof, _ = chi2_contingency(ct.values)
    n_total = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n_total * (min(ct.shape) - 1)))

    log.info("  Cohorts: %s", ct.index.tolist())
    log.info("  N=%d, chi2=%.1f, dof=%d, p=%.4g, Cramér's V=%.3f",
             n_total, chi2, dof, p, cramers_v)
    log.info("  Proportions:")
    for cohort, row in proportions.iterrows():
        log.info("    %s: %s",
                 cohort,
                 ", ".join(f"{c}={v:.1%}" for c, v in row.items()))

    # Save CSV (counts on top, proportions below)
    out_csv = out_dir / f"by_{name}.csv"
    with out_csv.open("w") as f:
        f.write(f"# Variant: {name} (group_col={group_col})\n")
        f.write(f"# N={n_total}, chi2={chi2:.1f}, dof={dof}, p={p:.4g}, Cramers_V={cramers_v:.3f}\n")
        f.write("# COUNTS\n")
        ct.to_csv(f)
        f.write("\n# PROPORTIONS\n")
        proportions.to_csv(f)
    log.info("  Saved %s", out_csv)

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(ct) * 0.6), 5))
    proportions.plot(kind="bar", stacked=True, ax=ax,
                     colormap="tab10", width=0.8)
    ax.set_ylabel("Proportion of syllables")
    ax.set_xlabel(group_col)
    ax.set_title(f"Scattoni-7 syllable proportions by {name}\n"
                 f"N={n_total}, χ²={chi2:.0f}, p={p:.2g}, V={cramers_v:.2f}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_png = out_dir / f"by_{name}.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_png)

    return {
        "name": name,
        "group_col": group_col,
        "n": int(n_total),
        "chi2": float(chi2),
        "dof": int(dof),
        "p": float(p),
        "cramers_v": float(cramers_v),
        "n_cohorts": int(len(ct)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", args.input)
    df = pd.read_csv(args.input)
    log.info("Loaded %d rows", len(df))

    # Defensive checks
    for col in ["tier", "couple", "couple_keep_set", "syllable_type"]:
        if col not in df.columns:
            log.error("Missing required column: %s", col)
            return 1

    results = []
    results.append(compare(df, "tier", "tier", args.output_dir))
    results.append(compare(df, "couple_keep_set", "couple_keep_set", args.output_dir))
    results.append(compare(df, "couple", "couple", args.output_dir))

    # Markdown summary
    summary_md = args.output_dir / "summary.md"
    with summary_md.open("w") as f:
        f.write("# Phase 2B repertoire comparisons (lab 131204)\n\n")
        f.write("Per-cohort Scattoni-7 syllable proportions with chi-square\n")
        f.write("tests of independence and Cramér's V effect sizes.\n\n")
        f.write("| Variant | Cohorts | N | χ² | dof | p | Cramér's V |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['n_cohorts']} | {r['n']:,} | "
                    f"{r['chi2']:.1f} | {r['dof']} | {r['p']:.4g} | "
                    f"{r['cramers_v']:.3f} |\n")
        f.write("\n## Interpretation guide\n\n")
        f.write("- **Cramér's V** is the standardized effect size: 0.1 = small, "
                "0.3 = moderate, 0.5 = large.\n")
        f.write("- **Large N inflates significance**: at 40k events, even tiny "
                "differences reach p<0.001. Trust V over p.\n")
        f.write("- **tier ≫ 0.1**: confirms post-labeling handoff finding "
                "that manual_review carries different syllable composition "
                "(noise leakage).\n")
        f.write("- **couple_keep_set ≫ 0.1**: confirms the 4 noise-prone "
                "couples have distinct repertoires (or noise patterns "
                "masquerading as distinct).\n")
    log.info("Saved %s", summary_md)
    log.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
