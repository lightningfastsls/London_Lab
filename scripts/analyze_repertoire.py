"""CLI entry point for syllable repertoire statistical analysis.

Compares syllable repertoires between mouse populations (e.g. wild vs lab)
using classified detection data from DeepSqueak (Phase 14.2 output).

Optionally joins a metadata CSV to attach population/animal_id columns
to the classified data before analysis.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.classification.repertoire_stats import (
    RepertoireConfig,
    analyze_repertoire,
    syllable_diversity,
    syllable_proportions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare syllable repertoires between mouse populations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analysis (data already has population + animal_id columns)
  python analyze_repertoire.py \\
      --classified-data classified_detections.csv \\
      --output-dir analysis/repertoire

  # Join metadata first (e.g. to add population labels)
  python analyze_repertoire.py \\
      --classified-data classified_detections.csv \\
      --metadata animal_metadata.csv \\
      --output-dir analysis/repertoire

  # Custom column names and fewer permutations (faster)
  python analyze_repertoire.py \\
      --classified-data classified_detections.csv \\
      --population-column group \\
      --syllable-column type \\
      --n-permutations 500 \\
      --output-dir analysis/repertoire

  # Dry run — show data summary without running analysis
  python analyze_repertoire.py \\
      --classified-data classified_detections.csv \\
      --dry-run -v
        """,
    )
    parser.add_argument(
        "--classified-data",
        required=True,
        help="CSV with classified detection data (output from Phase 14.2).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata CSV to join. Must share a key column with "
        "classified data (typically wav_stem or animal_id). Adds population "
        "and animal_id columns to the classified data before analysis.",
    )
    parser.add_argument(
        "--metadata-key",
        default="wav_stem",
        help="Column to join metadata on. Default: wav_stem",
    )
    parser.add_argument(
        "--population-column",
        default="population",
        help="Column with population labels. Default: population",
    )
    parser.add_argument(
        "--animal-id-column",
        default="animal_id",
        help="Column identifying individual animals. Default: animal_id",
    )
    parser.add_argument(
        "--syllable-column",
        default="label",
        help="Column with syllable type labels. Default: label",
    )
    parser.add_argument(
        "--time-column",
        default="begin_time_s",
        help="Column with detection times (seconds). Default: begin_time_s",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for statistical tests. Default: 1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "analysis" / "repertoire"),
        help="Output directory for results. Default: analysis/repertoire/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show summary without running analysis.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Configure logging.
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    import pandas as pd

    # -- Load classified data --
    classified_path = Path(args.classified_data)
    if not classified_path.is_file():
        print(
            f"Error: Classified data file not found: {classified_path}",
            file=sys.stderr,
        )
        return 1

    try:
        df = pd.read_csv(classified_path)
    except Exception as exc:
        print(f"Error reading classified data: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} rows from {classified_path}")

    # -- Optional metadata join --
    if args.metadata is not None:
        metadata_path = Path(args.metadata)
        if not metadata_path.is_file():
            print(
                f"Error: Metadata file not found: {metadata_path}",
                file=sys.stderr,
            )
            return 1

        try:
            meta_df = pd.read_csv(metadata_path)
        except Exception as exc:
            print(f"Error reading metadata: {exc}", file=sys.stderr)
            return 1

        key = args.metadata_key
        if key not in df.columns:
            print(
                f"Error: Join key '{key}' not found in classified data. "
                f"Available: {list(df.columns)}",
                file=sys.stderr,
            )
            return 1
        if key not in meta_df.columns:
            print(
                f"Error: Join key '{key}' not found in metadata. "
                f"Available: {list(meta_df.columns)}",
                file=sys.stderr,
            )
            return 1

        before_len = len(df)
        df = df.merge(meta_df, on=key, how="left")
        print(f"Joined metadata: {before_len} rows -> {len(df)} rows")

    # -- Validate required columns --
    required = [
        args.population_column,
        args.animal_id_column,
        args.syllable_column,
        args.time_column,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(
            f"Error: Missing required columns: {missing}. "
            f"Available: {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    # -- Dry run: show data summary --
    if args.dry_run:
        populations = df[args.population_column].unique()
        animals = df[args.animal_id_column].unique()
        syllables = df[args.syllable_column].unique()

        print(f"\nDry run — data summary:")
        print(f"  Populations: {list(populations)} ({len(populations)})")
        print(f"  Animals: {len(animals)}")
        print(f"  Syllable types: {len(syllables)}")
        print(f"  Total calls: {len(df)}")
        print()

        for pop in sorted(populations):
            pop_df = df[df[args.population_column] == pop]
            n_animals = pop_df[args.animal_id_column].nunique()
            n_calls = len(pop_df)
            print(f"  {pop}: {n_animals} animals, {n_calls} calls")

        print(f"\n  Output would go to: {args.output_dir}")
        return 0

    # -- Run full analysis --
    try:
        config = RepertoireConfig(
            classified_data_path=classified_path,
            population_column=args.population_column,
            animal_id_column=args.animal_id_column,
            syllable_column=args.syllable_column,
            time_column=args.time_column,
            n_permutations=args.n_permutations,
            random_seed=args.seed,
            output_dir=Path(args.output_dir),
        )
    except ValueError as exc:
        print(f"Error: Invalid configuration: {exc}", file=sys.stderr)
        return 1

    try:
        results = analyze_repertoire(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # -- Summary --
    print(f"\nAnalysis complete:")
    for method in ("permanova", "jsd", "chi_square"):
        if method in results:
            r = results[method]
            print(f"  {method}: statistic={r['statistic']:.4f}, p={r['p_value']:.4f}")
    if "transition_comparison" in results:
        tc = results["transition_comparison"]
        print(f"  transition: ||F||={tc['frobenius_norm']:.4f}, p={tc['p_value']:.4f}")
    print(f"  Output: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
