"""Analyze USV clustering results and extract cluster exemplars.

Generates exemplar grids, computes diversity metrics, and creates
quality report for Tier 2 manual validation.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from usv_spectrogram.clustering.analyzer import ClusterAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze USV clustering results"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("analysis/clustering/embeddings_all.csv"),
        help="Path to embeddings CSV (default: analysis/clustering/embeddings_all.csv)"
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("analysis/clustering/hdbscan/cluster_assignments.csv"),
        help="Path to cluster assignments CSV (default: analysis/clustering/hdbscan/cluster_assignments.csv)"
    )
    parser.add_argument(
        "--spectrograms-labeled",
        type=Path,
        default=Path("spectrograms_training"),
        help="Directory with labeled USV spectrograms (default: spectrograms_training/)"
    )
    parser.add_argument(
        "--spectrograms-auto",
        type=Path,
        default=Path("analysis/clustering/auto_detected/spectrograms"),
        help="Directory with auto-detected spectrograms (default: analysis/clustering/auto_detected/spectrograms/)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as cluster assignments directory)"
    )
    parser.add_argument(
        "--n-exemplars",
        type=int,
        default=5,
        help="Number of exemplars per cluster (default: 5)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.embeddings.exists():
        print(f"Error: Embeddings file not found: {args.embeddings}")
        sys.exit(1)

    if not args.clusters.exists():
        print(f"Error: Cluster assignments file not found: {args.clusters}")
        sys.exit(1)

    if not args.spectrograms_labeled.exists():
        print(f"Error: Labeled spectrograms directory not found: {args.spectrograms_labeled}")
        sys.exit(1)

    # Set output directory (default: same as cluster assignments)
    if args.output_dir is None:
        args.output_dir = args.clusters.parent

    print("=" * 70)
    print("USV Clustering Analysis")
    print("=" * 70)
    print(f"Embeddings: {args.embeddings}")
    print(f"Cluster assignments: {args.clusters}")
    print(f"Labeled spectrograms: {args.spectrograms_labeled}")
    print(f"Auto-detected spectrograms: {args.spectrograms_auto}")
    print(f"Output directory: {args.output_dir}")
    print(f"Exemplars per cluster: {args.n_exemplars}")
    print("=" * 70)

    # Initialize analyzer
    analyzer = ClusterAnalyzer(
        embeddings_csv=args.embeddings,
        clusters_csv=args.clusters
    )

    # Get unique cluster labels
    cluster_labels = sorted(analyzer.df['cluster_label'].unique())
    print(f"\nFound {len(cluster_labels)} clusters (including noise)")

    # Generate exemplar plots for each cluster
    print("\nGenerating exemplar plots...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id in cluster_labels:
        if cluster_id == -1:
            output_path = args.output_dir / "cluster_noise.png"
        else:
            output_path = args.output_dir / f"exemplars_cluster_{cluster_id}.png"

        analyzer.plot_exemplars(
            cluster_id=cluster_id,
            spectrograms_labeled_dir=args.spectrograms_labeled,
            spectrograms_auto_dir=args.spectrograms_auto,
            output_path=output_path,
            n=args.n_exemplars
        )

    # Compute recording diversity
    print("\nComputing recording diversity...")
    diversity_df = analyzer.compute_recording_diversity()

    # Save diversity CSV
    diversity_path = args.output_dir / "recording_diversity.csv"
    diversity_df.to_csv(diversity_path, index=False)
    print(f"Recording diversity saved to: {diversity_path}")

    # Print top 10 most diverse
    print("\nTop 10 Most Diverse Recordings:")
    for idx, row in diversity_df.head(10).iterrows():
        print(f"  {row['source_file']}: {row['n_clusters']} clusters, entropy={row['normalized_entropy']:.2f}")

    # Generate quality report
    print("\nGenerating cluster quality report...")
    report_path = args.output_dir / "cluster_quality_report.txt"
    analyzer.generate_cluster_quality_report(output_path=report_path)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. Review exemplar images in: {args.output_dir}")
    print(f"2. Open quality report: {report_path}")
    print(f"3. Perform Tier 2 QC (~5 min manual review)")
    print(f"4. Mark valid clusters in the quality report")
    print("=" * 70)


if __name__ == "__main__":
    main()
