"""Cluster USV embeddings to discover acoustic subtypes.

Applies K-means or HDBSCAN clustering to 128D CNN embeddings to identify
natural groupings in USV vocalizations.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from usv_spectrogram.clustering.clusterer import USVClusterer


def main():
    parser = argparse.ArgumentParser(
        description="Cluster USV embeddings to discover acoustic subtypes"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("analysis/clustering/embeddings_all.csv"),
        help="Path to embeddings CSV (default: analysis/clustering/embeddings_all.csv)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default='hdbscan',
        choices=['kmeans', 'hdbscan'],
        help="Clustering method (default: hdbscan)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/clustering"),
        help="Output directory (default: analysis/clustering/)"
    )
    # K-means parameters
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters for K-means (default: 5)"
    )
    # HDBSCAN parameters
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=50,
        help="HDBSCAN minimum cluster size (default: 50)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="HDBSCAN minimum samples (default: 5)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.embeddings.exists():
        print(f"Error: Embeddings file not found: {args.embeddings}")
        print("Run clustering_extract_features.py first to generate embeddings.")
        sys.exit(1)

    print("=" * 70)
    print("USV Clustering Analysis")
    print("=" * 70)
    print(f"Embeddings: {args.embeddings}")
    print(f"Method: {args.method}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Load embeddings
    df = pd.read_csv(args.embeddings)
    print(f"\nLoaded {len(df)} samples")

    # Extract embedding columns
    embedding_cols = [f'embedding_{i}' for i in range(128)]
    embeddings = df[embedding_cols].values
    print(f"Embedding shape: {embeddings.shape}")

    # Set method-specific parameters
    if args.method == 'kmeans':
        kwargs = {
            'n_clusters': args.k,
            'n_init': 50,
            'random_state': 42
        }
        output_subdir = args.output_dir / f"kmeans_k{args.k}"
    elif args.method == 'hdbscan':
        kwargs = {
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples
        }
        output_subdir = args.output_dir / "hdbscan"
    else:
        kwargs = {}
        output_subdir = args.output_dir

    output_subdir.mkdir(parents=True, exist_ok=True)

    # Initialize clusterer
    clusterer = USVClusterer(method=args.method, **kwargs)

    # Fit and predict
    labels = clusterer.fit_predict(embeddings)

    # Compute quality metrics
    metrics = clusterer.compute_metrics(embeddings, labels)

    # Save cluster assignments
    assignments_path = output_subdir / "cluster_assignments.csv"
    clusterer.save_assignments(
        embeddings_csv=args.embeddings,
        output_path=assignments_path
    )

    # Save metrics to text file
    metrics_path = output_subdir / "cluster_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Clustering Method: {args.method.upper()}\n")
        f.write(f"Parameters: {kwargs}\n")
        f.write(f"\n")
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"\n")

        # Cluster sizes
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)

        f.write(f"Number of clusters: {n_clusters}\n")
        if n_noise > 0:
            f.write(f"Noise/outliers: {n_noise} ({100*n_noise/len(labels):.1f}%)\n")
        f.write(f"\n")

        f.write("Cluster sizes:\n")
        for label in sorted(unique_labels):
            count = np.sum(labels == label)
            pct = 100 * count / len(labels)
            if label == -1:
                f.write(f"  Noise: {count} ({pct:.1f}%)\n")
            else:
                f.write(f"  Cluster {label}: {count} ({pct:.1f}%)\n")
        f.write(f"\n")

        # Quality metrics
        f.write("Quality Metrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.3f}\n")

        # Interpretation
        f.write(f"\n")
        f.write("Interpretation:\n")
        if 'silhouette_score' in metrics:
            sil = metrics['silhouette_score']
            if sil > 0.5:
                f.write(f"  Silhouette score ({sil:.3f}) indicates EXCELLENT cluster structure\n")
            elif sil > 0.3:
                f.write(f"  Silhouette score ({sil:.3f}) indicates ACCEPTABLE cluster structure\n")
            else:
                f.write(f"  Silhouette score ({sil:.3f}) indicates WEAK cluster structure\n")
                f.write(f"  Consider trying different parameters or methods\n")

    print(f"\nMetrics saved to: {metrics_path}")

    print("\n" + "=" * 70)
    print("Clustering complete!")
    print(f"Results saved to: {output_subdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
