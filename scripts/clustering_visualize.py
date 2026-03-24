"""Visualize USV embeddings using dimensionality reduction.

Generates t-SNE and/or UMAP 2D visualizations of 128D CNN embeddings
to explore acoustic structure before clustering.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from usv_spectrogram.clustering.visualizer import EmbeddingVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Visualize USV embeddings using dimensionality reduction"
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
        nargs='+',
        default=['tsne', 'umap'],
        choices=['tsne', 'umap'],
        help="Dimensionality reduction method(s) (default: tsne umap)"
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default='data_source',
        help="Column to use for coloring points (default: data_source)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/clustering"),
        help="Output directory for plots (default: analysis/clustering/)"
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter (default: 30)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="t-SNE number of iterations (default: 1000)"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.embeddings.exists():
        print(f"Error: Embeddings file not found: {args.embeddings}")
        print("Run clustering_extract_features.py first to generate embeddings.")
        sys.exit(1)

    print("=" * 70)
    print("USV Embedding Visualization")
    print("=" * 70)
    print(f"Embeddings: {args.embeddings}")
    print(f"Methods: {', '.join(args.method)}")
    print(f"Color by: {args.color_by}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run each method
    for method in args.method:
        print(f"\n{'='*70}")
        print(f"Running {method.upper()}")
        print(f"{'='*70}")

        # Set method-specific parameters
        if method == 'tsne':
            kwargs = {
                'perplexity': args.perplexity,
                'n_iter': args.n_iter
            }
        elif method == 'umap':
            kwargs = {
                'n_neighbors': args.n_neighbors,
                'min_dist': args.min_dist
            }
        else:
            kwargs = {}

        # Initialize visualizer
        visualizer = EmbeddingVisualizer(
            method=method,
            random_state=42,
            **kwargs
        )

        # Generate plot
        output_path = args.output_dir / f"{method}_plot.png"
        coords_2d, labels = visualizer.plot_from_embeddings_csv(
            embeddings_csv=args.embeddings,
            color_by=args.color_by,
            output_path=output_path
        )

        print(f"\n{method.upper()} complete!")
        print(f"Plot saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
