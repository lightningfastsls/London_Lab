"""Clustering algorithms for USV embeddings."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class USVClusterer:
    """Clustering for USV embeddings.

    Supports K-means (fixed number of clusters) and HDBSCAN (automatic detection
    with outlier identification).

    Args:
        method: Clustering method ('kmeans' or 'hdbscan')
        **kwargs: Additional parameters for the clustering algorithm
    """

    def __init__(self, method: str = 'kmeans', **kwargs):
        self.method = method.lower()
        self.kwargs = kwargs
        self.clusterer = None
        self.labels_ = None

        # Validate method
        if self.method == 'hdbscan' and not HDBSCAN_AVAILABLE:
            raise ImportError(
                "HDBSCAN not available. Install with: pip install hdbscan"
            )

        if self.method not in ['kmeans', 'hdbscan']:
            raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'hdbscan'.")

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit clustering algorithm and predict labels.

        Args:
            embeddings: Array of shape (N, 128)

        Returns:
            Cluster labels of shape (N,)
            For HDBSCAN: -1 indicates noise/outliers
        """
        print(f"\nRunning {self.method.upper()} clustering...")
        print(f"Input shape: {embeddings.shape}")

        if self.method == 'kmeans':
            # Default K-means parameters
            params = {
                'n_clusters': 5,
                'n_init': 50,
                'random_state': 42,
                'verbose': 0
            }
            params.update(self.kwargs)

            print(f"Parameters: {params}")
            self.clusterer = KMeans(**params)
            self.labels_ = self.clusterer.fit_predict(embeddings)

        elif self.method == 'hdbscan':
            # Default HDBSCAN parameters
            params = {
                'min_cluster_size': 50,
                'min_samples': 5,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'
            }
            params.update(self.kwargs)

            print(f"Parameters: {params}")
            self.clusterer = hdbscan.HDBSCAN(**params)
            self.labels_ = self.clusterer.fit_predict(embeddings)

        # Summary
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        n_noise = np.sum(self.labels_ == -1)

        print(f"\nClustering complete!")
        print(f"Number of clusters: {n_clusters}")
        if n_noise > 0:
            print(f"Noise/outliers: {n_noise} ({100*n_noise/len(self.labels_):.1f}%)")

        # Cluster sizes
        print("\nCluster sizes:")
        for label in sorted(unique_labels):
            count = np.sum(self.labels_ == label)
            pct = 100 * count / len(self.labels_)
            if label == -1:
                print(f"  Noise: {count} ({pct:.1f}%)")
            else:
                print(f"  Cluster {label}: {count} ({pct:.1f}%)")

        return self.labels_

    def compute_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering quality metrics.

        Args:
            embeddings: Array of shape (N, 128)
            labels: Cluster labels of shape (N,)

        Returns:
            Dictionary with metric names and values
        """
        print("\nComputing clustering metrics...")

        metrics = {}

        # Filter out noise points for metrics (HDBSCAN assigns -1 to noise)
        non_noise_mask = labels >= 0
        embeddings_clean = embeddings[non_noise_mask]
        labels_clean = labels[non_noise_mask]

        # Need at least 2 clusters and 2 samples per cluster
        n_clusters = len(np.unique(labels_clean))
        if n_clusters < 2:
            print("Warning: Need at least 2 clusters to compute metrics")
            return metrics

        # Silhouette score (range: -1 to 1, higher is better)
        # Measures how similar samples are to their own cluster vs other clusters
        silhouette = silhouette_score(embeddings_clean, labels_clean)
        metrics['silhouette_score'] = silhouette
        print(f"Silhouette score: {silhouette:.3f} ", end="")
        if silhouette > 0.5:
            print("(Excellent)")
        elif silhouette > 0.3:
            print("(Acceptable)")
        else:
            print("(Weak structure)")

        # Calinski-Harabasz score (higher is better)
        # Ratio of between-cluster to within-cluster variance
        ch_score = calinski_harabasz_score(embeddings_clean, labels_clean)
        metrics['calinski_harabasz_score'] = ch_score
        print(f"Calinski-Harabasz score: {ch_score:.1f}")

        return metrics

    def save_assignments(
        self,
        embeddings_csv: Union[str, Path],
        output_path: Union[str, Path]
    ):
        """Save cluster assignments to CSV.

        Args:
            embeddings_csv: Path to embeddings CSV (to get candidate_id)
            output_path: Path to save cluster assignments CSV
        """
        # Load embeddings CSV to get metadata
        df = pd.read_csv(embeddings_csv)

        # Add cluster labels
        df['cluster_label'] = self.labels_

        # Save only metadata + cluster label
        metadata_cols = ['candidate_id', 'source_file', 'data_source']
        if 'label' in df.columns:
            metadata_cols.append('label')

        output_df = df[metadata_cols + ['cluster_label']]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"\nCluster assignments saved to: {output_path}")

        return output_df
