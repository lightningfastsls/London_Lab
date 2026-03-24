"""Cluster analysis and interpretation for USV subtypes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image
from scipy.spatial.distance import cdist


class ClusterAnalyzer:
    """Analyze and interpret USV clustering results.

    Provides tools for extracting exemplars, computing diversity metrics,
    and visualizing cluster characteristics.

    Args:
        embeddings_csv: Path to embeddings CSV
        clusters_csv: Path to cluster assignments CSV
    """

    def __init__(
        self,
        embeddings_csv: Union[str, Path],
        clusters_csv: Union[str, Path]
    ):
        self.embeddings_csv = Path(embeddings_csv)
        self.clusters_csv = Path(clusters_csv)

        # Load data
        print("Loading embeddings and cluster assignments...")
        self.embeddings_df = pd.read_csv(self.embeddings_csv)
        self.clusters_df = pd.read_csv(self.clusters_csv)

        # Merge on candidate_id
        self.df = pd.merge(
            self.clusters_df,
            self.embeddings_df,
            on='candidate_id',
            suffixes=('', '_emb')
        )

        print(f"Total samples: {len(self.df)}")
        print(f"Unique clusters: {self.df['cluster_label'].nunique()}")

        # Extract embedding matrix
        embedding_cols = [f'embedding_{i}' for i in range(128)]
        self.embeddings = self.df[embedding_cols].values

    def extract_exemplars(
        self,
        cluster_id: int,
        n: int = 5,
        method: str = 'nearest'
    ) -> List[str]:
        """Extract exemplar samples for a cluster.

        Args:
            cluster_id: Cluster label (-1 for noise in HDBSCAN)
            n: Number of exemplars to extract
            method: Selection method ('nearest' = nearest to centroid)

        Returns:
            List of candidate_ids for exemplars
        """
        # Filter to cluster
        cluster_mask = self.df['cluster_label'] == cluster_id
        cluster_df = self.df[cluster_mask].copy()
        cluster_embeddings = self.embeddings[cluster_mask]

        if len(cluster_df) == 0:
            return []

        if method == 'nearest':
            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0, keepdims=True)

            # Compute distances to centroid
            distances = cdist(cluster_embeddings, centroid, metric='euclidean').flatten()

            # Get indices of n nearest samples
            nearest_indices = np.argsort(distances)[:n]

            # Get candidate IDs
            exemplars = cluster_df.iloc[nearest_indices]['candidate_id'].tolist()

        else:
            raise ValueError(f"Unknown method: {method}")

        return exemplars

    def plot_exemplars(
        self,
        cluster_id: int,
        spectrograms_labeled_dir: Union[str, Path],
        spectrograms_auto_dir: Union[str, Path],
        output_path: Union[str, Path],
        n: int = 5,
        figsize: tuple = (15, 3)
    ):
        """Create grid of exemplar spectrograms for a cluster.

        Args:
            cluster_id: Cluster label
            spectrograms_labeled_dir: Directory with labeled USV spectrograms
            spectrograms_auto_dir: Directory with auto-detected spectrograms
            output_path: Path to save figure
            n: Number of exemplars to show
            figsize: Figure size (width, height)
        """
        # Extract exemplars
        exemplar_ids = self.extract_exemplars(cluster_id, n=n)

        if len(exemplar_ids) == 0:
            print(f"Warning: Cluster {cluster_id} is empty")
            return

        # Get spectrogram paths for exemplars
        exemplar_rows = self.df[self.df['candidate_id'].isin(exemplar_ids)]

        # Create figure
        fig, axes = plt.subplots(1, len(exemplar_ids), figsize=figsize)
        if len(exemplar_ids) == 1:
            axes = [axes]

        for ax, (idx, row) in zip(axes, exemplar_rows.iterrows()):
            # Determine which directory to use
            if row['data_source'] == 'labeled':
                spec_path = Path(spectrograms_labeled_dir) / f"{row['candidate_id']}.png"
            else:
                spec_path = Path(spectrograms_auto_dir) / f"{row['candidate_id']}.png"

            # Load spectrogram image
            if spec_path.exists():
                img = Image.open(spec_path)
                ax.imshow(img, aspect='auto')
                ax.axis('off')
                # Add title with candidate ID
                ax.set_title(f"{row['candidate_id'][:15]}...", fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
                ax.axis('off')

        # Overall title
        cluster_size = (self.df['cluster_label'] == cluster_id).sum()
        if cluster_id == -1:
            fig.suptitle(f"Noise/Outliers (n={cluster_size})", fontsize=12, fontweight='bold')
        else:
            fig.suptitle(f"Cluster {cluster_id} Exemplars (n={cluster_size})", fontsize=12, fontweight='bold')

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Exemplars plot saved: {output_path}")

    def compute_recording_diversity(self) -> pd.DataFrame:
        """Compute acoustic diversity metrics per recording.

        For each recording, computes:
        - Number of USVs
        - Number of unique clusters
        - Entropy (measure of cluster diversity)
        - Dominant cluster

        Returns:
            DataFrame with per-recording diversity metrics
        """
        print("\nComputing recording diversity...")

        # Group by source file
        recordings = []

        for source_file, group in self.df.groupby('source_file'):
            # Skip noise cluster for diversity calculation
            group_no_noise = group[group['cluster_label'] >= 0]

            if len(group_no_noise) == 0:
                continue

            # Cluster distribution
            cluster_counts = group_no_noise['cluster_label'].value_counts()
            n_clusters = len(cluster_counts)

            # Entropy (normalized)
            proportions = cluster_counts / cluster_counts.sum()
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            max_entropy = np.log2(n_clusters) if n_clusters > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Dominant cluster
            dominant_cluster = cluster_counts.idxmax()
            dominant_fraction = cluster_counts.max() / cluster_counts.sum()

            recordings.append({
                'source_file': source_file,
                'n_usvs': len(group),
                'n_usvs_no_noise': len(group_no_noise),
                'n_clusters': n_clusters,
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'dominant_cluster': dominant_cluster,
                'dominant_fraction': dominant_fraction
            })

        df_diversity = pd.DataFrame(recordings)
        df_diversity = df_diversity.sort_values('entropy', ascending=False)

        return df_diversity

    def generate_cluster_quality_report(
        self,
        output_path: Union[str, Path]
    ):
        """Generate comprehensive cluster quality report.

        Args:
            output_path: Path to save report text file
        """
        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("USV Clustering Quality Report\n")
            f.write("=" * 70 + "\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"  Total samples: {len(self.df)}\n")
            f.write(f"  Data sources:\n")
            for source, count in self.df['data_source'].value_counts().items():
                pct = 100 * count / len(self.df)
                f.write(f"    {source}: {count} ({pct:.1f}%)\n")
            f.write("\n")

            # Cluster summary
            f.write("Cluster Summary:\n")
            for cluster_id in sorted(self.df['cluster_label'].unique()):
                cluster_df = self.df[self.df['cluster_label'] == cluster_id]
                count = len(cluster_df)
                pct = 100 * count / len(self.df)

                if cluster_id == -1:
                    f.write(f"  Noise/Outliers: {count} ({pct:.1f}%)\n")
                else:
                    f.write(f"  Cluster {cluster_id}: {count} ({pct:.1f}%)\n")

                    # Data source breakdown
                    for source in ['labeled', 'auto_detected']:
                        source_count = (cluster_df['data_source'] == source).sum()
                        source_pct = 100 * source_count / count if count > 0 else 0
                        f.write(f"    {source}: {source_count} ({source_pct:.1f}%)\n")
            f.write("\n")

            # Tier 2 QC checklist
            f.write("Tier 2 QC Validation (Manual Review):\n")
            f.write("-" * 70 + "\n")
            f.write("Review exemplar images for each cluster and mark:\n")
            f.write("  [ ] Valid cluster - Exemplars show consistent acoustic pattern\n")
            f.write("  [ ] Noise cluster - Exemplars are inconsistent or artifacts\n")
            f.write("  [ ] Mixed cluster - Some exemplars good, some bad\n")
            f.write("\n")

            for cluster_id in sorted(self.df['cluster_label'].unique()):
                if cluster_id >= 0:  # Skip noise
                    f.write(f"Cluster {cluster_id}: [ ] Valid  [ ] Noise  [ ] Mixed\n")
                    f.write(f"  Notes: _______________________________________________\n")
                    f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("\n")

            # Recording diversity top 10
            diversity_df = self.compute_recording_diversity()
            f.write("Top 10 Most Diverse Recordings (by entropy):\n")
            for idx, row in diversity_df.head(10).iterrows():
                f.write(f"  {row['source_file']}: ")
                f.write(f"{row['n_clusters']} clusters, ")
                f.write(f"entropy={row['normalized_entropy']:.2f}\n")

        print(f"Quality report saved to: {output_path}")
