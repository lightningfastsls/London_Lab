"""Dimensionality reduction and visualization for USV embeddings."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EmbeddingVisualizer:
    """Visualize high-dimensional embeddings using dimensionality reduction.

    Supports t-SNE and UMAP for reducing 128D embeddings to 2D for visualization.

    Args:
        method: Dimensionality reduction method ('tsne' or 'umap')
        random_state: Random seed for reproducibility (default: 42)
        **kwargs: Additional parameters for the reduction algorithm
    """

    def __init__(
        self,
        method: str = 'tsne',
        random_state: int = 42,
        **kwargs
    ):
        self.method = method.lower()
        self.random_state = random_state
        self.kwargs = kwargs
        self.reducer = None

        # Validate method
        if self.method == 'umap' and not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP not available. Install with: pip install umap-learn"
            )

        if self.method not in ['tsne', 'umap']:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embeddings from 128D to 2D.

        Args:
            embeddings: Array of shape (N, 128)

        Returns:
            Reduced coordinates of shape (N, 2)
        """
        print(f"\nRunning {self.method.upper()} dimensionality reduction...")
        print(f"Input shape: {embeddings.shape}")

        if self.method == 'tsne':
            # Default t-SNE parameters
            params = {
                'n_components': 2,
                'perplexity': 30,
                'n_iter': 1000,
                'random_state': self.random_state,
                'verbose': 1
            }
            params.update(self.kwargs)

            self.reducer = TSNE(**params)
            coords_2d = self.reducer.fit_transform(embeddings)

        elif self.method == 'umap':
            # Default UMAP parameters
            params = {
                'n_components': 2,
                'n_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'euclidean',
                'random_state': self.random_state,
                'verbose': True
            }
            params.update(self.kwargs)

            self.reducer = umap.UMAP(**params)
            coords_2d = self.reducer.fit_transform(embeddings)

        print(f"Output shape: {coords_2d.shape}")
        return coords_2d

    def plot_scatter(
        self,
        coords_2d: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = None,
        output_path: Union[str, Path] = None,
        figsize: tuple = (10, 8),
        alpha: float = 0.6,
        s: int = 20
    ):
        """Create 2D scatter plot of reduced embeddings.

        Args:
            coords_2d: 2D coordinates (N, 2)
            labels: Optional labels for coloring points (N,)
            title: Plot title
            output_path: Path to save figure (optional)
            figsize: Figure size (width, height)
            alpha: Point transparency
            s: Point size
        """
        fig, ax = plt.subplots(figsize=figsize)

        if labels is not None:
            # Get unique labels
            unique_labels = np.unique(labels)
            n_labels = len(unique_labels)

            # Use tab10 colormap for up to 10 categories
            if n_labels <= 10:
                cmap = plt.cm.tab10
            else:
                cmap = plt.cm.tab20

            # Plot each label with different color
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    c=[cmap(i)],
                    label=str(label),
                    alpha=alpha,
                    s=s,
                    edgecolors='none'
                )

            ax.legend(loc='best', framealpha=0.9)

        else:
            # Single color if no labels
            ax.scatter(
                coords_2d[:, 0],
                coords_2d[:, 1],
                c='steelblue',
                alpha=alpha,
                s=s,
                edgecolors='none'
            )

        ax.set_xlabel(f'{self.method.upper()} 1', fontsize=12)
        ax.set_ylabel(f'{self.method.upper()} 2', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{self.method.upper()} Visualization', fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")

        plt.close()

    def plot_from_embeddings_csv(
        self,
        embeddings_csv: Union[str, Path],
        color_by: str = 'data_source',
        output_path: Union[str, Path] = None
    ):
        """Load embeddings from CSV and create visualization.

        Args:
            embeddings_csv: Path to embeddings CSV with embedding_0...embedding_127 columns
            color_by: Column name to use for coloring points (e.g., 'data_source', 'cluster')
            output_path: Path to save figure
        """
        # Load embeddings
        df = pd.read_csv(embeddings_csv)
        print(f"Loaded {len(df)} samples from {embeddings_csv}")

        # Extract embedding columns
        embedding_cols = [f'embedding_{i}' for i in range(128)]
        embeddings = df[embedding_cols].values

        # Extract labels for coloring
        if color_by in df.columns:
            labels = df[color_by].values
        else:
            print(f"Warning: Column '{color_by}' not found. Using no coloring.")
            labels = None

        # Reduce dimensionality
        coords_2d = self.fit_transform(embeddings)

        # Plot
        title = f'{self.method.upper()} Visualization (colored by {color_by})'
        self.plot_scatter(
            coords_2d=coords_2d,
            labels=labels,
            title=title,
            output_path=output_path
        )

        return coords_2d, labels
