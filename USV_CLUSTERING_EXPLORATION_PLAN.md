# USV Clustering Exploration - Implementation Plan

## Overview

Before building a supervised USV type classifier, we'll explore the natural structure in the USV data using unsupervised methods. This will reveal:

1. How many natural USV types exist in the data
2. Whether wild and lab mice (or different recording sources) produce different call types
3. Whether predefined categories (flat, sweep, chevron) match the natural structure

**Approach:** Use the trained CNN as a feature extractor, then cluster and visualize the resulting feature space.

---

## Project Structure

```
src/usv_spectrogram/
├── clustering/
│   ├── __init__.py
│   ├── feature_extractor.py    # Extract CNN features from USVs
│   ├── visualize.py            # t-SNE/UMAP visualization
│   ├── cluster.py              # Clustering algorithms
│   └── analysis.py             # Compare clusters across populations
scripts/
├── extract_features.py         # CLI: extract features from all USVs
├── visualize_clusters.py       # CLI: generate visualizations
├── analyze_clusters.py         # CLI: statistical analysis
└── explore_clusters.py         # CLI: interactive cluster exploration
analysis/
└── clustering/
    ├── features.npy            # Extracted feature vectors
    ├── metadata.csv            # USV metadata (source, times, etc.)
    ├── tsne_plot.png           # Visualization
    ├── cluster_assignments.csv # Which cluster each USV belongs to
    └── cluster_exemplars/      # Example spectrograms from each cluster
```

---

## Phase 1: Feature Extraction

### Goal
Extract feature vectors from all detected USVs using the trained CNN as a feature extractor.

### Implementation

**File:** `src/usv_spectrogram/clustering/feature_extractor.py`

```python
"""
Extract feature vectors from USV spectrograms using trained CNN.

The CNN learned useful representations for USV detection.
We use these representations (before the classification head) as features for clustering.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN


class FeatureExtractor:
    """
    Extract features from spectrograms using trained CNN.
    
    Removes the classification head and uses the output of the 
    global average pooling layer as the feature vector.
    """
    
    def __init__(self, model_path: Path, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load full model
        self.model = USVClassifierCNN()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create feature extractor (remove classification head)
        # This depends on your CNN architecture - adjust as needed
        self.feature_extractor = self._create_feature_extractor()
    
    def _create_feature_extractor(self) -> nn.Module:
        """
        Create a model that outputs features instead of class probabilities.
        
        For USVClassifierCNN, we want output after global_pool, before classifier.
        """
        class FeatureModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.conv_blocks = original_model.conv_blocks
                self.global_pool = original_model.global_pool
            
            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)  # Flatten
                return x
        
        return FeatureModel(self.model).to(self.device)
    
    def extract(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a single spectrogram.
        
        Args:
            spectrogram: 2D numpy array (frequency x time), normalized
        
        Returns:
            1D numpy array of features (e.g., 128 dimensions)
        """
        with torch.no_grad():
            # Prepare input
            x = torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(0)
            x = x.to(self.device)
            
            # Extract features
            features = self.feature_extractor(x)
            
            return features.cpu().numpy().squeeze()
    
    def extract_batch(self, spectrograms: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple spectrograms efficiently.
        
        Args:
            spectrograms: List of 2D numpy arrays
            batch_size: Batch size for inference
        
        Returns:
            2D numpy array of shape (num_spectrograms, num_features)
        """
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(spectrograms), batch_size), desc="Extracting features"):
                batch = spectrograms[i:i + batch_size]
                
                # Stack into tensor
                x = torch.stack([
                    torch.from_numpy(s).float().unsqueeze(0) 
                    for s in batch
                ])
                x = x.to(self.device)
                
                # Extract features
                features = self.feature_extractor(x)
                all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def get_feature_dim(self) -> int:
        """Return the dimensionality of feature vectors."""
        # Run a dummy input to get output size
        dummy = torch.zeros(1, 1, 128, 128).to(self.device)
        with torch.no_grad():
            out = self.feature_extractor(dummy)
        return out.shape[1]


def extract_features_from_dataset(
    model_path: Path,
    spectrogram_dir: Path,
    metadata_csv: Path,
    output_dir: Path
):
    """
    Extract features from all USV spectrograms in a dataset.
    
    Args:
        model_path: Path to trained CNN model
        spectrogram_dir: Directory containing spectrogram images
        metadata_csv: CSV with spectrogram paths and metadata
        output_dir: Where to save features and metadata
    
    Saves:
        - features.npy: (N, feature_dim) array of features
        - metadata.csv: Copy of input metadata with feature indices
    """
    import pandas as pd
    from PIL import Image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(metadata_csv)
    
    # Filter to only USVs (not noise)
    df_usv = df[df['label'] == 'usv'].copy()
    print(f"Found {len(df_usv)} USV samples")
    
    # Load spectrograms
    print("Loading spectrograms...")
    spectrograms = []
    valid_indices = []
    
    for idx, row in tqdm(df_usv.iterrows(), total=len(df_usv)):
        spec_path = spectrogram_dir / row['spectrogram_path']
        if spec_path.exists():
            img = np.array(Image.open(spec_path).convert('L')).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-8)  # Normalize
            spectrograms.append(img)
            valid_indices.append(idx)
        else:
            print(f"Warning: {spec_path} not found")
    
    df_usv = df_usv.loc[valid_indices].reset_index(drop=True)
    print(f"Loaded {len(spectrograms)} spectrograms")
    
    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor(model_path)
    features = extractor.extract_batch(spectrograms)
    
    print(f"Feature shape: {features.shape}")
    
    # Save
    np.save(output_dir / 'features.npy', features)
    df_usv.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f"Saved features to {output_dir / 'features.npy'}")
    print(f"Saved metadata to {output_dir / 'metadata.csv'}")
    
    return features, df_usv
```

**Script:** `scripts/extract_features.py`

```python
"""
CLI script to extract features from USV dataset.

Usage:
    python scripts/extract_features.py \
        --model models/production/best_model.pt \
        --spectrograms data/candidates/spectrograms \
        --metadata splits/all_labeled.csv \
        --output analysis/clustering
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.clustering.feature_extractor import extract_features_from_dataset


def main():
    parser = argparse.ArgumentParser(description='Extract CNN features from USVs')
    parser.add_argument('--model', type=Path, required=True, help='Path to trained model')
    parser.add_argument('--spectrograms', type=Path, required=True, help='Spectrogram directory')
    parser.add_argument('--metadata', type=Path, required=True, help='Metadata CSV')
    parser.add_argument('--output', type=Path, default=Path('analysis/clustering'))
    
    args = parser.parse_args()
    
    extract_features_from_dataset(
        args.model,
        args.spectrograms,
        args.metadata,
        args.output
    )


if __name__ == '__main__':
    main()
```

---

## Phase 2: Visualization

### Goal
Visualize the feature space using dimensionality reduction (t-SNE and UMAP) to see if natural clusters exist.

### Implementation

**File:** `src/usv_spectrogram/clustering/visualize.py`

```python
"""
Visualize USV feature space using dimensionality reduction.

Creates 2D plots showing:
- Natural clustering structure
- Separation by recording source (wild vs lab, or by recording ID)
- Distribution of USV characteristics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def run_tsne(
    features: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """
    Run t-SNE dimensionality reduction.
    
    Args:
        features: (N, D) array of feature vectors
        perplexity: t-SNE perplexity parameter (typically 5-50)
        n_iter: Number of iterations
        random_state: Random seed for reproducibility
    
    Returns:
        (N, 2) array of 2D coordinates
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    
    coords = tsne.fit_transform(features_scaled)
    
    return coords


def run_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction.
    
    UMAP often preserves global structure better than t-SNE.
    
    Args:
        features: (N, D) array of feature vectors
        n_neighbors: Number of neighbors (larger = more global structure)
        min_dist: Minimum distance between points (smaller = tighter clusters)
        random_state: Random seed
    
    Returns:
        (N, 2) array of 2D coordinates
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Run UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    
    coords = reducer.fit_transform(features_scaled)
    
    return coords


def plot_embedding(
    coords: np.ndarray,
    labels: np.ndarray = None,
    label_names: dict = None,
    title: str = "USV Feature Space",
    save_path: Path = None,
    figsize: tuple = (12, 10)
):
    """
    Plot 2D embedding with optional color labels.
    
    Args:
        coords: (N, 2) array of 2D coordinates
        labels: Optional (N,) array of integer labels for coloring
        label_names: Optional dict mapping label int to name
        title: Plot title
        save_path: Path to save figure (None = display)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=10)
    else:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names.get(label, str(label)) if label_names else str(label)
            ax.scatter(
                coords[mask, 0], 
                coords[mask, 1], 
                c=[colors[i]], 
                label=name,
                alpha=0.6, 
                s=15
            )
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embedding_by_recording(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    recording_column: str = 'source_file',
    save_path: Path = None
):
    """
    Plot embedding colored by recording source.
    
    This reveals whether different recordings occupy different regions,
    which would indicate recording-specific characteristics or
    different populations (wild vs lab).
    """
    # Create numeric labels from recording names
    recordings = metadata[recording_column].unique()
    recording_to_int = {r: i for i, r in enumerate(recordings)}
    labels = metadata[recording_column].map(recording_to_int).values
    
    # Create label names (shortened)
    label_names = {i: r.split('/')[-1][:20] for r, i in recording_to_int.items()}
    
    plot_embedding(
        coords,
        labels=labels,
        label_names=label_names,
        title="USV Feature Space by Recording",
        save_path=save_path
    )


def create_visualization_report(
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path
):
    """
    Create comprehensive visualization report.
    
    Generates:
    - t-SNE plot (no labels)
    - t-SNE plot by recording
    - UMAP plot (if available)
    - UMAP plot by recording
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running t-SNE...")
    tsne_coords = run_tsne(features)
    np.save(output_dir / 'tsne_coords.npy', tsne_coords)
    
    # Basic t-SNE plot
    plot_embedding(
        tsne_coords,
        title="USV Feature Space (t-SNE)",
        save_path=output_dir / 'tsne_basic.png'
    )
    
    # t-SNE by recording
    plot_embedding_by_recording(
        tsne_coords,
        metadata,
        save_path=output_dir / 'tsne_by_recording.png'
    )
    
    # Try UMAP
    print("Running UMAP...")
    umap_coords = run_umap(features)
    if umap_coords is not None:
        np.save(output_dir / 'umap_coords.npy', umap_coords)
        
        plot_embedding(
            umap_coords,
            title="USV Feature Space (UMAP)",
            save_path=output_dir / 'umap_basic.png'
        )
        
        plot_embedding_by_recording(
            umap_coords,
            metadata,
            save_path=output_dir / 'umap_by_recording.png'
        )
    
    print(f"Visualizations saved to {output_dir}")
```

**Script:** `scripts/visualize_clusters.py`

```python
"""
CLI script to visualize USV feature space.

Usage:
    python scripts/visualize_clusters.py \
        --features analysis/clustering/features.npy \
        --metadata analysis/clustering/metadata.csv \
        --output analysis/clustering
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.clustering.visualize import create_visualization_report


def main():
    parser = argparse.ArgumentParser(description='Visualize USV feature space')
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('analysis/clustering'))
    
    args = parser.parse_args()
    
    features = np.load(args.features)
    metadata = pd.read_csv(args.metadata)
    
    create_visualization_report(features, metadata, args.output)


if __name__ == '__main__':
    main()
```

---

## Phase 3: Clustering

### Goal
Apply clustering algorithms to find natural groupings and evaluate cluster quality.

### Implementation

**File:** `src/usv_spectrogram/clustering/cluster.py`

```python
"""
Clustering algorithms for USV type discovery.

Tries multiple approaches:
- K-means with various k values
- HDBSCAN (automatic k selection)
- Gaussian Mixture Models

Evaluates cluster quality using silhouette score and other metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def run_kmeans_sweep(
    features: np.ndarray,
    k_range: range = range(2, 16),
    random_state: int = 42
) -> dict:
    """
    Run k-means with different k values and evaluate.
    
    Returns dict with results for each k.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {}
    
    for k in k_range:
        print(f"Running k-means with k={k}...")
        
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Compute metrics
        silhouette = silhouette_score(features_scaled, labels)
        calinski = calinski_harabasz_score(features_scaled, labels)
        inertia = kmeans.inertia_
        
        results[k] = {
            'labels': labels,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'inertia': inertia,
            'model': kmeans
        }
        
        print(f"  k={k}: silhouette={silhouette:.3f}, calinski={calinski:.1f}")
    
    return results


def run_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5
) -> tuple[np.ndarray, dict]:
    """
    Run HDBSCAN clustering (automatic k selection).
    
    HDBSCAN is good at finding clusters of varying density
    and automatically determines the number of clusters.
    
    Returns (labels, metrics_dict)
    """
    try:
        import hdbscan
    except ImportError:
        print("HDBSCAN not installed. Install with: pip install hdbscan")
        return None, None
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    labels = clusterer.fit_predict(features_scaled)
    
    # Count clusters (excluding noise labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    # Compute silhouette only on non-noise points
    non_noise_mask = labels != -1
    if non_noise_mask.sum() > 1 and n_clusters > 1:
        silhouette = silhouette_score(
            features_scaled[non_noise_mask], 
            labels[non_noise_mask]
        )
    else:
        silhouette = 0
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels),
        'silhouette': silhouette
    }
    
    print(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points ({metrics['noise_ratio']:.1%})")
    print(f"Silhouette score: {silhouette:.3f}")
    
    return labels, metrics


def run_gmm(
    features: np.ndarray,
    n_components_range: range = range(2, 16),
    random_state: int = 42
) -> dict:
    """
    Run Gaussian Mixture Model with different component counts.
    
    GMM allows soft clustering (probability of belonging to each cluster).
    Uses BIC/AIC for model selection.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {}
    
    for n in n_components_range:
        print(f"Running GMM with {n} components...")
        
        gmm = GaussianMixture(
            n_components=n,
            random_state=random_state,
            n_init=5
        )
        gmm.fit(features_scaled)
        
        labels = gmm.predict(features_scaled)
        probs = gmm.predict_proba(features_scaled)
        
        results[n] = {
            'labels': labels,
            'probabilities': probs,
            'bic': gmm.bic(features_scaled),
            'aic': gmm.aic(features_scaled),
            'model': gmm
        }
        
        print(f"  n={n}: BIC={results[n]['bic']:.1f}, AIC={results[n]['aic']:.1f}")
    
    return results


def plot_cluster_metrics(
    kmeans_results: dict,
    gmm_results: dict = None,
    save_path: Path = None
):
    """
    Plot clustering metrics to help choose optimal k.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # K-means metrics
    ks = sorted(kmeans_results.keys())
    
    # Elbow plot (inertia)
    ax = axes[0, 0]
    inertias = [kmeans_results[k]['inertia'] for k in ks]
    ax.plot(ks, inertias, 'bo-')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('K-means Elbow Plot')
    
    # Silhouette score
    ax = axes[0, 1]
    silhouettes = [kmeans_results[k]['silhouette'] for k in ks]
    ax.plot(ks, silhouettes, 'go-')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('K-means Silhouette Score')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Good threshold')
    ax.legend()
    
    # GMM BIC/AIC
    if gmm_results:
        ns = sorted(gmm_results.keys())
        
        ax = axes[1, 0]
        bics = [gmm_results[n]['bic'] for n in ns]
        aics = [gmm_results[n]['aic'] for n in ns]
        ax.plot(ns, bics, 'bo-', label='BIC')
        ax.plot(ns, aics, 'go-', label='AIC')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Score (lower is better)')
        ax.set_title('GMM Model Selection')
        ax.legend()
    
    # Best k summary
    ax = axes[1, 1]
    ax.axis('off')
    
    best_silhouette_k = max(ks, key=lambda k: kmeans_results[k]['silhouette'])
    summary_text = f"""
    Clustering Summary:
    
    K-means:
    - Best k by silhouette: {best_silhouette_k}
    - Silhouette score: {kmeans_results[best_silhouette_k]['silhouette']:.3f}
    
    Interpretation:
    - Silhouette > 0.5: Good clustering
    - Silhouette 0.25-0.5: Reasonable clustering  
    - Silhouette < 0.25: Weak clustering
    """
    
    if gmm_results:
        best_bic_n = min(ns, key=lambda n: gmm_results[n]['bic'])
        summary_text += f"""
    GMM:
    - Best n by BIC: {best_bic_n}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_cluster_assignments(
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    method: str = 'kmeans'
):
    """Save cluster assignments to CSV."""
    df = metadata.copy()
    df['cluster'] = labels
    df['cluster_method'] = method
    df.to_csv(output_path, index=False)
    print(f"Saved cluster assignments to {output_path}")


def run_full_clustering_analysis(
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path
):
    """
    Run complete clustering analysis.
    
    Saves:
    - Cluster assignments for best k-means
    - Cluster assignments for HDBSCAN
    - Metric plots
    - Summary report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # K-means sweep
    print("\n=== K-means Clustering ===")
    kmeans_results = run_kmeans_sweep(features)
    
    # GMM sweep
    print("\n=== GMM Clustering ===")
    gmm_results = run_gmm(features)
    
    # HDBSCAN
    print("\n=== HDBSCAN Clustering ===")
    hdbscan_labels, hdbscan_metrics = run_hdbscan(features)
    
    # Plot metrics
    plot_cluster_metrics(
        kmeans_results, 
        gmm_results,
        save_path=output_dir / 'cluster_metrics.png'
    )
    
    # Save best k-means assignments
    best_k = max(kmeans_results.keys(), 
                 key=lambda k: kmeans_results[k]['silhouette'])
    save_cluster_assignments(
        kmeans_results[best_k]['labels'],
        metadata,
        output_dir / f'kmeans_k{best_k}_assignments.csv',
        method=f'kmeans_k{best_k}'
    )
    
    # Save HDBSCAN assignments
    if hdbscan_labels is not None:
        save_cluster_assignments(
            hdbscan_labels,
            metadata,
            output_dir / 'hdbscan_assignments.csv',
            method='hdbscan'
        )
    
    # Return summary
    return {
        'kmeans': kmeans_results,
        'gmm': gmm_results,
        'hdbscan': {'labels': hdbscan_labels, 'metrics': hdbscan_metrics},
        'best_kmeans_k': best_k
    }
```

**Script:** `scripts/run_clustering.py`

```python
"""
CLI script to run clustering analysis.

Usage:
    python scripts/run_clustering.py \
        --features analysis/clustering/features.npy \
        --metadata analysis/clustering/metadata.csv \
        --output analysis/clustering
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.clustering.cluster import run_full_clustering_analysis


def main():
    parser = argparse.ArgumentParser(description='Run clustering analysis')
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('analysis/clustering'))
    
    args = parser.parse_args()
    
    features = np.load(args.features)
    metadata = pd.read_csv(args.metadata)
    
    run_full_clustering_analysis(features, metadata, args.output)


if __name__ == '__main__':
    main()
```

---

## Phase 4: Cluster Analysis and Interpretation

### Goal
Examine clusters to understand what they represent and compare across populations.

### Implementation

**File:** `src/usv_spectrogram/clustering/analysis.py`

```python
"""
Analyze and interpret discovered clusters.

Functions to:
- Extract exemplar spectrograms from each cluster
- Compare cluster distributions across populations/recordings
- Generate interpretable cluster summaries
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def extract_cluster_exemplars(
    cluster_labels: np.ndarray,
    features: np.ndarray,
    metadata: pd.DataFrame,
    spectrogram_dir: Path,
    output_dir: Path,
    n_exemplars: int = 10
):
    """
    Extract and save exemplar spectrograms from each cluster.
    
    Exemplars are the samples closest to each cluster centroid.
    """
    from sklearn.metrics import pairwise_distances
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_clusters = sorted(set(cluster_labels) - {-1})  # Exclude noise
    
    for cluster_id in unique_clusters:
        cluster_dir = output_dir / f'cluster_{cluster_id}'
        cluster_dir.mkdir(exist_ok=True)
        
        # Get indices in this cluster
        mask = cluster_labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_features = features[mask]
        
        # Find centroid
        centroid = cluster_features.mean(axis=0)
        
        # Find samples closest to centroid
        distances = pairwise_distances(cluster_features, centroid.reshape(1, -1)).squeeze()
        closest_indices = cluster_indices[np.argsort(distances)[:n_exemplars]]
        
        # Copy spectrograms
        for rank, idx in enumerate(closest_indices):
            row = metadata.iloc[idx]
            src_path = spectrogram_dir / row['spectrogram_path']
            
            if src_path.exists():
                # Copy with descriptive name
                dst_path = cluster_dir / f'exemplar_{rank:02d}_{src_path.name}'
                import shutil
                shutil.copy(src_path, dst_path)
        
        print(f"Cluster {cluster_id}: {mask.sum()} samples, saved {n_exemplars} exemplars")
    
    # Handle noise cluster if present
    if -1 in cluster_labels:
        noise_dir = output_dir / 'cluster_noise'
        noise_dir.mkdir(exist_ok=True)
        noise_mask = cluster_labels == -1
        noise_indices = np.where(noise_mask)[0][:n_exemplars]
        
        for rank, idx in enumerate(noise_indices):
            row = metadata.iloc[idx]
            src_path = spectrogram_dir / row['spectrogram_path']
            if src_path.exists():
                dst_path = noise_dir / f'noise_{rank:02d}_{src_path.name}'
                import shutil
                shutil.copy(src_path, dst_path)
        
        print(f"Noise: {noise_mask.sum()} samples")


def compare_clusters_by_recording(
    cluster_labels: np.ndarray,
    metadata: pd.DataFrame,
    recording_column: str = 'source_file',
    save_path: Path = None
):
    """
    Compare cluster distributions across recordings.
    
    Creates a heatmap showing which clusters are dominant in which recordings.
    This reveals whether certain recordings (e.g., wild mice) have different
    USV type distributions.
    """
    # Create cross-tabulation
    df = metadata.copy()
    df['cluster'] = cluster_labels
    
    # Filter out noise
    df = df[df['cluster'] != -1]
    
    # Cross-tabulate
    ct = pd.crosstab(df[recording_column], df['cluster'], normalize='index')
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(ct) * 0.5)))
    
    im = ax.imshow(ct.values, aspect='auto', cmap='YlOrRd')
    
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels([f'Cluster {c}' for c in ct.columns])
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels([str(r)[:30] for r in ct.index])
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Recording')
    ax.set_title('USV Cluster Distribution by Recording\n(row-normalized)')
    
    plt.colorbar(im, ax=ax, label='Proportion')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return ct


def generate_cluster_summary(
    cluster_labels: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path
):
    """
    Generate text summary of cluster characteristics.
    """
    df = metadata.copy()
    df['cluster'] = cluster_labels
    
    summary_lines = ["# USV Cluster Summary\n"]
    
    # Overall stats
    unique_clusters = sorted(set(cluster_labels) - {-1})
    summary_lines.append(f"Total USVs: {len(df)}")
    summary_lines.append(f"Number of clusters: {len(unique_clusters)}")
    
    if -1 in cluster_labels:
        n_noise = (cluster_labels == -1).sum()
        summary_lines.append(f"Noise samples: {n_noise} ({n_noise/len(df):.1%})")
    
    summary_lines.append("\n## Cluster Sizes\n")
    for cluster_id in unique_clusters:
        count = (cluster_labels == cluster_id).sum()
        summary_lines.append(f"- Cluster {cluster_id}: {count} samples ({count/len(df):.1%})")
    
    # Per-recording breakdown
    summary_lines.append("\n## Clusters by Recording\n")
    for recording in df['source_file'].unique():
        rec_df = df[df['source_file'] == recording]
        rec_clusters = rec_df['cluster'].value_counts()
        summary_lines.append(f"\n### {recording}")
        for cluster_id, count in rec_clusters.items():
            if cluster_id != -1:
                summary_lines.append(f"  - Cluster {cluster_id}: {count}")
    
    # Write summary
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Saved summary to {output_path}")
```

**Script:** `scripts/analyze_clusters.py`

```python
"""
CLI script to analyze clusters.

Usage:
    python scripts/analyze_clusters.py \
        --features analysis/clustering/features.npy \
        --metadata analysis/clustering/metadata.csv \
        --assignments analysis/clustering/kmeans_k8_assignments.csv \
        --spectrograms data/candidates/spectrograms \
        --output analysis/clustering
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.clustering.analysis import (
    extract_cluster_exemplars,
    compare_clusters_by_recording,
    generate_cluster_summary
)


def main():
    parser = argparse.ArgumentParser(description='Analyze USV clusters')
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--assignments', type=Path, required=True)
    parser.add_argument('--spectrograms', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('analysis/clustering'))
    
    args = parser.parse_args()
    
    features = np.load(args.features)
    assignments = pd.read_csv(args.assignments)
    cluster_labels = assignments['cluster'].values
    
    # Extract exemplars
    print("\nExtracting cluster exemplars...")
    extract_cluster_exemplars(
        cluster_labels,
        features,
        assignments,
        args.spectrograms,
        args.output / 'exemplars'
    )
    
    # Compare by recording
    print("\nComparing clusters by recording...")
    compare_clusters_by_recording(
        cluster_labels,
        assignments,
        save_path=args.output / 'cluster_by_recording.png'
    )
    
    # Generate summary
    print("\nGenerating summary...")
    generate_cluster_summary(
        cluster_labels,
        assignments,
        args.output / 'cluster_summary.md'
    )


if __name__ == '__main__':
    main()
```

---

## Execution Order

Run these commands in sequence:

```powershell
# 1. Install additional dependencies
pip install umap-learn hdbscan

# 2. Extract features from all USVs
python scripts/extract_features.py `
    --model models/production/best_model.pt `
    --spectrograms data/candidates/spectrograms `
    --metadata splits/all_labeled.csv `
    --output analysis/clustering

# 3. Visualize feature space (t-SNE, UMAP)
python scripts/visualize_clusters.py `
    --features analysis/clustering/features.npy `
    --metadata analysis/clustering/metadata.csv `
    --output analysis/clustering

# 4. Run clustering analysis
python scripts/run_clustering.py `
    --features analysis/clustering/features.npy `
    --metadata analysis/clustering/metadata.csv `
    --output analysis/clustering

# 5. Analyze best clustering result
# (adjust filename based on which k was best)
python scripts/analyze_clusters.py `
    --features analysis/clustering/features.npy `
    --metadata analysis/clustering/metadata.csv `
    --assignments analysis/clustering/kmeans_k8_assignments.csv `
    --spectrograms data/candidates/spectrograms `
    --output analysis/clustering
```

---

## Expected Outputs

After running all phases:

```
analysis/clustering/
├── features.npy                    # Extracted CNN features
├── metadata.csv                    # USV metadata with feature indices
├── tsne_coords.npy                 # t-SNE 2D coordinates
├── tsne_basic.png                  # t-SNE visualization (no labels)
├── tsne_by_recording.png           # t-SNE colored by recording
├── umap_coords.npy                 # UMAP 2D coordinates
├── umap_basic.png                  # UMAP visualization
├── umap_by_recording.png           # UMAP colored by recording
├── cluster_metrics.png             # Elbow plot, silhouette scores
├── kmeans_k8_assignments.csv       # Cluster assignments (best k)
├── hdbscan_assignments.csv         # HDBSCAN cluster assignments
├── cluster_by_recording.png        # Heatmap: clusters vs recordings
├── cluster_summary.md              # Text summary
└── exemplars/
    ├── cluster_0/                  # Example spectrograms from cluster 0
    ├── cluster_1/
    ├── ...
    └── cluster_noise/              # HDBSCAN noise points
```

---

## Interpreting Results

### What to Look For

1. **t-SNE/UMAP plots:**
   - Are there distinct clusters or a continuous blob?
   - Do recordings separate into different regions?
   - Is there overlap between recordings or clean separation?

2. **Cluster metrics:**
   - Silhouette > 0.5: Good clustering
   - Silhouette 0.25-0.5: Reasonable clustering
   - Silhouette < 0.25: Weak structure (maybe not discrete types)

3. **Cluster exemplars:**
   - Do exemplars within a cluster look similar?
   - Can you name the clusters (flat, sweep, chevron)?
   - Are some clusters mixed or unclear?

4. **Recording comparison:**
   - Do certain recordings have unique clusters?
   - This could indicate wild vs. lab differences

### Next Steps Based on Results

**If clear clusters exist (silhouette > 0.4):**
- Name the clusters based on exemplars
- Use cluster assignments as pseudo-labels
- Train supervised classifier for USV types

**If weak clusters (silhouette < 0.3):**
- USV types may not be discrete categories
- Consider continuous representation instead
- Or try different features (spectral, temporal)

**If recordings separate strongly:**
- Potential population differences (wild vs lab)
- Or recording condition differences
- Worth investigating which clusters are population-specific
