"""Extract CNN embeddings from USV spectrograms for clustering."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union, List

from ..models.data_loader import USVDataset, pad_collate_fn


class FeatureExtractor:
    """Extract 128-dimensional embeddings from CNN global_pool layer.

    This extracts features from the trained CNN's intermediate representation
    (after convolutional layers, before classification) for clustering analysis.

    Args:
        model_path: Path to trained CNN model (.pt file)
        device: Device for inference ('cpu' or 'cuda')
        batch_size: Batch size for feature extraction (default 32)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'cpu',
        batch_size: int = 32
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = batch_size

        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = torch.load(
            self.model_path,
            map_location=device,
            weights_only=False
        )
        self.model.to(device)
        self.model.eval()

        # Register hook to capture global_pool output
        self.embeddings_cache = []

        def hook_fn(module, input, output):
            """Capture (B, 128, 1, 1) output from global_pool."""
            # Flatten to (B, 128)
            embeddings = output.view(output.size(0), -1)
            self.embeddings_cache.append(embeddings.cpu().detach().numpy())

        # Hook the global_pool layer
        self.model.global_pool.register_forward_hook(hook_fn)

        print(f"Model loaded on {device}")
        print(f"Embedding dimension: 128")

    def extract_from_csv(
        self,
        csv_path: Union[str, Path],
        output_path: Union[str, Path],
        data_source: str = "unknown"
    ) -> pd.DataFrame:
        """Extract embeddings from spectrograms listed in CSV.

        Args:
            csv_path: Path to CSV with columns: candidate_id, spectrogram_path, [label]
            output_path: Path to save embeddings CSV
            data_source: Label for data source (e.g., 'labeled', 'auto_detected')

        Returns:
            DataFrame with embeddings and metadata
        """
        csv_path = Path(csv_path)
        output_path = Path(output_path)

        print(f"\nExtracting features from {csv_path}")
        print(f"Data source: {data_source}")

        # Load dataset
        dataset = USVDataset(csv_path, normalize_mode='per_image')
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collate_fn
        )

        print(f"Total samples: {len(dataset)}")

        # Extract embeddings
        self.embeddings_cache = []
        all_embeddings = []

        with torch.no_grad():
            for spectrograms, labels in tqdm(loader, desc="Extracting embeddings"):
                # Move to device
                spectrograms = spectrograms.to(self.device)

                # Forward pass (hook captures embeddings)
                _ = self.model(spectrograms)

        # Concatenate all batches
        all_embeddings = np.vstack(self.embeddings_cache)
        print(f"Embeddings shape: {all_embeddings.shape}")

        # Create DataFrame with metadata + embeddings
        df = dataset.df.copy()

        # Add data source column
        df['data_source'] = data_source

        # Add embedding columns (embedding_0 to embedding_127)
        for i in range(all_embeddings.shape[1]):
            df[f'embedding_{i}'] = all_embeddings[:, i]

        # Reorder columns: metadata first, then embeddings
        metadata_cols = ['candidate_id', 'source_file', 'data_source']
        if 'label' in df.columns:
            metadata_cols.append('label')
        if 'max_prob' in df.columns:
            metadata_cols.extend(['max_prob', 'mean_prob'])

        embedding_cols = [f'embedding_{i}' for i in range(all_embeddings.shape[1])]
        df = df[metadata_cols + embedding_cols]

        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Embeddings saved to {output_path}")

        return df

    def extract_from_multiple_csvs(
        self,
        csv_paths: List[tuple],
        output_path: Union[str, Path]
    ) -> pd.DataFrame:
        """Extract embeddings from multiple CSV files and combine.

        Args:
            csv_paths: List of (csv_path, data_source) tuples
            output_path: Path to save combined embeddings CSV

        Returns:
            Combined DataFrame with all embeddings
        """
        all_dfs = []

        for csv_path, data_source in csv_paths:
            # Extract to temporary DataFrame
            df = self.extract_from_csv(
                csv_path=csv_path,
                output_path=Path(output_path).with_suffix('.tmp.csv'),
                data_source=data_source
            )
            all_dfs.append(df)

        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)

        print(f"\nCombined dataset:")
        print(f"Total samples: {len(combined_df)}")
        print(f"Data sources:")
        print(combined_df['data_source'].value_counts())

        # Save combined CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print(f"\nCombined embeddings saved to {output_path}")

        # Clean up temporary file
        tmp_file = Path(output_path).with_suffix('.tmp.csv')
        if tmp_file.exists():
            tmp_file.unlink()

        return combined_df
