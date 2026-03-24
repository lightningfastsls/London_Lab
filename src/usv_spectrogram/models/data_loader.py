"""PyTorch data loaders for USV spectrograms."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, Optional


class USVDataset(Dataset):
    """PyTorch Dataset for USV spectrograms.

    Loads spectrogram images and labels from CSV file. Spectrograms are
    converted from RGBA PNGs to grayscale, normalized to [0, 1], and
    returned as single-channel tensors.

    Args:
        csv_path: Path to CSV file with columns: candidate_id, spectrogram_path, label
        transform: Optional torchvision transforms to apply
        normalize_mode: Normalization mode ('per_image' or 'global')
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform=None,
        normalize_mode: str = 'per_image'
    ):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.normalize_mode = normalize_mode

        # Load CSV
        self.df = pd.read_csv(self.csv_path)

        # Filter valid labels (NOT 'Uncertain')
        valid_labels = ['USV', 'Not USV']
        initial_count = len(self.df)
        self.df = self.df[self.df['label'].isin(valid_labels)].copy()
        filtered_count = initial_count - len(self.df)
        if filtered_count > 0:
            print(f"Filtered {filtered_count} samples with invalid labels")

        # Label mapping (case-sensitive, NOT 'noise')
        self.label_map = {'Not USV': 0, 'USV': 1}

        # Verify all spectrograms exist
        self.df['spec_path'] = self.df['spectrogram_path'].apply(Path)
        missing = ~self.df['spec_path'].apply(lambda p: p.exists())
        if missing.any():
            missing_count = missing.sum()
            print(f"Warning: {missing_count} spectrogram files not found, removing from dataset")
            self.df = self.df[~missing].copy()

        self.df = self.df.reset_index(drop=True)

        # Compute class counts
        self.class_counts = self.df['label'].value_counts().to_dict()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Returns:
            spectrogram: (1, H, W) tensor, normalized to [0, 1]
            label: Scalar tensor (0.0 or 1.0)
        """
        row = self.df.iloc[idx]

        # Load image (RGBA PNG) and convert to grayscale
        spec_path = row['spec_path']
        img = Image.open(spec_path).convert('L')  # L = grayscale
        spectrogram = np.array(img, dtype=np.float32)

        # Normalize
        if self.normalize_mode == 'per_image':
            # Per-image normalization to [0, 1]
            spec_min = spectrogram.min()
            spec_max = spectrogram.max()
            if spec_max > spec_min:
                spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
            else:
                spectrogram = np.zeros_like(spectrogram)
        elif self.normalize_mode == 'global':
            # Assume images are already in [0, 255] range
            spectrogram = spectrogram / 255.0
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

        # Convert to tensor (1, H, W)
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)

        # Apply transforms if provided
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        # Get label
        label_str = row['label']
        label = float(self.label_map[label_str])

        return spectrogram, torch.tensor(label, dtype=torch.float32)

    def get_class_weights(self) -> Dict[int, float]:
        """Compute class weights for imbalanced dataset.

        Returns:
            Dictionary mapping class index to weight (inverse frequency)
        """
        total = len(self.df)
        weights = {}
        for label_str, class_idx in self.label_map.items():
            count = self.class_counts.get(label_str, 0)
            if count > 0:
                weights[class_idx] = total / (len(self.label_map) * count)
            else:
                weights[class_idx] = 1.0
        return weights


def pad_collate_fn(batch):
    """Custom collate function to pad variable-sized spectrograms.

    Pads all spectrograms in the batch to match the largest dimensions.

    Args:
        batch: List of (spectrogram, label) tuples

    Returns:
        spectrograms: Padded batch tensor (B, 1, H, W)
        labels: Batch tensor (B,)
    """
    spectrograms, labels = zip(*batch)

    # Find max dimensions in batch
    max_height = max(s.shape[1] for s in spectrograms)
    max_width = max(s.shape[2] for s in spectrograms)

    # Pad each spectrogram to max dimensions
    padded = []
    for spec in spectrograms:
        _, h, w = spec.shape
        pad_h = max_height - h
        pad_w = max_width - w

        # Pad: (left, right, top, bottom)
        padded_spec = torch.nn.functional.pad(
            spec,
            (0, pad_w, 0, pad_h),
            mode='constant',
            value=0
        )
        padded.append(padded_spec)

    # Stack into batch
    spectrograms_batch = torch.stack(padded, dim=0)
    labels_batch = torch.stack(labels, dim=0)

    return spectrograms_batch, labels_batch


def create_data_loaders(
    train_csv: str | Path,
    val_csv: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    normalize_mode: str = 'per_image',
    use_class_weights: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[Dict[int, float]]]:
    """Create training and validation data loaders.

    Args:
        train_csv: Path to training split CSV
        val_csv: Path to validation split CSV
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        normalize_mode: Normalization mode ('per_image' or 'global')
        use_class_weights: Whether to compute class weights

    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        class_weights: Dictionary of class weights (if use_class_weights=True)
    """
    # Create datasets
    train_dataset = USVDataset(train_csv, normalize_mode=normalize_mode)
    val_dataset = USVDataset(val_csv, normalize_mode=normalize_mode)

    # Compute class weights from training set
    class_weights = None
    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"Class weights: {class_weights}")

    # Create data loaders with custom collate function for padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training class distribution: {train_dataset.class_counts}")
    print(f"Validation class distribution: {val_dataset.class_counts}")

    return train_loader, val_loader, class_weights
