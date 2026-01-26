"""Sliding window CNN inference for USV detection.

Runs trained CNN model across spectrogram using sliding window to
generate per-timepoint probability predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...models.cnn_classifier import USVClassifierCNN


@dataclass
class InferenceResult:
    """Container for sliding window inference results."""

    probabilities: np.ndarray  # Shape: (n_windows,), probabilities in [0, 1]
    column_indices: np.ndarray  # Shape: (n_windows,), center column index for each window
    times: np.ndarray  # Shape: (n_windows,), center time in seconds for each window


class SlidingInference:
    """Performs sliding window CNN inference on spectrograms.

    Slides a fixed-width window (150px) across the spectrogram with specified
    hop length (10px default), runs CNN on each window, and returns per-window
    probability predictions.
    """

    def __init__(
        self,
        model_path: str | Path,
        window_width_px: int = 150,
        hop_px: int = 10,
        batch_size: int = 32,
        device: str | None = None
    ):
        """Initialize sliding inference.

        Args:
            model_path: Path to trained CNN checkpoint (.pt file)
            window_width_px: Width of sliding window in pixels
            hop_px: Hop length in pixels between consecutive windows
            batch_size: Batch size for CNN inference
            device: PyTorch device ('cpu', 'cuda', or None for auto)
        """
        self.window_width_px = window_width_px
        self.hop_px = hop_px
        self.batch_size = batch_size

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str | Path) -> USVClassifierCNN:
        """Load trained CNN model from checkpoint.

        Args:
            model_path: Path to checkpoint file

        Returns:
            Loaded CNN model

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Load checkpoint (weights_only=False for our trusted checkpoint)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Extract model state
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint

        # Instantiate model with same architecture
        model = USVClassifierCNN()
        model.load_state_dict(state_dict)

        return model

    def infer(
        self,
        spectrogram_db: np.ndarray,
        times: np.ndarray
    ) -> InferenceResult:
        """Run sliding window inference on spectrogram.

        Args:
            spectrogram_db: Spectrogram in dB, shape (freqs, times)
            times: Time bins in seconds, shape (times,)

        Returns:
            InferenceResult with probabilities and window positions

        Raises:
            ValueError: If spectrogram is too narrow for window size
        """
        n_freqs, n_times = spectrogram_db.shape

        # Check if spectrogram is wide enough
        if n_times < self.window_width_px:
            raise ValueError(
                f"Spectrogram too narrow ({n_times} columns) for "
                f"window width {self.window_width_px} px"
            )

        # Generate window positions (center column indices)
        window_centers = []
        col = self.window_width_px // 2
        while col + self.window_width_px // 2 < n_times:
            window_centers.append(col)
            col += self.hop_px

        window_centers = np.array(window_centers)
        n_windows = len(window_centers)

        if n_windows == 0:
            # No valid windows
            return InferenceResult(
                probabilities=np.empty(0),
                column_indices=np.empty(0, dtype=int),
                times=np.empty(0)
            )

        # Extract windows and run inference in batches
        all_probabilities = []

        for batch_start in range(0, n_windows, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_windows)
            batch_centers = window_centers[batch_start:batch_end]

            # Extract windows for this batch
            windows = []
            for center in batch_centers:
                start_col = center - self.window_width_px // 2
                end_col = start_col + self.window_width_px
                window = spectrogram_db[:, start_col:end_col]
                windows.append(window)

            # Convert to tensor batch
            batch_tensor = self._prepare_batch(windows)
            batch_tensor = batch_tensor.to(self.device)

            # Run inference (no gradient computation)
            with torch.no_grad():
                batch_probs = self.model.predict_proba(batch_tensor)

            # Convert to numpy
            batch_probs = batch_probs.cpu().numpy().flatten()
            all_probabilities.append(batch_probs)

        # Concatenate all batches
        probabilities = np.concatenate(all_probabilities)

        # Get times for window centers
        window_times = times[window_centers]

        return InferenceResult(
            probabilities=probabilities,
            column_indices=window_centers,
            times=window_times
        )

    def _prepare_batch(self, windows: list[np.ndarray]) -> torch.Tensor:
        """Prepare batch of spectrogram windows for CNN input.

        Args:
            windows: List of spectrogram windows, each shape (freqs, width)

        Returns:
            Tensor of shape (batch, 1, freqs, width)
        """
        # Stack windows and add channel dimension
        batch = np.stack(windows, axis=0)  # (batch, freqs, width)
        batch = batch[:, np.newaxis, :, :]  # (batch, 1, freqs, width)

        # Normalize to [0, 1] (assume dB range is approx -80 to 0)
        # Use robust normalization per image
        batch_normalized = []
        for img in batch:
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img_norm = (img - img_min) / (img_max - img_min)
            else:
                img_norm = np.zeros_like(img)
            batch_normalized.append(img_norm)

        batch = np.stack(batch_normalized, axis=0)

        # Convert to torch tensor
        batch_tensor = torch.from_numpy(batch).float()

        return batch_tensor
