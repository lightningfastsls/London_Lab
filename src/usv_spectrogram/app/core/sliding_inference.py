"""Sliding window CNN inference for USV detection.

Runs trained CNN model across spectrogram using sliding window to
generate per-timepoint probability predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...models.cnn_classifier import USVClassifierCNN

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for sliding window inference results."""

    probabilities: np.ndarray  # Shape: (n_windows,), probabilities in [0, 1]
    column_indices: np.ndarray  # Shape: (n_windows,), center column index for each window
    times: np.ndarray  # Shape: (n_windows,), center time in seconds for each window
    logits: np.ndarray | None = None  # Shape: (n_windows,), raw model outputs before sigmoid


class SlidingInference:
    """Performs sliding window CNN inference on spectrograms.

    Slides a fixed-width window (150px) across the spectrogram with specified
    hop length (10px default), runs CNN on each window, and returns per-window
    probability predictions.
    """

    def __init__(
        self,
        model_path: str | Path,
        window_width_px: int = 100,
        hop_px: int = 10,
        batch_size: int = 32,
        device: str | None = None,
        energy_threshold: float = 0.1,
        enable_per_window_norm: bool = False
    ):
        """Initialize sliding inference.

        Args:
            model_path: Path to trained CNN checkpoint (.pt file)
            window_width_px: Width of sliding window in pixels (default 100 = ~43ms at 300kHz)
            hop_px: Hop length in pixels between consecutive windows
            batch_size: Batch size for CNN inference
            device: PyTorch device ('cpu', 'cuda', or None for auto)
            energy_threshold: Skip CNN inference on windows with max < threshold (0-1 scale)
            enable_per_window_norm: Apply per-window normalization to match training distribution
        """
        self.window_width_px = window_width_px
        self.hop_px = hop_px
        self.batch_size = batch_size
        self.energy_threshold = energy_threshold
        self.enable_per_window_norm = enable_per_window_norm

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

        # Instantiate model with architecture from checkpoint (if available)
        model_kwargs = {}
        if isinstance(checkpoint, dict):
            if 'num_filters' in checkpoint:
                model_kwargs['num_filters'] = checkpoint['num_filters']
            if 'dense_units' in checkpoint:
                model_kwargs['dense_units'] = checkpoint['dense_units']
        model = USVClassifierCNN(**model_kwargs)
        model.load_state_dict(state_dict)

        return model

    def infer(
        self,
        spectrogram_db: np.ndarray,
        times: np.ndarray,
        return_logits: bool = False,
    ) -> InferenceResult:
        """Run sliding window inference on spectrogram.

        Args:
            spectrogram_db: Spectrogram in dB, shape (freqs, times)
            times: Time bins in seconds, shape (times,)
            return_logits: If True, also return raw logits for calibration.

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

        # CRITICAL FIX: Apply MAD normalization to ENTIRE spectrogram ONCE
        # This preserves the global energy distribution so quiet regions stay quiet
        # and loud USV regions stay loud. During training, spectrograms were
        # normalized per-candidate (which contained USVs), but at inference time
        # we need global normalization so the CNN can distinguish signal from noise.
        spectrogram_normalized = self._apply_mad_normalization(spectrogram_db)

        # NOTE: Vertical flip is done in _prepare_batch AFTER colormap,
        # to match training pipeline (flip RGB, not normalized spectrogram)

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
                times=np.empty(0),
                logits=np.empty(0, dtype=np.float32) if return_logits else None,
            )

        # Extract windows and run inference in batches
        all_probabilities = np.zeros(n_windows, dtype=np.float32)  # Pre-allocate, default 0.0
        # Energy-skipped windows get logit=-20.0 so sigmoid(-20/T) ≈ 0.0,
        # matching the probabilities=0.0 convention for skipped windows.
        all_logits = np.full(n_windows, -20.0, dtype=np.float32) if return_logits else None
        skipped_count = 0  # Track how many windows we skip
        window_max_values = []  # Track max values for debugging

        for batch_start in range(0, n_windows, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_windows)
            batch_centers = window_centers[batch_start:batch_end]

            # Extract windows for this batch (from globally-normalized spectrogram)
            # Apply energy pre-filter and per-window normalization
            windows = []
            window_indices = []  # Track which windows we actually process
            for batch_idx, center in enumerate(batch_centers):
                start_col = center - self.window_width_px // 2
                end_col = start_col + self.window_width_px
                window = spectrogram_normalized[:, start_col:end_col]

                # Track max value for debugging
                window_max = np.max(window)
                window_max_values.append(window_max)

                # Energy pre-filter: skip obviously quiet windows
                if self._should_skip_window_by_energy(window):
                    # Skip CNN inference, probability stays at 0.0
                    skipped_count += 1
                    continue

                # Per-window normalization: match training distribution
                if self.enable_per_window_norm:
                    window = self._normalize_window_to_training_distribution(window)

                windows.append(window)
                window_indices.append(batch_start + batch_idx)

            # Only run CNN if we have windows to process
            if len(windows) > 0:
                # Convert to tensor batch
                batch_tensor = self._prepare_batch(windows)
                batch_tensor = batch_tensor.to(self.device)

                # Run inference (no gradient computation)
                with torch.no_grad():
                    if return_logits:
                        batch_logits = self.model.forward(batch_tensor).squeeze(dim=1)
                        batch_probs = torch.sigmoid(batch_logits)
                        batch_logits_np = batch_logits.cpu().numpy().flatten()
                    else:
                        batch_probs = self.model.predict_proba(batch_tensor)

                # Convert to numpy
                batch_probs = batch_probs.cpu().numpy().flatten()

                # Fill in probabilities for processed windows
                for idx, prob in zip(window_indices, batch_probs):
                    all_probabilities[idx] = prob

                # Fill in logits if requested
                if return_logits:
                    for idx, logit in zip(window_indices, batch_logits_np):
                        all_logits[idx] = logit

        # Get times for window centers
        window_times = times[window_centers]

        # Debug output
        window_max_array = np.array(window_max_values)
        logger.debug("Total windows: %d", n_windows)
        logger.debug(
            "Window max values - min: %.3f, median: %.3f, max: %.3f",
            window_max_array.min(), np.median(window_max_array), window_max_array.max(),
        )
        logger.debug("Energy threshold: %.3f", self.energy_threshold)
        logger.info(
            "Windows skipped by energy filter: %d (%.1f%%)",
            skipped_count, 100 * skipped_count / n_windows,
        )
        logger.info("Windows processed by CNN: %d", n_windows - skipped_count)
        logger.debug(
            "Per-window normalization: %s",
            "enabled" if self.enable_per_window_norm else "disabled",
        )
        if self.enable_per_window_norm:
            logger.warning(
                "Per-window normalization adds extra normalization "
                "not present in training pipeline"
            )

        return InferenceResult(
            probabilities=all_probabilities,
            column_indices=window_centers,
            times=window_times,
            logits=all_logits,
        )

    def _prepare_batch(self, windows: list[np.ndarray]) -> torch.Tensor:
        """Prepare batch of spectrogram windows for CNN input.

        CRITICAL: Must match training pipeline EXACTLY:
        1. Input: Already MAD-normalized windows in [0, 1]
        2. Apply magma colormap (RGB)
        3. Convert to grayscale
        4. **RESIZE to 256px height** (training images were resized!)
        5. Per-image min/max normalization

        Args:
            windows: List of spectrogram windows already normalized to [0, 1], each shape (freqs, width)

        Returns:
            Tensor of shape (batch, 1, 256, width) - height MUST be 256 to match training
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        # Get magma colormap
        cmap = plt.get_cmap('magma')

        # Target height from training pipeline (ExtractionConfig.image_height_px)
        TARGET_HEIGHT = 256

        batch_normalized = []
        for window in windows:
            # Step 1: Apply magma colormap (windows already MAD-normalized in [0, 1])
            rgb = cmap(window)[:, :, :3]  # Drop alpha, shape (freqs, width, 3)

            # Step 2: FLIP VERTICALLY (same as training: flip RGB, not normalized spec!)
            # Training does: rgb_flipped = np.flipud(rgb)
            rgb = np.flipud(rgb)

            # Step 3: Convert RGB to uint8 (matches training pipeline)
            # Training does: rgb_uint8 = (rgb_flipped * 255).astype(np.uint8)
            rgb_uint8 = (rgb * 255).astype(np.uint8)

            # Step 4: RESIZE using PIL (matches training exactly)
            # Training: img = Image.fromarray(rgb_uint8); img_resized = img.resize(...)
            current_height, current_width = rgb_uint8.shape[0], rgb_uint8.shape[1]

            if current_height != TARGET_HEIGHT:
                img = Image.fromarray(rgb_uint8)
                img_resized = img.resize((current_width, TARGET_HEIGHT), Image.LANCZOS)
                rgb_uint8_resized = np.array(img_resized, dtype=np.uint8)
            else:
                rgb_uint8_resized = rgb_uint8

            # Step 5: Convert to grayscale (PIL's convert('L') equivalent)
            # Training data loader does: img.convert('L')
            # PIL uses: L = 0.299*R + 0.587*G + 0.114*B (on uint8 values)
            grayscale_uint8 = (
                0.299 * rgb_uint8_resized[:, :, 0] +
                0.587 * rgb_uint8_resized[:, :, 1] +
                0.114 * rgb_uint8_resized[:, :, 2]
            ).astype(np.uint8)

            # Step 6: Convert to float32 (same as data loader)
            grayscale = grayscale_uint8.astype(np.float32)

            # Step 7: Per-image min/max normalization (same as USVDataset)
            g_min, g_max = grayscale.min(), grayscale.max()
            if g_max > g_min:
                grayscale = (grayscale - g_min) / (g_max - g_min)
            else:
                grayscale = np.zeros_like(grayscale)

            batch_normalized.append(grayscale)

        # Stack and add channel dimension
        batch = np.stack(batch_normalized, axis=0)  # (batch, 256, width)
        batch = batch[:, np.newaxis, :, :]  # (batch, 1, 256, width)

        # Convert to torch tensor
        batch_tensor = torch.from_numpy(batch).float()

        # No padding — all windows are fixed width (window_columns), matching
        # training pipeline. The old 512px padding diluted signal through
        # GlobalAvgPool and is removed as part of the matched-window retrain.

        return batch_tensor

    def _normalize_window_to_training_distribution(self, window: np.ndarray) -> np.ndarray:
        """Normalize window to match training distribution.

        Training pipeline normalized magnitude before dB conversion, making max dB ≈ 0
        for each ~37ms candidate. This method applies similar boost to quiet windows
        so they match the brightness distribution the CNN was trained on.

        Args:
            window: MAD-normalized window, shape (freqs, width) in [0, 1]

        Returns:
            Normalized window where brightest value approaches 1.0
        """
        max_val = np.max(window)
        if max_val > 0.01:  # Avoid boosting pure noise
            # Rescale so max value → 1.0
            window = window / max_val
        return window

    def _should_skip_window_by_energy(self, window: np.ndarray) -> bool:
        """Check if window should be skipped due to low energy.

        Args:
            window: MAD-normalized window in [0, 1]

        Returns:
            True if window should be skipped (too quiet)
        """
        max_val = np.max(window)
        return max_val < self.energy_threshold

    def _apply_mad_normalization(self, spec_db: np.ndarray) -> np.ndarray:
        """Apply MAD-based normalization to entire spectrogram.

        CRITICAL: Must match training pipeline EXACTLY:
        1. Compute vmin/vmax using MAD
        2. CLIP spec_db to [vmin, vmax] FIRST
        3. THEN normalize to [0, 1]

        This is what _render_training() does in spectrogram_extractor.py.

        Args:
            spec_db: Spectrogram in dB, shape (freqs, times)

        Returns:
            Normalized spectrogram in [0, 1], shape (freqs, times)
        """
        # Compute MAD-based vmin/vmax (same as ExtractionConfig)
        median = np.median(spec_db)
        mad = np.median(np.abs(spec_db - median))

        # Use same scale factors as training spectrograms
        mad_vmin_scale = 2.0
        mad_vmax_scale = 4.0

        vmin = median - mad_vmin_scale * mad
        vmax = median + mad_vmax_scale * mad

        # CRITICAL: Clip BEFORE normalizing (matches training pipeline)
        spec_clipped = np.clip(spec_db, vmin, vmax)

        # Normalize to [0, 1]
        if vmax > vmin:
            spec_norm = (spec_clipped - vmin) / (vmax - vmin + 1e-12)
        else:
            spec_norm = np.zeros_like(spec_db)

        return spec_norm
