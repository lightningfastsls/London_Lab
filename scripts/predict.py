"""Run inference with trained CNN model.

Usage:
    # Single image
    python scripts/predict.py --model models/production/best_model.pt --image path/to/spectrogram.png

    # Batch prediction from CSV
    python scripts/predict.py --model models/production/best_model.pt --csv candidates.csv --output predictions.csv
"""

import argparse
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint
from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
from torch.utils.data import DataLoader

# Maximum spectrogram width in training set (for consistent padding)
MAX_SPEC_WIDTH = 512


def pad_to_max_width(spectrogram: torch.Tensor, max_width: int = MAX_SPEC_WIDTH) -> torch.Tensor:
    """Pad spectrogram to fixed maximum width.

    Args:
        spectrogram: Input tensor of shape (B, C, H, W)
        max_width: Target width to pad to

    Returns:
        Padded tensor of shape (B, C, H, max_width)
    """
    current_width = spectrogram.shape[3]
    if current_width < max_width:
        pad_width = max_width - current_width
        # Pad: (left, right, top, bottom)
        spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_width, 0, 0), value=0)
    return spectrogram


def predict_single(model, image_path: Path, device, normalize_mode: str = 'per_image'):
    """Run prediction on a single image.

    Args:
        model: Trained model
        image_path: Path to spectrogram image
        device: Device to run on
        normalize_mode: Normalization mode

    Returns:
        probability: Probability of USV (0 to 1)
        prediction: Binary prediction (0 or 1)
    """
    # Create temporary CSV for single image
    temp_csv = Path(tempfile.mktemp(suffix='.csv'))

    # Write minimal CSV with just this image
    temp_df = pd.DataFrame({
        'candidate_id': ['temp'],
        'spectrogram_path': [str(image_path.resolve())],
        'label': ['USV']  # Dummy label (not used for prediction)
    })
    temp_df.to_csv(temp_csv, index=False)

    try:
        # Load image with DataLoader
        dataset = USVDataset(temp_csv, normalize_mode=normalize_mode)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x  # No collate, just batch
        )

        # Predict
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                spectrograms, labels = zip(*batch)
                spectrogram = torch.stack(spectrograms)

                # Pad to fixed max width (consistent with training)
                spectrogram = pad_to_max_width(spectrogram)
                spectrogram = spectrogram.to(device)

                probability = model.predict_proba(spectrogram).item()
                prediction = model.predict(spectrogram).item()
                break
    finally:
        # Clean up temp file
        temp_csv.unlink(missing_ok=True)

    return probability, int(prediction)


def predict_batch(
    model,
    csv_path: Path,
    device,
    normalize_mode: str = 'per_image',
    output_path: Path = None,
    batch_size: int = 16
):
    """Run predictions on a batch of images from CSV.

    Args:
        model: Trained model
        csv_path: Path to CSV with spectrogram_path column
        device: Device to run on
        normalize_mode: Normalization mode
        output_path: Optional path to save predictions CSV
        batch_size: Batch size for processing (default: 16)

    Returns:
        DataFrame with predictions
    """
    # Load CSV
    df_original = pd.read_csv(csv_path)

    if 'spectrogram_path' not in df_original.columns:
        raise ValueError("CSV must have 'spectrogram_path' column")

    # Add label column if missing (dummy labels for prediction)
    if 'label' not in df_original.columns:
        df_original['label'] = 'USV'

    print(f"Running predictions on {len(df_original)} samples...")

    # Use DataLoader
    dataset = USVDataset(csv_path, normalize_mode=normalize_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,  # Pads within batch
        pin_memory=False
    )

    all_probabilities = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (spectrogram_batch, _) in enumerate(dataloader):
            # Pad to fixed max width (consistent across all batches)
            spectrogram_batch = pad_to_max_width(spectrogram_batch)
            spectrogram_batch = spectrogram_batch.to(device)

            # Get predictions
            proba_batch = model.predict_proba(spectrogram_batch)
            pred_batch = model.predict(spectrogram_batch)

            # Store results
            all_probabilities.extend(proba_batch.cpu().numpy().flatten())
            all_predictions.extend(pred_batch.cpu().numpy().flatten())

            if (batch_idx + 1) % 10 == 0:
                processed = min((batch_idx + 1) * batch_size, len(dataset))
                print(f"  Processed {processed}/{len(dataset)} samples")

    # Handle case where dataset filtered some samples
    if len(all_probabilities) < len(df_original):
        # Match predictions to original dataframe
        df_valid = df_original[df_original['spectrogram_path'].apply(
            lambda x: Path(x).exists()
        )].copy()
        df_valid['probability'] = all_probabilities
        df_valid['prediction'] = [int(p) for p in all_predictions]

        # Merge back to original
        df_result = df_original.merge(
            df_valid[['spectrogram_path', 'probability', 'prediction']],
            on='spectrogram_path',
            how='left'
        )
    else:
        df_result = df_original.copy()
        df_result['probability'] = all_probabilities
        df_result['prediction'] = [int(p) for p in all_predictions]

    # Add predicted label
    df_result['predicted_label'] = df_result['prediction'].apply(
        lambda x: 'USV' if x == 1 else 'Not USV' if x == 0 else None
    )

    # Save if output path provided
    if output_path:
        df_result.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")

    # Print summary
    pred_counts = df_result['predicted_label'].value_counts()
    print("\nPrediction Summary:")
    print(pred_counts)

    return df_result


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained CNN model')

    # Model
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=Path,
        help='Path to single spectrogram image'
    )
    input_group.add_argument(
        '--csv',
        type=Path,
        help='Path to CSV with spectrogram_path column'
    )

    # Options
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV path (for batch mode)'
    )
    parser.add_argument(
        '--normalize-mode',
        type=str,
        default='per_image',
        choices=['per_image', 'global'],
        help='Normalization mode (default: per_image)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu, default: auto)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for batch prediction (default: 16)'
    )

    args = parser.parse_args()

    # Verify model exists
    if not args.model.exists():
        print(f"Error: Model checkpoint not found: {args.model}")
        sys.exit(1)

    print("=" * 80)
    print("USV Prediction")
    print("=" * 80)
    print(f"Model: {args.model}")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model, checkpoint = load_model_checkpoint(
        args.model,
        USVClassifierCNN,
        device=args.device
    )
    device = next(model.parameters()).device
    print(f"Model loaded successfully")
    print()

    # Run prediction
    if args.image:
        # Single image prediction
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

        print(f"Predicting: {args.image}")
        probability, prediction = predict_single(
            model,
            args.image,
            device,
            normalize_mode=args.normalize_mode
        )

        print()
        print("=" * 80)
        print("Prediction Result")
        print("=" * 80)
        print(f"Probability: {probability:.4f}")
        print(f"Prediction: {'USV' if prediction == 1 else 'Not USV'}")
        print("=" * 80)

    elif args.csv:
        # Batch prediction
        if not args.csv.exists():
            print(f"Error: CSV not found: {args.csv}")
            sys.exit(1)

        df = predict_batch(
            model,
            args.csv,
            device,
            normalize_mode=args.normalize_mode,
            output_path=args.output,
            batch_size=args.batch_size
        )

        print()
        print("=" * 80)
        print("Batch Prediction Complete!")
        print("=" * 80)


if __name__ == '__main__':
    main()
