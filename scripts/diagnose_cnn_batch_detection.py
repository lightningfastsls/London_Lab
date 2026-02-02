"""Diagnostic script to understand why CNN batch detection isn't working.

This script runs several tests to identify the root cause:
1. Test CNN on known labeled USVs (should give high probability)
2. Test CNN on known labeled noise (should give low probability)
3. Test CNN on random 40ms chunks from training files
4. Compare probability distributions

If the CNN works on labeled data but not on random chunks,
it confirms the hypothesis: CNN was trained as a CANDIDATE CLASSIFIER,
not as a CHUNK DETECTOR.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.data_loader import USVDataset


def load_model(model_path: Path, device: str = 'cpu'):
    """Load trained CNN model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model = USVClassifierCNN()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def test_on_labeled_data(model, csv_path: Path, spec_dir: Path, device: str = 'cpu'):
    """Test CNN on labeled data from training/validation splits.

    Returns dict with probabilities for USV and Not USV samples.
    """
    df = pd.read_csv(csv_path)

    # Handle different CSV formats
    if 'label' in df.columns:
        label_col = 'label'
    elif 'final_label' in df.columns:
        label_col = 'final_label'
    else:
        raise ValueError(f"No label column found in {csv_path}")

    usv_probs = []
    noise_probs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing on {csv_path.name}"):
        # Construct spectrogram path from candidate_id
        candidate_id = row['candidate_id']
        spec_path = spec_dir / f"{candidate_id}.png"
        if not spec_path.exists():
            continue

        # Load and preprocess (same as USVDataset)
        img = Image.open(spec_path).convert('L')
        img_array = np.array(img, dtype=np.float32)

        # Per-image normalization
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min)
        else:
            img_array = np.zeros_like(img_array)

        # Convert to tensor
        tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0).to(device)

        # Get probability
        with torch.no_grad():
            prob = model.predict_proba(tensor).item()

        if row[label_col] == 'USV':
            usv_probs.append(prob)
        else:
            noise_probs.append(prob)

    return {
        'usv_probs': np.array(usv_probs),
        'noise_probs': np.array(noise_probs)
    }


def test_on_random_chunks(model, wav_dir: Path, n_files: int = 3,
                          chunks_per_file: int = 50, device: str = 'cpu'):
    """Test CNN on random 40ms chunks from WAV files.

    This simulates what batch detection is trying to do.
    """
    from usv_spectrogram.app.core.audio_loader import AudioLoader
    from usv_spectrogram.app.core.sliding_inference import SlidingInference

    # Load audio
    audio_loader = AudioLoader()

    all_probs = []
    wav_files = sorted(wav_dir.glob("*.wav"))[:n_files]

    for wav_file in tqdm(wav_files, desc="Testing random chunks"):
        audio_data = audio_loader.load(str(wav_file))

        # Apply MAD normalization (same as SlidingInference)
        spec_db = audio_data.spectrogram_db
        median = np.median(spec_db)
        mad = np.median(np.abs(spec_db - median))
        vmin = median - 2.0 * mad
        vmax = median + 4.0 * mad
        spec_norm = np.clip(spec_db, vmin, vmax)
        spec_norm = (spec_norm - vmin) / (vmax - vmin + 1e-12)

        n_freqs, n_times = spec_norm.shape
        window_width = 100  # ~43ms

        # Sample random positions
        if n_times > window_width:
            positions = np.random.randint(0, n_times - window_width, size=chunks_per_file)

            for start_col in positions:
                window = spec_norm[:, start_col:start_col + window_width]

                # Apply colormap and convert to grayscale (same as SlidingInference._prepare_batch)
                cmap = plt.get_cmap('magma')
                rgb = cmap(window)[:, :, :3]
                rgb = np.flipud(rgb)
                rgb_uint8 = (rgb * 255).astype(np.uint8)

                # Resize to 256 height
                img = Image.fromarray(rgb_uint8)
                img_resized = img.resize((window_width, 256), Image.LANCZOS)

                # Convert to grayscale
                grayscale = np.array(img_resized.convert('L'), dtype=np.float32)

                # Per-image normalization
                g_min, g_max = grayscale.min(), grayscale.max()
                if g_max > g_min:
                    grayscale = (grayscale - g_min) / (g_max - g_min)
                else:
                    grayscale = np.zeros_like(grayscale)

                # Convert to tensor
                tensor = torch.from_numpy(grayscale).float().unsqueeze(0).unsqueeze(0).to(device)

                # Get probability
                with torch.no_grad():
                    prob = model.predict_proba(tensor).item()

                all_probs.append(prob)

    return np.array(all_probs)


def plot_diagnostic_results(labeled_results: dict, random_probs: np.ndarray,
                           output_path: Path):
    """Create diagnostic plot comparing probability distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram of labeled USV probabilities
    ax = axes[0, 0]
    ax.hist(labeled_results['usv_probs'], bins=50, alpha=0.7, label='Labeled USV', color='green')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    ax.axvline(x=0.9, color='orange', linestyle='--', label='Threshold 0.9')
    ax.set_xlabel('CNN Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'Labeled USV Samples (n={len(labeled_results["usv_probs"])})')
    ax.legend()

    # Plot 2: Histogram of labeled noise probabilities
    ax = axes[0, 1]
    ax.hist(labeled_results['noise_probs'], bins=50, alpha=0.7, label='Labeled Not USV', color='red')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    ax.axvline(x=0.9, color='orange', linestyle='--', label='Threshold 0.9')
    ax.set_xlabel('CNN Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'Labeled Not USV Samples (n={len(labeled_results["noise_probs"])})')
    ax.legend()

    # Plot 3: Histogram of random chunk probabilities
    ax = axes[1, 0]
    ax.hist(random_probs, bins=50, alpha=0.7, label='Random 40ms chunks', color='blue')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    ax.axvline(x=0.9, color='orange', linestyle='--', label='Threshold 0.9')
    ax.set_xlabel('CNN Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'Random Chunks from WAV Files (n={len(random_probs)})')
    ax.legend()

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    usv_probs = labeled_results['usv_probs']
    noise_probs = labeled_results['noise_probs']

    summary = f"""
    DIAGNOSTIC SUMMARY
    ==================

    LABELED USV SAMPLES (n={len(usv_probs)}):
      Mean probability: {usv_probs.mean():.3f}
      Median probability: {np.median(usv_probs):.3f}
      % above 0.5: {100 * (usv_probs > 0.5).mean():.1f}%
      % above 0.9: {100 * (usv_probs > 0.9).mean():.1f}%

    LABELED NOT USV SAMPLES (n={len(noise_probs)}):
      Mean probability: {noise_probs.mean():.3f}
      Median probability: {np.median(noise_probs):.3f}
      % above 0.5: {100 * (noise_probs > 0.5).mean():.1f}%
      % above 0.9: {100 * (noise_probs > 0.9).mean():.1f}%

    RANDOM CHUNKS (n={len(random_probs)}):
      Mean probability: {random_probs.mean():.3f}
      Median probability: {np.median(random_probs):.3f}
      % above 0.5: {100 * (random_probs > 0.5).mean():.1f}%
      % above 0.9: {100 * (random_probs > 0.9).mean():.1f}%

    INTERPRETATION:
    """

    # Add interpretation
    if usv_probs.mean() > 0.7 and noise_probs.mean() < 0.3:
        summary += """
    - CNN correctly separates labeled USV vs Not USV ✓
        """
    else:
        summary += """
    - WARNING: CNN struggles on labeled data ✗
        """

    if random_probs.mean() > 0.8:
        summary += """
    - Random chunks have HIGH probability
    - CNN thinks everything is USV!
    - This explains why batch detection produces giant detections.

    ROOT CAUSE: CNN was trained on energy-detector candidates,
    not arbitrary audio chunks. It expects USV to be centered/prominent.
        """
    elif random_probs.mean() < 0.3:
        summary += """
    - Random chunks have LOW probability
    - CNN correctly rejects most random chunks
    - Issue might be elsewhere in the pipeline.
        """
    else:
        summary += """
    - Random chunks have MIXED probability
    - CNN partially discriminates but not reliably.
        """

    ax.text(0.05, 0.95, summary, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved diagnostic plot to {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose CNN batch detection issues")
    parser.add_argument("--model", type=Path, default=Path("models/production/best_model.pt"))
    parser.add_argument("--test-csv", type=Path, default=Path("splits/test.csv"))
    parser.add_argument("--spec-dir", type=Path, default=Path("spectrograms_training"))
    parser.add_argument("--wav-dir", type=Path, default=Path("5970 USV"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/cnn_diagnosis"))
    parser.add_argument("--n-wav-files", type=int, default=3)
    parser.add_argument("--chunks-per-file", type=int, default=100)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CNN BATCH DETECTION DIAGNOSTIC")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model, args.device)

    # Test 1: Labeled data
    print(f"\nTest 1: Running CNN on labeled test data...")
    labeled_results = test_on_labeled_data(model, args.test_csv, args.spec_dir, args.device)

    print(f"  Labeled USV: n={len(labeled_results['usv_probs'])}, "
          f"mean prob={labeled_results['usv_probs'].mean():.3f}")
    print(f"  Labeled Not USV: n={len(labeled_results['noise_probs'])}, "
          f"mean prob={labeled_results['noise_probs'].mean():.3f}")

    # Test 2: Random chunks
    print(f"\nTest 2: Running CNN on random 40ms chunks from WAV files...")
    random_probs = test_on_random_chunks(
        model, args.wav_dir,
        n_files=args.n_wav_files,
        chunks_per_file=args.chunks_per_file,
        device=args.device
    )

    print(f"  Random chunks: n={len(random_probs)}, mean prob={random_probs.mean():.3f}")

    # Generate diagnostic plot
    print(f"\nGenerating diagnostic plot...")
    plot_diagnostic_results(
        labeled_results, random_probs,
        args.output_dir / "cnn_diagnostic_plot.png"
    )

    # Save raw data
    np.save(args.output_dir / "usv_probs.npy", labeled_results['usv_probs'])
    np.save(args.output_dir / "noise_probs.npy", labeled_results['noise_probs'])
    np.save(args.output_dir / "random_probs.npy", random_probs)

    # Print conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if random_probs.mean() > 0.7:
        print("""
The CNN gives HIGH probability to almost all random chunks.
This confirms the hypothesis:

  The CNN was trained as a CANDIDATE CLASSIFIER, not a CHUNK DETECTOR.

  Training data consisted of energy-detector candidates - spectrograms
  where USVs were pre-located and centered. The CNN learned to recognize
  USV features that are PROMINENT in the image, not to find them.

  When given arbitrary chunks, it sees "acoustic energy" and says "USV!"
  because it wasn't trained on true negatives from random audio positions.

OPTIONS:
  1. Use Energy Detector + CNN (original pipeline design)
  2. Retrain CNN with random negative chunks
  3. Use the labeled dataset for clustering (no expansion needed)
""")
    else:
        print("""
The CNN gives LOW/MIXED probability to random chunks.
The issue might be elsewhere - check:
  1. Preprocessing mismatch between training and inference
  2. Threshold settings
  3. Specific file characteristics
""")


if __name__ == "__main__":
    main()
