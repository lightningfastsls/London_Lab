"""Compare preprocessing between training spectrograms and sliding inference windows.

Extract a window from sliding inference pipeline and save it, then compare
to a training spectrogram visually and numerically.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.app.core.audio_loader import AudioLoader
from usv_spectrogram.app.core.sliding_inference import SlidingInference
from usv_spectrogram.models.data_loader import USVDataset

# Test on known USV file
wav_file = Path("5970 USV/2024-09-30_11-18-17_0000001.wav")
model_path = Path("models/production/best_model.pt")
train_spec = Path("data/full_training_dataset/spectrograms/2024-09-30_11-19-34_0000014_00000413.png")
output_dir = Path("analysis/preprocessing_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PREPROCESSING COMPARISON")
print("=" * 80)
print()

# 1. Load and process a training spectrogram (via data_loader.py)
print("1. Loading training spectrogram through data_loader.py...")
img_train = Image.open(train_spec).convert('L')
spec_train = np.array(img_train, dtype=np.float32)

# data_loader.py normalization
spec_min = spec_train.min()
spec_max = spec_train.max()
if spec_max > spec_min:
    spec_train_norm = (spec_train - spec_min) / (spec_max - spec_min)
else:
    spec_train_norm = np.zeros_like(spec_train)

print(f"   Shape: {spec_train.shape}")
print(f"   Raw range: [{spec_train.min():.1f}, {spec_train.max():.1f}]")
print(f"   Normalized range: [{spec_train_norm.min():.3f}, {spec_train_norm.max():.3f}]")
print(f"   Mean: {spec_train_norm.mean():.3f}, Std: {spec_train_norm.std():.3f}")
print()

# 2. Extract a window from sliding inference at a likely USV location
print("2. Extracting window from sliding inference...")
audio_loader = AudioLoader()
audio_data = audio_loader.load(wav_file)

# Try window around first USV (candidate 788 → frame 788 → ~0.34s)
target_time = 0.34  # seconds (frame 788 * 128/300000)
target_col = int(target_time * audio_data.sample_rate / audio_loader.config.hop_length)
window_width = 100

if target_col + window_width // 2 < audio_data.spectrogram_db.shape[1]:
    # Extract window from spec_db
    start_col = target_col - window_width // 2
    end_col = start_col + window_width
    window_db = audio_data.spectrogram_db[:, start_col:end_col]

    print(f"   Extracted at time={target_time}s, col={target_col}")
    print(f"   Window dB range: [{window_db.min():.1f}, {window_db.max():.1f}]")

    # Now process through SlidingInference._prepare_batch
    inference = SlidingInference(
        model_path=model_path,
        window_width_px=100,
        hop_px=10,
        batch_size=32,
        energy_threshold=0.35,
        enable_per_window_norm=False
    )

    # Apply MAD normalization (what SlidingInference does)
    spec_norm = inference._apply_mad_normalization(audio_data.spectrogram_db)
    window_norm = spec_norm[:, start_col:end_col]

    print(f"   After MAD norm range: [{window_norm.min():.3f}, {window_norm.max():.3f}]")

    # Apply _prepare_batch preprocessing
    batch_tensor = inference._prepare_batch([window_norm])
    window_processed = batch_tensor[0, 0].cpu().numpy()  # Shape: (256, width)

    print(f"   After prepare_batch shape: {window_processed.shape}")
    print(f"   After prepare_batch range: [{window_processed.min():.3f}, {window_processed.max():.3f}]")
    print(f"   Mean: {window_processed.mean():.3f}, Std: {window_processed.std():.3f}")
    print()

    # Save as PNG for visual inspection
    window_png = (window_processed * 255).astype(np.uint8)
    Image.fromarray(window_png).save(output_dir / "inference_window.png")
    print(f"   Saved to: {output_dir / 'inference_window.png'}")
    print()

    # 3. Visual comparison
    print("3. Creating visual comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training spectrogram
    axes[0].imshow(spec_train_norm, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title(f"Training Spectrogram\n{train_spec.name}")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")

    # Inference window
    axes[1].imshow(window_processed, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(f"Inference Window\nFrom {wav_file.name} @ {target_time}s")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency")

    # Histogram comparison
    axes[2].hist(spec_train_norm.flatten(), bins=50, alpha=0.5, label='Training', density=True)
    axes[2].hist(window_processed.flatten(), bins=50, alpha=0.5, label='Inference', density=True)
    axes[2].set_xlabel("Pixel Value")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Pixel Value Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "preprocessing_comparison.png", dpi=150)
    print(f"   Saved comparison to: {output_dir / 'preprocessing_comparison.png'}")
    print()

    # 4. Statistical comparison
    print("4. Statistical Comparison:")
    print(f"   Training - Mean: {spec_train_norm.mean():.3f}, Std: {spec_train_norm.std():.3f}")
    print(f"   Inference - Mean: {window_processed.mean():.3f}, Std: {window_processed.std():.3f}")
    print(f"   Difference - Mean: {abs(spec_train_norm.mean() - window_processed.mean()):.3f}, Std: {abs(spec_train_norm.std() - window_processed.std()):.3f}")
    print()

    if abs(spec_train_norm.mean() - window_processed.mean()) > 0.1:
        print("   ⚠️  Large difference in mean values!")
    if abs(spec_train_norm.std() - window_processed.std()) > 0.1:
        print("   ⚠️  Large difference in standard deviation!")

    # 5. Run CNN predictions on both
    print("5. Running CNN predictions:")
    import torch
    from usv_spectrogram.models.cnn_classifier import USVClassifierCNN

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model = USVClassifierCNN()
    model.load_state_dict(state_dict)
    model.eval()

    # Pad function (same as predict.py)
    def pad_to_512(tensor):
        """Pad tensor to 512px width."""
        current_width = tensor.shape[3]
        if current_width < 512:
            pad_width = 512 - current_width
            tensor = torch.nn.functional.pad(tensor, (0, pad_width, 0, 0), value=0)
        return tensor

    # Predict on training spectrogram (with padding)
    with torch.no_grad():
        x_train = torch.from_numpy(spec_train_norm).unsqueeze(0).unsqueeze(0).float()
        x_train_padded = pad_to_512(x_train)
        prob_train = model.predict_proba(x_train_padded).item()

    # Predict on inference window (with padding)
    with torch.no_grad():
        x_inference = torch.from_numpy(window_processed).unsqueeze(0).unsqueeze(0).float()
        x_inference_padded = pad_to_512(x_inference)
        prob_inference = model.predict_proba(x_inference_padded).item()

    print(f"   Training spectrogram: P = {prob_train:.6f}")
    print(f"   Inference window: P = {prob_inference:.6f}")
    print(f"   Difference: {abs(prob_train - prob_inference):.6f}")
    print()

    if prob_inference < 0.05:
        print("   🚨 Inference window gets low probability!")
        print("   Even though preprocessing looks similar, model outputs different results.")
        print("   Possible causes:")
        print("   1. Window doesn't contain a USV (check labels.csv for exact time)")
        print("   2. Subtle preprocessing difference not visible in statistics")
        print("   3. Model issue (wrong model loaded?)")
    elif prob_train > 0.5 and prob_inference > 0.5:
        print("   ✓ Both get high probabilities - preprocessing is equivalent!")
    else:
        print("   ? Mixed results - need further investigation")

    print()
    print("=" * 80)
    print("Open the comparison PNG to visually inspect preprocessing differences.")
    print("=" * 80)

else:
    print(f"   ERROR: Target column {target_col} out of range")
