"""Test inference on a specific WAV file to verify fixes."""
import numpy as np
from pathlib import Path
from src.usv_spectrogram.app.core.audio_loader import AudioLoader
from src.usv_spectrogram.app.core.sliding_inference import SlidingInference

# Specify which WAV file and model to test
WAV_FILE = "2024-09-30_11-18-17_0000001.wav"  # Change this to test different files
MODEL_PATH = "models/production/best_model.pt"  # Or "checkpoints/best_model.pt"
wav_path = Path("5970 USV") / WAV_FILE

if not wav_path.exists():
    print(f"WAV file not found: {wav_path}")
    print("\nAvailable WAV files in '5970 USV':")
    for f in sorted(Path("5970 USV").glob("*.wav"))[:10]:
        print(f"  {f.name}")
    exit(1)

print(f"Testing: {WAV_FILE}")
print("="*60)

# Load audio
audio_loader = AudioLoader()
audio_data = audio_loader.load(wav_path)

print(f"Duration: {audio_data.duration_s:.2f}s")
print(f"Spectrogram shape: {audio_data.spectrogram_db.shape}")
print(f"Spec dB range: [{audio_data.spectrogram_db.min():.1f}, {audio_data.spectrogram_db.max():.1f}]")
print()

# Check if model exists
model_path = Path(MODEL_PATH)
if not model_path.exists():
    print(f"Model not found: {model_path}")
    print("Looking for alternative model paths...")
    for alt in ["checkpoints/best_model.pt", "models/production/best_model.pt", "models/clean_test/best_model.pt"]:
        if Path(alt).exists():
            print(f"  Found: {alt}")
            model_path = Path(alt)
            break
    if not model_path.exists():
        print("No model found!")
        exit(1)

print(f"Using model: {model_path}")
print()

# Run inference with current settings
inference = SlidingInference(
    model_path=model_path,
    window_width_px=100,
    hop_px=10,
    batch_size=32
)

result = inference.infer(audio_data.spectrogram_db, audio_data.times)

print(f"Inference results:")
print(f"  Number of windows: {len(result.probabilities)}")
print(f"  Window times: {result.times[0]:.3f}s to {result.times[-1]:.3f}s")
print(f"  Column indices: {result.column_indices[0]} to {result.column_indices[-1]}")
print()

print(f"Probability statistics:")
print(f"  Min:    {result.probabilities.min():.6f}")
print(f"  Max:    {result.probabilities.max():.6f}")
print(f"  Mean:   {result.probabilities.mean():.4f}")
print(f"  Median: {np.median(result.probabilities):.4f}")
print(f"  Std:    {result.probabilities.std():.4f}")
print()

print(f"Distribution:")
print(f"  Values == 1.0:     {(result.probabilities == 1.0).sum():4d} / {len(result.probabilities)} ({100*(result.probabilities == 1.0).sum()/len(result.probabilities):.1f}%)")
print(f"  Values > 0.99:     {(result.probabilities > 0.99).sum():4d} / {len(result.probabilities)} ({100*(result.probabilities > 0.99).sum()/len(result.probabilities):.1f}%)")
print(f"  Values > 0.50:     {(result.probabilities > 0.50).sum():4d} / {len(result.probabilities)} ({100*(result.probabilities > 0.50).sum()/len(result.probabilities):.1f}%)")
print(f"  Values < 0.10:     {(result.probabilities < 0.10).sum():4d} / {len(result.probabilities)} ({100*(result.probabilities < 0.10).sum()/len(result.probabilities):.1f}%)")
print()

print(f"First 20 probabilities:")
print(result.probabilities[:20])
print()

print(f"Last 20 probabilities:")
print(result.probabilities[-20:])
print()

if (result.probabilities == 1.0).all():
    print("⚠️  WARNING: All probabilities are exactly 1.0!")
    print("This indicates the preprocessing pipeline is still broken.")
    print()
    print("Please check:")
    print("1. Did you reload/restart the app after the fixes?")
    print("2. Is the sliding_inference.py file saved with all fixes?")
elif (result.probabilities > 0.95).sum() / len(result.probabilities) > 0.9:
    print("⚠️  WARNING: >90% of probabilities are >0.95")
    print("This still looks suspicious - most windows shouldn't be high confidence.")
else:
    print("✓ Probabilities look reasonable!")
    print(f"  Good variation from {result.probabilities.min():.4f} to {result.probabilities.max():.4f}")
