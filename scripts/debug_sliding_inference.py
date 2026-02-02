"""Debug script to understand why sliding inference produces 0 detections.

Compare:
1. Probability from predict.py on saved training spectrogram (works: 0.9266)
2. Probability from sliding inference on same USV region (fails: presumably low)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.app.core.audio_loader import AudioLoader
from usv_spectrogram.app.core.sliding_inference import SlidingInference

# Known USV from labels.csv: 2024-09-30_11-18-17_0000001_00000788
# This is the first USV in the file, candidate_id format suggests time ~788
wav_file = Path("5970 USV/2024-09-30_11-18-17_0000001.wav")
model_path = Path("models/production/best_model.pt")

print("=" * 80)
print("SLIDING INFERENCE DEBUG")
print("=" * 80)
print(f"WAV file: {wav_file}")
print(f"Model: {model_path}")
print()

# Load audio
print("Loading audio...")
audio_loader = AudioLoader()
audio_data = audio_loader.load(wav_file)
print(f"  Duration: {audio_data.duration_s:.2f} s")
print(f"  Spectrogram shape: {audio_data.spectrogram_db.shape}")
print(f"  Spectrogram dB range: [{audio_data.spectrogram_db.min():.1f}, {audio_data.spectrogram_db.max():.1f}]")
print()

# Run sliding inference
print("Running sliding inference...")
inference = SlidingInference(
    model_path=model_path,
    window_width_px=100,
    hop_px=10,
    batch_size=32,
    energy_threshold=0.35,
    enable_per_window_norm=False  # Back to original setting
)

result = inference.infer(
    audio_data.spectrogram_db,
    audio_data.times
)

print(f"  Total windows: {len(result.probabilities)}")
print(f"  Probability range: [{result.probabilities.min():.6f}, {result.probabilities.max():.6f}]")
print(f"  Probability mean: {result.probabilities.mean():.6f}")
print(f"  Probability median: {np.median(result.probabilities):.6f}")
print()

# Show top 10 probabilities
top_10_idx = np.argsort(result.probabilities)[-10:][::-1]
print("Top 10 probabilities:")
for i, idx in enumerate(top_10_idx, 1):
    prob = result.probabilities[idx]
    time = result.times[idx]
    col = result.column_indices[idx]
    print(f"  {i}. Time={time:.3f}s, Col={col}, P={prob:.6f}")
print()

# Check around known USV time (~788 in candidate_id, but unclear what that means)
# Let's check probabilities above different thresholds
thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50]
print("Windows above threshold:")
for thresh in thresholds:
    n_above = (result.probabilities >= thresh).sum()
    pct = 100 * n_above / len(result.probabilities)
    print(f"  >= {thresh:.2f}: {n_above} windows ({pct:.2f}%)")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
if result.probabilities.max() < 0.05:
    print("All probabilities are below threshold 0.05!")
    print("This confirms the preprocessing mismatch hypothesis.")
    print()
    print("Next steps:")
    print("1. Extract a window where we know there's a USV")
    print("2. Save it as PNG")
    print("3. Compare it to a training spectrogram visually")
    print("4. Check if they look different (brightness, contrast, etc.)")
else:
    print(f"Max probability {result.probabilities.max():.6f} is above 0.05")
    print("The model IS detecting something, but maybe detection logic filters it out?")
