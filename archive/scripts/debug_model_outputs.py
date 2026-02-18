"""Debug what the models are actually outputting (logits and probabilities)."""
import torch
import numpy as np
from pathlib import Path
from src.usv_spectrogram.app.core.audio_loader import AudioLoader
from src.usv_spectrogram.app.core.sliding_inference import SlidingInference

# Test both models
models = {
    "checkpoints": Path("checkpoints/best_model.pt"),
    "production": Path("models/production/best_model.pt"),
}

wav_path = Path("5970 USV/2024-09-30_11-18-17_0000001.wav")
audio_loader = AudioLoader()
audio_data = audio_loader.load(wav_path)

print("="*60)
print("COMPARING MODEL OUTPUTS (Logits vs Probabilities)")
print("="*60)

for model_name, model_path in models.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_path})")
    print(f"{'='*60}")

    # Create inference engine
    inference = SlidingInference(
        model_path=model_path,
        window_width_px=100,
        hop_px=10,
        batch_size=32
    )

    # Manually extract first few windows and check logits
    spec_db = audio_data.spectrogram_db

    # Apply same preprocessing as in sliding_inference
    median = np.median(spec_db)
    mad = np.median(np.abs(spec_db - median))
    vmin = median - 2.0 * mad
    vmax = median + 4.0 * mad
    spec_clipped = np.clip(spec_db, vmin, vmax)
    spec_norm = (spec_clipped - vmin) / (vmax - vmin + 1e-12)
    spec_norm = np.flipud(spec_norm)

    # Extract first 5 windows
    print(f"\nFirst 5 windows:")
    print(f"{'Window':<8} {'Logit':<12} {'Probability':<12} {'After Sigmoid':<15}")
    print("-" * 60)

    for i in range(5):
        center = 50 + i * 10
        start_col = center - 50
        end_col = center + 50
        window = spec_norm[:, start_col:end_col]

        # Prepare window same as _prepare_batch
        import matplotlib.pyplot as plt
        from PIL import Image

        cmap = plt.get_cmap('magma')
        rgb = cmap(window)[:, :, :3]
        grayscale = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

        # Resize to 256
        grayscale_uint8 = (grayscale * 255).astype(np.uint8)
        img = Image.fromarray(grayscale_uint8)
        img_resized = img.resize((grayscale.shape[1], 256), Image.LANCZOS)
        grayscale = np.array(img_resized, dtype=np.float32) / 255.0

        # Per-image normalize
        g_min, g_max = grayscale.min(), grayscale.max()
        if g_max > g_min:
            grayscale = (grayscale - g_min) / (g_max - g_min)

        # Convert to tensor
        tensor = torch.from_numpy(grayscale).unsqueeze(0).unsqueeze(0).float()
        tensor = tensor.to(inference.device)

        # Get logit and probability
        with torch.no_grad():
            logit = inference.model.forward(tensor).cpu().numpy()[0, 0]
            prob = inference.model.predict_proba(tensor).cpu().numpy()[0, 0]

        # Manual sigmoid for verification
        manual_prob = 1.0 / (1.0 + np.exp(-logit))

        print(f"{i:<8} {logit:<12.4f} {prob:<12.6f} {manual_prob:<15.6f}")

    # Now run full inference
    result = inference.infer(audio_data.spectrogram_db, audio_data.times)

    print(f"\nFull inference statistics:")
    print(f"  Probability range: [{result.probabilities.min():.6f}, {result.probabilities.max():.6f}]")
    print(f"  Mean: {result.probabilities.mean():.4f}")
    print(f"  Median: {np.median(result.probabilities):.4f}")
    print(f"  Std: {result.probabilities.std():.4f}")
    print(f"  Values > 0.99: {(result.probabilities > 0.99).sum()}/{len(result.probabilities)} ({100*(result.probabilities > 0.99).sum()/len(result.probabilities):.1f}%)")

print(f"\n{'='*60}")
print("DIAGNOSIS")
print(f"{'='*60}")
print("""
If checkpoints model shows:
- Very high logits (e.g., > 5.0): Model is too confident
- All probabilities > 0.99: Model saturated (outputs extreme values)

This could indicate:
1. Model was trained with different preprocessing
2. Model architecture mismatch
3. Different training data distribution
4. Overfitting to training set
""")
