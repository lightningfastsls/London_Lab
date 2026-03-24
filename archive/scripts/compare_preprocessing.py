"""Compare preprocessing pipeline step-by-step."""
import numpy as np
from pathlib import Path

print("="*70)
print("TRAINING EXTRACTION (_render_training in spectrogram_extractor.py)")
print("="*70)
print("""
1. spec_db in dB (raw)
2. Compute vmin, vmax using MAD
3. CLIP spec_db to [vmin, vmax]
4. Normalize: normalized = (clipped - vmin) / (vmax - vmin + 1e-12)  → [0, 1]
5. Apply magma colormap: rgb = cmap(normalized)  → RGB in [0, 1]
6. FLIP VERTICALLY: rgb_flipped = np.flipud(rgb)
7. Convert to uint8: rgb_uint8 = (rgb_flipped * 255).astype(np.uint8)
8. RESIZE to (target_width, 256) using PIL LANCZOS
9. Save as PNG (RGB, uint8)

Then DATA LOADER (data_loader.py):
1. Load PNG (RGB, uint8 [0-255])
2. Convert to grayscale: img.convert('L')  → grayscale uint8 [0-255]
3. Convert to float32: np.array(img, dtype=np.float32)
4. Per-image normalize: (gray - min) / (max - min)  → [0, 1]
5. Add channel dim, feed to CNN
""")

print("="*70)
print("CURRENT SLIDING INFERENCE (_prepare_batch)")
print("="*70)
print("""
In infer():
1. spec_db in dB (raw)
2. Compute vmin, vmax using MAD
3. CLIP spec_db to [vmin, vmax]
4. Normalize: normalized = (clipped - vmin) / (vmax - vmin + 1e-12)  → [0, 1]
5. FLIP VERTICALLY: spec_norm = np.flipud(normalized)  ← FLIPPED BEFORE COLORMAP!
6. Extract 100-column windows

In _prepare_batch():
7. Apply magma colormap: rgb = cmap(window)  → RGB in [0, 1]
8. Convert to grayscale: gray = 0.299*R + 0.587*G + 0.114*B  → [0, 1]
9. Convert to uint8: gray_uint8 = (gray * 255).astype(np.uint8)
10. RESIZE to (width, 256) using PIL LANCZOS
11. Convert back to float: gray = np.array(img_resized, dtype=np.float32) / 255.0
12. Per-image normalize: (gray - min) / (max - min)  → [0, 1]
13. Add channel dim, feed to CNN
""")

print("="*70)
print("IDENTIFIED MISMATCH!")
print("="*70)
print("""
DIFFERENCE: Order of FLIP vs COLORMAP

Training:     Normalize → Colormap → FLIP → Resize → Save → Load → Grayscale
Inference:    Normalize → FLIP → Extract → Colormap → Grayscale → Resize

The FLIP happens at different stages!

Training:  Flips the RGB image (after colormap)
Inference: Flips the normalized spectrogram (before colormap)

This could cause the frequency axis to be in opposite directions!

ALSO: Quantization difference
Training:  RGB [0,1] → uint8 [0,255] → save → load → grayscale uint8
Inference: RGB [0,1] → grayscale float → uint8 → resize → float

The uint8 quantization happens at different points.
""")

print("\nFIX NEEDED:")
print("1. Move vertical flip to AFTER colormap (in _prepare_batch)")
print("2. Match the uint8 quantization path exactly")
