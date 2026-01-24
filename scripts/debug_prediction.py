"""Debug prediction discrepancy."""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint
from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
from torch.utils.data import DataLoader

# Load model
model, _ = load_model_checkpoint(
    Path('models/clean_test/best_model.pt'),
    USVClassifierCNN,
    device='cpu'
)
model.eval()

# Test sample
spec_path = Path('spectrograms_training/2024-09-30_11-19-09_0000008_00001861.png')

print("=" * 80)
print("METHOD 1: Direct image loading (like predict.py)")
print("=" * 80)

img = Image.open(spec_path).convert('L')
spectrogram = np.array(img, dtype=np.float32)
spec_min = spectrogram.min()
spec_max = spectrogram.max()
if spec_max > spec_min:
    spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
spectrogram_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    prob1 = model.predict_proba(spectrogram_tensor).item()

print(f"Probability: {prob1:.4f}")
print(f"Image shape: {spectrogram_tensor.shape}")
print(f"Min/Max: {spectrogram_tensor.min():.4f} / {spectrogram_tensor.max():.4f}")

print()
print("=" * 80)
print("METHOD 2: DataLoader (like threshold_sweep.py)")
print("=" * 80)

# Create minimal dataset with just this one sample
import pandas as pd
test_df = pd.read_csv('splits/test.csv')
single_sample_df = test_df[test_df['spectrogram_path'].str.contains('00001861')].head(1)
single_csv = Path('temp_single.csv')
single_sample_df.to_csv(single_csv, index=False)

dataset = USVDataset(single_csv)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=pad_collate_fn)

for spec_batch, label_batch in dataloader:
    with torch.no_grad():
        prob2 = model.predict_proba(spec_batch).item()
    print(f"Probability: {prob2:.4f}")
    print(f"Batch shape: {spec_batch.shape}")
    print(f"Min/Max: {spec_batch.min():.4f} / {spec_batch.max():.4f}")

single_csv.unlink()

print()
print("=" * 80)
print(f"DIFFERENCE: {abs(prob1 - prob2):.4f}")
print("=" * 80)
