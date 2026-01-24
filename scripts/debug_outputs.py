"""Debug model outputs."""
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint
from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
from torch.utils.data import DataLoader

model, _ = load_model_checkpoint(
    Path('models/clean_test/best_model.pt'),
    USVClassifierCNN,
    device='cpu'
)
model.eval()

print("=" * 80)
print("METHOD 1: DataLoader (batch)")
print("=" * 80)

dataset = USVDataset('splits/test.csv')
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=pad_collate_fn, pin_memory=False)

for spec_batch, label_batch in dataloader:
    with torch.no_grad():
        outputs = model(spec_batch)
        proba_batch = torch.sigmoid(outputs)

    print(f'Batch size: {spec_batch.shape[0]}')
    print(f'Batch shape: {spec_batch.shape}')
    print(f'First sample shape in batch: {spec_batch[0].shape}')
    print(f'Outputs (first 5): {outputs.squeeze()[:5]}')
    print(f'Probabilities (first 5): {proba_batch.squeeze()[:5]}')
    break

print()
print("=" * 80)
print("METHOD 2: Direct image loading")
print("=" * 80)

img = Image.open('spectrograms_training/2024-09-30_11-19-09_0000008_00001861.png').convert('L')
spec = np.array(img, dtype=np.float32)
spec = (spec - spec.min()) / (spec.max() - spec.min())
spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)

print(f'Image shape: {spec_tensor.shape}')

with torch.no_grad():
    logit = model(spec_tensor)
    prob = torch.sigmoid(logit)

print(f'Logit: {logit.item():.4f}')
print(f'Probability: {prob.item():.6f}')

print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f'DataLoader first sample logit: {outputs[0].item():.4f}')
print(f'Direct loading logit: {logit.item():.4f}')
print(f'Difference: {abs(outputs[0].item() - logit.item()):.4f}')
