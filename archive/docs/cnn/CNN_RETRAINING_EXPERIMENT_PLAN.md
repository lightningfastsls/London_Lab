# Quick Experiment: Testing Random Negatives for CNN Batch Detection

## Goal

Test whether adding random audio chunks as negative samples will fix the CNN's batch detection problem. Currently the CNN thinks everything is a USV (0.997 mean probability on random chunks) because it was only trained on energy-detector candidates.

**Hypothesis:** If we add random audio chunks as negatives, the CNN will learn what "no USV" looks like and random chunk probability will drop significantly.

**Success Criteria:** Random chunk probability drops from 0.997 to <0.5 after training with new negatives.

If this works → proceed to full retraining with comprehensive negatives.
If this fails → investigate why before investing more time.

---

## Phase 1: Generate Random Negative Samples

### Task 1.1: Create negative sample generator

**File:** `scripts/generate_random_negatives.py`

```python
"""
Generate random negative samples for CNN retraining experiment.

Extracts spectrogram chunks from random positions in WAV files,
ensuring they don't overlap with known USV regions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io.wavfile as wav
import scipy.signal as signal
from PIL import Image
import argparse
from tqdm import tqdm
import random


def load_usv_regions(labels_csv: Path) -> dict:
    """
    Load labeled USV regions to avoid.
    
    Returns dict: {source_file: [(start_ms, end_ms), ...]}
    """
    df = pd.read_csv(labels_csv)
    df_usv = df[df['label'] == 'usv']
    
    regions = {}
    for _, row in df_usv.iterrows():
        source = row['source_file']
        if source not in regions:
            regions[source] = []
        # Add buffer around USV (50ms each side)
        start = max(0, row['start_ms'] - 50)
        end = row['end_ms'] + 50
        regions[source].append((start, end))
    
    return regions


def overlaps_usv(start_ms: float, end_ms: float, usv_regions: list) -> bool:
    """Check if a time range overlaps with any USV region."""
    for usv_start, usv_end in usv_regions:
        if start_ms < usv_end and end_ms > usv_start:
            return True
    return False


def extract_spectrogram_chunk(
    audio: np.ndarray,
    sample_rate: int,
    start_ms: float,
    duration_ms: float = 40,
    n_fft: int = 512,
    hop_length: int = 128,
    freq_min_hz: int = 20000,
    freq_max_hz: int = 120000
) -> np.ndarray:
    """
    Extract a spectrogram chunk from audio.
    
    Returns normalized spectrogram as 2D numpy array.
    """
    # Convert ms to samples
    start_sample = int(start_ms * sample_rate / 1000)
    duration_samples = int(duration_ms * sample_rate / 1000)
    end_sample = start_sample + duration_samples
    
    # Extract audio chunk
    chunk = audio[start_sample:end_sample]
    
    if len(chunk) < duration_samples:
        return None
    
    # Compute spectrogram
    frequencies, times, spec = signal.spectrogram(
        chunk,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        scaling='spectrum'
    )
    
    # Convert to dB
    spec_db = 10 * np.log10(spec + 1e-10)
    
    # Crop to frequency range
    freq_mask = (frequencies >= freq_min_hz) & (frequencies <= freq_max_hz)
    spec_db = spec_db[freq_mask, :]
    
    # Normalize using dynamic range (same as training)
    vmin = np.mean(spec_db) - 2 * np.std(spec_db)
    vmax = np.mean(spec_db) + 3 * np.std(spec_db)
    spec_db = np.clip(spec_db, vmin, vmax)
    spec_db = ((spec_db - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    return spec_db


def generate_random_negatives(
    wav_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    n_samples: int = 100,
    duration_ms: float = 40,
    seed: int = 42
):
    """
    Generate random negative samples from WAV files.
    
    Samples random positions that don't overlap with labeled USVs.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load USV regions to avoid
    usv_regions = load_usv_regions(labels_csv)
    
    # Get all WAV files
    wav_files = list(Path(wav_dir).glob('*.wav'))
    if not wav_files:
        # Try looking in subdirectories
        wav_files = list(Path(wav_dir).glob('**/*.wav'))
    
    print(f"Found {len(wav_files)} WAV files")
    print(f"Generating {n_samples} random negative samples...")
    
    samples_per_file = n_samples // len(wav_files) + 1
    
    metadata = []
    sample_count = 0
    
    for wav_path in tqdm(wav_files, desc="Processing WAV files"):
        # Load audio
        try:
            sample_rate, audio = wav.read(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        duration_total_ms = len(audio) / sample_rate * 1000
        
        # Get USV regions for this file
        file_key = wav_path.name
        file_usv_regions = usv_regions.get(file_key, [])
        
        # Generate random positions
        attempts = 0
        file_samples = 0
        
        while file_samples < samples_per_file and attempts < samples_per_file * 10:
            attempts += 1
            
            # Random start position
            start_ms = random.uniform(0, duration_total_ms - duration_ms)
            end_ms = start_ms + duration_ms
            
            # Check if overlaps with USV
            if overlaps_usv(start_ms, end_ms, file_usv_regions):
                continue
            
            # Extract spectrogram
            spec = extract_spectrogram_chunk(audio, sample_rate, start_ms, duration_ms)
            
            if spec is None:
                continue
            
            # Save spectrogram
            filename = f"random_neg_{sample_count:05d}.png"
            save_path = output_dir / filename
            Image.fromarray(spec).save(save_path)
            
            # Record metadata
            metadata.append({
                'candidate_id': f'random_neg_{sample_count:05d}',
                'spectrogram_path': filename,
                'source_file': wav_path.name,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'label': 'noise',
                'sample_type': 'random_negative'
            })
            
            sample_count += 1
            file_samples += 1
            
            if sample_count >= n_samples:
                break
        
        if sample_count >= n_samples:
            break
    
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / 'random_negatives_metadata.csv', index=False)
    
    print(f"\nGenerated {sample_count} random negative samples")
    print(f"Saved to {output_dir}")
    print(f"Metadata saved to {output_dir / 'random_negatives_metadata.csv'}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate random negative samples')
    parser.add_argument('--wav-dir', type=Path, required=True, 
                        help='Directory containing WAV files')
    parser.add_argument('--labels-csv', type=Path, required=True,
                        help='CSV with labeled USV data (to avoid)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for spectrograms')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--duration-ms', type=float, default=40,
                        help='Duration of each chunk in ms')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generate_random_negatives(
        args.wav_dir,
        args.labels_csv,
        args.output_dir,
        args.n_samples,
        args.duration_ms,
        args.seed
    )


if __name__ == '__main__':
    main()
```

### Task 1.2: Run the generator

```powershell
# Generate 100 random negative samples
python scripts/generate_random_negatives.py `
    --wav-dir data/raw/recordings `
    --labels-csv splits/all_labeled.csv `
    --output-dir data/experiment_negatives `
    --n-samples 100 `
    --seed 42
```

---

## Phase 2: Create Experiment Dataset

### Task 2.1: Combine existing data with new negatives

**File:** `scripts/create_experiment_dataset.py`

```python
"""
Create experiment dataset combining existing labeled data with new random negatives.
"""

import pandas as pd
from pathlib import Path
import shutil
import argparse


def create_experiment_dataset(
    original_train_csv: Path,
    original_val_csv: Path,
    random_negatives_csv: Path,
    random_negatives_dir: Path,
    original_spectrograms_dir: Path,
    output_dir: Path
):
    """
    Create experiment dataset:
    - Keep all original training data
    - Add random negatives to training set
    - Keep validation set unchanged (for fair comparison)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    train_df = pd.read_csv(original_train_csv)
    val_df = pd.read_csv(original_val_csv)
    random_neg_df = pd.read_csv(random_negatives_csv)
    
    print(f"Original training samples: {len(train_df)}")
    print(f"  - USV: {(train_df['label'] == 'usv').sum()}")
    print(f"  - Not USV: {(train_df['label'] != 'usv').sum()}")
    print(f"Random negatives to add: {len(random_neg_df)}")
    
    # Copy random negative spectrograms to experiment directory
    exp_spectrograms_dir = output_dir / 'spectrograms'
    exp_spectrograms_dir.mkdir(exist_ok=True)
    
    # Copy original spectrograms (or create symlinks)
    print("\nCopying spectrograms...")
    
    # Update random negatives paths and copy files
    for idx, row in random_neg_df.iterrows():
        src = random_negatives_dir / row['spectrogram_path']
        dst = exp_spectrograms_dir / row['spectrogram_path']
        if src.exists():
            shutil.copy(src, dst)
    
    # Copy original spectrograms referenced in train
    for idx, row in train_df.iterrows():
        src = original_spectrograms_dir / row['spectrogram_path']
        dst = exp_spectrograms_dir / row['spectrogram_path']
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
    
    # Copy original spectrograms referenced in val
    for idx, row in val_df.iterrows():
        src = original_spectrograms_dir / row['spectrogram_path']
        dst = exp_spectrograms_dir / row['spectrogram_path']
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
    
    # Standardize random negatives dataframe columns to match training data
    random_neg_df['label'] = 'noise'  # Ensure label is 'noise' not 'not_usv'
    
    # Make sure columns match
    required_cols = ['candidate_id', 'spectrogram_path', 'source_file', 'label']
    for col in required_cols:
        if col not in random_neg_df.columns:
            print(f"Warning: {col} not in random negatives, adding placeholder")
            random_neg_df[col] = 'unknown'
    
    # Combine training data with random negatives
    # Only keep columns that exist in both
    common_cols = list(set(train_df.columns) & set(random_neg_df.columns))
    
    train_combined = pd.concat([
        train_df[common_cols],
        random_neg_df[common_cols]
    ], ignore_index=True)
    
    # Shuffle training data
    train_combined = train_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save experiment datasets
    train_combined.to_csv(output_dir / 'train_experiment.csv', index=False)
    val_df.to_csv(output_dir / 'val_experiment.csv', index=False)
    
    print(f"\nExperiment dataset created:")
    print(f"  Training: {len(train_combined)} samples")
    print(f"    - USV: {(train_combined['label'] == 'usv').sum()}")
    print(f"    - Not USV: {(train_combined['label'] != 'usv').sum()}")
    print(f"  Validation: {len(val_df)} samples (unchanged)")
    print(f"\nSaved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create experiment dataset')
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    parser.add_argument('--random-negatives-csv', type=Path, required=True)
    parser.add_argument('--random-negatives-dir', type=Path, required=True)
    parser.add_argument('--spectrograms-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    
    args = parser.parse_args()
    
    create_experiment_dataset(
        args.train_csv,
        args.val_csv,
        args.random_negatives_csv,
        args.random_negatives_dir,
        args.spectrograms_dir,
        args.output_dir
    )


if __name__ == '__main__':
    main()
```

### Task 2.2: Run dataset creation

```powershell
# Create experiment dataset
python scripts/create_experiment_dataset.py `
    --train-csv splits/train.csv `
    --val-csv splits/val.csv `
    --random-negatives-csv data/experiment_negatives/random_negatives_metadata.csv `
    --random-negatives-dir data/experiment_negatives `
    --spectrograms-dir data/candidates/spectrograms `
    --output-dir data/experiment_dataset
```

---

## Phase 3: Train Experiment Model

### Task 3.1: Train with new dataset

```powershell
# Train experiment model (20 epochs is enough to see if it's working)
python scripts/train_cnn.py `
    --train-csv data/experiment_dataset/train_experiment.csv `
    --val-csv data/experiment_dataset/val_experiment.csv `
    --spectrogram-dir data/experiment_dataset/spectrograms `
    --output-dir models/experiment_random_negatives `
    --num-epochs 20 `
    --patience 10 `
    --use-class-weights
```

---

## Phase 4: Evaluate Experiment Results

### Task 4.1: Run same diagnostic as before

**File:** `scripts/evaluate_experiment.py`

```python
"""
Evaluate experiment model with same tests as original diagnostic.

Tests:
1. Labeled USV samples → should be high probability
2. Labeled "Not USV" samples → should be low probability  
3. Random chunks from WAV files → should be LOW probability (this is the key test!)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io.wavfile as wav
import scipy.signal as signal
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN


def load_model(model_path: Path, device: str = 'auto'):
    """Load trained model."""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = USVClassifierCNN()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def predict_spectrogram(model, spec: np.ndarray, device) -> float:
    """Get probability for a single spectrogram."""
    with torch.no_grad():
        # Normalize
        spec_norm = (spec - spec.mean()) / (spec.std() + 1e-8)
        
        # To tensor
        x = torch.from_numpy(spec_norm).float().unsqueeze(0).unsqueeze(0)
        x = x.to(device)
        
        # Predict
        prob = model(x).item()
        
    return prob


def test_labeled_samples(model, device, csv_path: Path, spec_dir: Path, label_filter: str = None):
    """Test on labeled samples."""
    df = pd.read_csv(csv_path)
    
    if label_filter:
        if label_filter == 'usv':
            df = df[df['label'] == 'usv']
        else:
            df = df[df['label'] != 'usv']
    
    probs = []
    for _, row in df.iterrows():
        spec_path = spec_dir / row['spectrogram_path']
        if spec_path.exists():
            spec = np.array(Image.open(spec_path).convert('L')).astype(np.float32)
            prob = predict_spectrogram(model, spec, device)
            probs.append(prob)
    
    return np.array(probs)


def test_random_chunks(model, device, wav_dir: Path, n_samples: int = 100, duration_ms: float = 40):
    """Test on random chunks from WAV files."""
    import random
    
    wav_files = list(Path(wav_dir).glob('*.wav'))
    if not wav_files:
        wav_files = list(Path(wav_dir).glob('**/*.wav'))
    
    probs = []
    samples_per_file = n_samples // len(wav_files) + 1
    
    for wav_path in wav_files:
        try:
            sample_rate, audio = wav.read(wav_path)
        except:
            continue
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        duration_total_ms = len(audio) / sample_rate * 1000
        
        for _ in range(samples_per_file):
            if len(probs) >= n_samples:
                break
            
            start_ms = random.uniform(0, duration_total_ms - duration_ms)
            
            # Extract spectrogram (same as training)
            start_sample = int(start_ms * sample_rate / 1000)
            duration_samples = int(duration_ms * sample_rate / 1000)
            chunk = audio[start_sample:start_sample + duration_samples]
            
            if len(chunk) < duration_samples:
                continue
            
            frequencies, times, spec = signal.spectrogram(
                chunk, fs=sample_rate, nperseg=512, noverlap=512-128, scaling='spectrum'
            )
            spec_db = 10 * np.log10(spec + 1e-10)
            
            freq_mask = (frequencies >= 20000) & (frequencies <= 120000)
            spec_db = spec_db[freq_mask, :]
            
            vmin = np.mean(spec_db) - 2 * np.std(spec_db)
            vmax = np.mean(spec_db) + 3 * np.std(spec_db)
            spec_db = np.clip(spec_db, vmin, vmax)
            spec_norm = (spec_db - vmin) / (vmax - vmin) * 255
            
            prob = predict_spectrogram(model, spec_norm.astype(np.float32), device)
            probs.append(prob)
        
        if len(probs) >= n_samples:
            break
    
    return np.array(probs)


def run_evaluation(
    model_path: Path,
    test_csv: Path,
    spec_dir: Path,
    wav_dir: Path,
    output_dir: Path
):
    """Run full evaluation and comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model, device = load_model(model_path)
    
    print("\nTest 1: Labeled USV samples...")
    usv_probs = test_labeled_samples(model, device, test_csv, spec_dir, label_filter='usv')
    print(f"  N = {len(usv_probs)}")
    print(f"  Mean probability: {usv_probs.mean():.3f}")
    print(f"  Above 0.5: {(usv_probs > 0.5).mean()*100:.1f}%")
    print(f"  Above 0.9: {(usv_probs > 0.9).mean()*100:.1f}%")
    
    print("\nTest 2: Labeled 'Not USV' samples...")
    not_usv_probs = test_labeled_samples(model, device, test_csv, spec_dir, label_filter='not_usv')
    print(f"  N = {len(not_usv_probs)}")
    print(f"  Mean probability: {not_usv_probs.mean():.3f}")
    print(f"  Above 0.5: {(not_usv_probs > 0.5).mean()*100:.1f}%")
    print(f"  Above 0.9: {(not_usv_probs > 0.9).mean()*100:.1f}%")
    
    print("\nTest 3: Random chunks from WAV files...")
    random_probs = test_random_chunks(model, device, wav_dir, n_samples=100)
    print(f"  N = {len(random_probs)}")
    print(f"  Mean probability: {random_probs.mean():.3f}")
    print(f"  Above 0.5: {(random_probs > 0.5).mean()*100:.1f}%")
    print(f"  Above 0.9: {(random_probs > 0.9).mean()*100:.1f}%")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(usv_probs, bins=20, range=(0, 1), color='green', alpha=0.7)
    axes[0].axvline(x=0.5, color='red', linestyle='--')
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Labeled USV (n={len(usv_probs)})\nMean: {usv_probs.mean():.3f}')
    
    axes[1].hist(not_usv_probs, bins=20, range=(0, 1), color='orange', alpha=0.7)
    axes[1].axvline(x=0.5, color='red', linestyle='--')
    axes[1].set_xlabel('Probability')
    axes[1].set_title(f'Labeled Not USV (n={len(not_usv_probs)})\nMean: {not_usv_probs.mean():.3f}')
    
    axes[2].hist(random_probs, bins=20, range=(0, 1), color='blue', alpha=0.7)
    axes[2].axvline(x=0.5, color='red', linestyle='--')
    axes[2].set_xlabel('Probability')
    axes[2].set_title(f'Random Chunks (n={len(random_probs)})\nMean: {random_probs.mean():.3f}')
    
    plt.suptitle('Experiment Model: Probability Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_evaluation.png', dpi=150)
    plt.close()
    
    # Save results
    results = {
        'usv_mean': usv_probs.mean(),
        'usv_above_0.5': (usv_probs > 0.5).mean(),
        'not_usv_mean': not_usv_probs.mean(),
        'not_usv_above_0.5': (not_usv_probs > 0.5).mean(),
        'random_mean': random_probs.mean(),
        'random_above_0.5': (random_probs > 0.5).mean(),
    }
    
    with open(output_dir / 'experiment_results.txt', 'w') as f:
        f.write("EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("SUCCESS CRITERIA:\n")
        f.write("  - Random chunk mean probability < 0.5\n")
        f.write("  - USV samples still recognized (mean > 0.8)\n\n")
        f.write("RESULTS:\n")
        for k, v in results.items():
            f.write(f"  {k}: {v:.3f}\n")
        f.write("\n")
        
        if results['random_mean'] < 0.5 and results['usv_mean'] > 0.8:
            f.write("VERDICT: SUCCESS! Proceed to full retraining.\n")
        elif results['random_mean'] < 0.7:
            f.write("VERDICT: PARTIAL SUCCESS. Random prob reduced but not enough.\n")
            f.write("         Try adding more negative samples in full retraining.\n")
        else:
            f.write("VERDICT: FAILED. Random negatives didn't help.\n")
            f.write("         Need to investigate further.\n")
    
    print(f"\nResults saved to {output_dir}")
    print(f"See {output_dir / 'experiment_evaluation.png'} for visualization")
    
    # Print verdict
    print("\n" + "=" * 50)
    if results['random_mean'] < 0.5 and results['usv_mean'] > 0.8:
        print("VERDICT: SUCCESS! ✓")
        print("Random chunk probability dropped significantly.")
        print("Proceed to full retraining with comprehensive negatives.")
    elif results['random_mean'] < 0.7:
        print("VERDICT: PARTIAL SUCCESS")
        print(f"Random prob reduced from 0.997 to {results['random_mean']:.3f}")
        print("Full retraining with more negatives should work.")
    else:
        print("VERDICT: NEEDS INVESTIGATION")
        print("Random negatives didn't reduce probability enough.")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Evaluate experiment model')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--spec-dir', type=Path, required=True)
    parser.add_argument('--wav-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/experiment_evaluation'))
    
    args = parser.parse_args()
    
    run_evaluation(
        args.model,
        args.test_csv,
        args.spec_dir,
        args.wav_dir,
        args.output_dir
    )


if __name__ == '__main__':
    main()
```

### Task 4.2: Run evaluation

```powershell
# Evaluate experiment model
python scripts/evaluate_experiment.py `
    --model models/experiment_random_negatives/best_model.pt `
    --test-csv splits/test.csv `
    --spec-dir data/candidates/spectrograms `
    --wav-dir data/raw/recordings `
    --output-dir analysis/experiment_evaluation
```

---

## Expected Results

### Success Criteria

| Test | Before (Original) | After (Experiment) | Goal |
|------|-------------------|-------------------|------|
| Labeled USVs | 0.992 | >0.80 | Still recognizes USVs |
| Labeled "Not USV" | 0.684 | <0.50 | Lower than before |
| Random chunks | 0.997 | <0.50 | **KEY: Much lower** |

### Interpreting Results

**If SUCCESS (random < 0.5, USV > 0.8):**
- Approach works! 
- Proceed to full retraining with 1000+ comprehensive negatives
- Create script for full negative sample generation

**If PARTIAL SUCCESS (random < 0.7 but not < 0.5):**
- Approach is working, just needs more negatives
- Proceed to full retraining with even more negatives
- Consider adding different types (inter-USV gaps, low-energy regions)

**If FAILED (random still > 0.8):**
- Need to investigate why
- Check if spectrograms are being generated the same way
- May need different approach

---

## Execution Summary

Run these commands in order:

```powershell
# Phase 1: Generate random negatives
python scripts/generate_random_negatives.py `
    --wav-dir data/raw/recordings `
    --labels-csv splits/all_labeled.csv `
    --output-dir data/experiment_negatives `
    --n-samples 100

# Phase 2: Create experiment dataset
python scripts/create_experiment_dataset.py `
    --train-csv splits/train.csv `
    --val-csv splits/val.csv `
    --random-negatives-csv data/experiment_negatives/random_negatives_metadata.csv `
    --random-negatives-dir data/experiment_negatives `
    --spectrograms-dir data/candidates/spectrograms `
    --output-dir data/experiment_dataset

# Phase 3: Train experiment model
python scripts/train_cnn.py `
    --train-csv data/experiment_dataset/train_experiment.csv `
    --val-csv data/experiment_dataset/val_experiment.csv `
    --spectrogram-dir data/experiment_dataset/spectrograms `
    --output-dir models/experiment_random_negatives `
    --num-epochs 20 `
    --patience 10 `
    --use-class-weights

# Phase 4: Evaluate
python scripts/evaluate_experiment.py `
    --model models/experiment_random_negatives/best_model.pt `
    --test-csv splits/test.csv `
    --spec-dir data/candidates/spectrograms `
    --wav-dir data/raw/recordings `
    --output-dir analysis/experiment_evaluation
```

---

## Next Steps Based on Results

**If experiment succeeds**, create a follow-up task:

> "The experiment worked! Now create a comprehensive negative sample generator that produces:
> - 500 random position negatives
> - 300 inter-USV gap negatives  
> - 200 low-energy region negatives
> 
> Then retrain the full model with this expanded dataset."

**If experiment fails**, investigate:

> "The experiment didn't work. Please investigate:
> 1. Are the random negative spectrograms being normalized the same way as training data?
> 2. What do the random negatives look like visually compared to training negatives?
> 3. Print some statistics comparing training spectrograms vs random negative spectrograms."
