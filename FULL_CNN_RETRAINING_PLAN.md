# Full CNN Retraining Plan: Diverse Negatives for Batch Detection

## Background

The experiment confirmed that adding random negatives fixes batch detection:
- Random chunk probability dropped from 0.997 → 0.000 ✓
- But USV recognition dropped from 0.992 → 0.624 ✗

This retraining plan addresses both issues by:
1. Adding more diverse negative samples
2. Using stronger class weights to protect USV recall
3. Training longer with proper monitoring
4. Optimizing the classification threshold

---

## Target Outcomes

| Test | Current | Target |
|------|---------|--------|
| USV samples | 0.624 | >0.85 |
| Not USV samples | 0.120 | <0.30 |
| Random chunks | 0.000 | <0.20 |
| Batch detection viable | No | Yes |

---

## Phase 1: Generate Comprehensive Negative Samples

### 1.1 Create Negative Sample Generator

**File:** `scripts/generate_comprehensive_negatives.py`

```python
"""
Generate comprehensive negative samples for CNN retraining.

Creates three types of negatives:
1. Random positions - arbitrary chunks from anywhere in recordings
2. Inter-USV gaps - silence between known USVs
3. Low-energy regions - quiet parts with minimal acoustic activity

All samples avoid overlap with labeled USV regions (with 50ms buffer).
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
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class NegativeSample:
    """A negative sample with metadata."""
    source_file: str
    start_ms: float
    end_ms: float
    sample_type: str  # 'random', 'inter_usv_gap', 'low_energy'
    spectrogram: np.ndarray


class ComprehensiveNegativeGenerator:
    """Generate diverse negative samples from WAV files."""
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        freq_min_hz: int = 20000,
        freq_max_hz: int = 120000,
        duration_ms: float = 40,
        buffer_ms: float = 50,
        seed: int = 42
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_min_hz = freq_min_hz
        self.freq_max_hz = freq_max_hz
        self.duration_ms = duration_ms
        self.buffer_ms = buffer_ms
        
        random.seed(seed)
        np.random.seed(seed)
    
    def load_usv_regions(self, labels_csv: Path) -> dict:
        """
        Load labeled USV regions to avoid.
        
        Returns dict: {source_file: [(start_ms, end_ms), ...]}
        """
        df = pd.read_csv(labels_csv)
        
        # Handle different possible label column names
        if 'label' in df.columns:
            df_usv = df[df['label'].str.lower().isin(['usv', 'yes', '1', 'true'])]
        else:
            # Assume all entries are USVs if no label column
            df_usv = df
        
        regions = {}
        for _, row in df_usv.iterrows():
            source = row['source_file']
            if source not in regions:
                regions[source] = []
            
            # Add buffer around USV
            start = max(0, row['start_ms'] - self.buffer_ms)
            end = row['end_ms'] + self.buffer_ms
            regions[source].append((start, end))
        
        # Sort regions by start time for each file
        for source in regions:
            regions[source] = sorted(regions[source], key=lambda x: x[0])
        
        return regions
    
    def overlaps_usv(self, start_ms: float, end_ms: float, usv_regions: List[Tuple]) -> bool:
        """Check if a time range overlaps with any USV region."""
        for usv_start, usv_end in usv_regions:
            if start_ms < usv_end and end_ms > usv_start:
                return True
        return False
    
    def extract_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_ms: float
    ) -> Optional[np.ndarray]:
        """Extract a spectrogram chunk from audio."""
        start_sample = int(start_ms * sample_rate / 1000)
        duration_samples = int(self.duration_ms * sample_rate / 1000)
        end_sample = start_sample + duration_samples
        
        if end_sample > len(audio):
            return None
        
        chunk = audio[start_sample:end_sample]
        
        if len(chunk) < duration_samples:
            return None
        
        # Compute spectrogram
        frequencies, times, spec = signal.spectrogram(
            chunk,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            scaling='spectrum'
        )
        
        # Convert to dB
        spec_db = 10 * np.log10(spec + 1e-10)
        
        # Crop to frequency range
        freq_mask = (frequencies >= self.freq_min_hz) & (frequencies <= self.freq_max_hz)
        spec_db = spec_db[freq_mask, :]
        
        if spec_db.size == 0:
            return None
        
        # Normalize using dynamic range
        vmin = np.mean(spec_db) - 2 * np.std(spec_db)
        vmax = np.mean(spec_db) + 3 * np.std(spec_db)
        spec_db = np.clip(spec_db, vmin, vmax)
        spec_db = ((spec_db - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
        
        return spec_db
    
    def compute_energy(self, audio: np.ndarray, sample_rate: int, start_ms: float) -> float:
        """Compute energy in USV frequency band for a chunk."""
        start_sample = int(start_ms * sample_rate / 1000)
        duration_samples = int(self.duration_ms * sample_rate / 1000)
        chunk = audio[start_sample:start_sample + duration_samples]
        
        if len(chunk) < duration_samples:
            return float('inf')
        
        # Bandpass filter to USV range
        nyquist = sample_rate / 2
        low = self.freq_min_hz / nyquist
        high = min(self.freq_max_hz / nyquist, 0.99)
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, chunk)
            energy = np.sqrt(np.mean(filtered ** 2))
        except:
            energy = np.sqrt(np.mean(chunk ** 2))
        
        return energy
    
    def generate_random_negatives(
        self,
        wav_files: List[Path],
        usv_regions: dict,
        n_samples: int,
        progress_callback=None
    ) -> List[NegativeSample]:
        """Generate random position negatives."""
        samples = []
        samples_per_file = max(1, n_samples // len(wav_files))
        
        for wav_path in wav_files:
            audio, sample_rate = self._load_audio(wav_path)
            if audio is None:
                continue
            
            duration_total_ms = len(audio) / sample_rate * 1000
            file_usv_regions = usv_regions.get(wav_path.name, [])
            
            file_samples = 0
            attempts = 0
            max_attempts = samples_per_file * 20
            
            while file_samples < samples_per_file and attempts < max_attempts:
                attempts += 1
                
                start_ms = random.uniform(0, duration_total_ms - self.duration_ms)
                end_ms = start_ms + self.duration_ms
                
                if self.overlaps_usv(start_ms, end_ms, file_usv_regions):
                    continue
                
                spec = self.extract_spectrogram(audio, sample_rate, start_ms)
                if spec is None:
                    continue
                
                samples.append(NegativeSample(
                    source_file=wav_path.name,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    sample_type='random',
                    spectrogram=spec
                ))
                
                file_samples += 1
                
                if progress_callback and len(samples) % 50 == 0:
                    progress_callback(len(samples), n_samples, 'random')
            
            if len(samples) >= n_samples:
                break
        
        return samples[:n_samples]
    
    def generate_inter_usv_gap_negatives(
        self,
        wav_files: List[Path],
        usv_regions: dict,
        n_samples: int,
        min_gap_ms: float = 100,
        progress_callback=None
    ) -> List[NegativeSample]:
        """Generate negatives from gaps between USVs."""
        samples = []
        
        # Find all gaps across all files
        all_gaps = []
        for wav_path in wav_files:
            file_regions = usv_regions.get(wav_path.name, [])
            if len(file_regions) < 2:
                continue
            
            for i in range(len(file_regions) - 1):
                gap_start = file_regions[i][1]  # End of current USV
                gap_end = file_regions[i + 1][0]  # Start of next USV
                gap_duration = gap_end - gap_start
                
                if gap_duration >= min_gap_ms:
                    all_gaps.append((wav_path, gap_start, gap_end))
        
        if not all_gaps:
            print("Warning: No inter-USV gaps found")
            return []
        
        # Sample from gaps
        random.shuffle(all_gaps)
        samples_per_gap = max(1, n_samples // len(all_gaps))
        
        for wav_path, gap_start, gap_end in all_gaps:
            audio, sample_rate = self._load_audio(wav_path)
            if audio is None:
                continue
            
            # Sample positions within this gap
            for _ in range(samples_per_gap):
                if len(samples) >= n_samples:
                    break
                
                # Random position within gap (with margin)
                margin = self.duration_ms / 2
                if gap_end - gap_start < self.duration_ms + 2 * margin:
                    start_ms = gap_start + margin
                else:
                    start_ms = random.uniform(gap_start + margin, gap_end - self.duration_ms - margin)
                
                spec = self.extract_spectrogram(audio, sample_rate, start_ms)
                if spec is None:
                    continue
                
                samples.append(NegativeSample(
                    source_file=wav_path.name,
                    start_ms=start_ms,
                    end_ms=start_ms + self.duration_ms,
                    sample_type='inter_usv_gap',
                    spectrogram=spec
                ))
                
                if progress_callback and len(samples) % 50 == 0:
                    progress_callback(len(samples), n_samples, 'inter_usv_gap')
            
            if len(samples) >= n_samples:
                break
        
        return samples[:n_samples]
    
    def generate_low_energy_negatives(
        self,
        wav_files: List[Path],
        usv_regions: dict,
        n_samples: int,
        energy_percentile: float = 20,
        progress_callback=None
    ) -> List[NegativeSample]:
        """Generate negatives from low-energy regions."""
        samples = []
        
        for wav_path in wav_files:
            audio, sample_rate = self._load_audio(wav_path)
            if audio is None:
                continue
            
            duration_total_ms = len(audio) / sample_rate * 1000
            file_usv_regions = usv_regions.get(wav_path.name, [])
            
            # Sample many positions and compute energy
            candidate_positions = []
            for _ in range(200):
                start_ms = random.uniform(0, duration_total_ms - self.duration_ms)
                if not self.overlaps_usv(start_ms, start_ms + self.duration_ms, file_usv_regions):
                    energy = self.compute_energy(audio, sample_rate, start_ms)
                    candidate_positions.append((start_ms, energy))
            
            if not candidate_positions:
                continue
            
            # Sort by energy and take lowest percentile
            candidate_positions.sort(key=lambda x: x[1])
            n_low_energy = max(1, int(len(candidate_positions) * energy_percentile / 100))
            low_energy_positions = candidate_positions[:n_low_energy]
            
            # Extract spectrograms
            for start_ms, _ in low_energy_positions:
                if len(samples) >= n_samples:
                    break
                
                spec = self.extract_spectrogram(audio, sample_rate, start_ms)
                if spec is None:
                    continue
                
                samples.append(NegativeSample(
                    source_file=wav_path.name,
                    start_ms=start_ms,
                    end_ms=start_ms + self.duration_ms,
                    sample_type='low_energy',
                    spectrogram=spec
                ))
                
                if progress_callback and len(samples) % 50 == 0:
                    progress_callback(len(samples), n_samples, 'low_energy')
            
            if len(samples) >= n_samples:
                break
        
        return samples[:n_samples]
    
    def _load_audio(self, wav_path: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file."""
        try:
            sample_rate, audio = wav.read(wav_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            return audio, sample_rate
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return None, None


def generate_all_negatives(
    wav_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    n_random: int = 500,
    n_inter_usv: int = 300,
    n_low_energy: int = 200,
    seed: int = 42
):
    """
    Generate all types of negative samples.
    
    Total: n_random + n_inter_usv + n_low_energy samples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ComprehensiveNegativeGenerator(seed=seed)
    
    # Load USV regions
    print("Loading USV regions...")
    usv_regions = generator.load_usv_regions(labels_csv)
    print(f"  Loaded regions for {len(usv_regions)} files")
    
    # Get WAV files
    wav_files = list(Path(wav_dir).glob('*.wav'))
    if not wav_files:
        wav_files = list(Path(wav_dir).glob('**/*.wav'))
    print(f"  Found {len(wav_files)} WAV files")
    
    def progress(current, total, sample_type):
        print(f"  Generated {current}/{total} {sample_type} samples...")
    
    all_samples = []
    
    # Generate random negatives
    print(f"\n1. Generating {n_random} random position negatives...")
    random_samples = generator.generate_random_negatives(
        wav_files, usv_regions, n_random, progress
    )
    all_samples.extend(random_samples)
    print(f"   Generated {len(random_samples)} random samples")
    
    # Generate inter-USV gap negatives
    print(f"\n2. Generating {n_inter_usv} inter-USV gap negatives...")
    gap_samples = generator.generate_inter_usv_gap_negatives(
        wav_files, usv_regions, n_inter_usv, progress_callback=progress
    )
    all_samples.extend(gap_samples)
    print(f"   Generated {len(gap_samples)} gap samples")
    
    # Generate low-energy negatives
    print(f"\n3. Generating {n_low_energy} low-energy negatives...")
    low_energy_samples = generator.generate_low_energy_negatives(
        wav_files, usv_regions, n_low_energy, progress_callback=progress
    )
    all_samples.extend(low_energy_samples)
    print(f"   Generated {len(low_energy_samples)} low-energy samples")
    
    # Save spectrograms and metadata
    print(f"\n4. Saving {len(all_samples)} spectrograms...")
    metadata = []
    
    for i, sample in enumerate(tqdm(all_samples, desc="Saving")):
        filename = f"neg_{sample.sample_type}_{i:05d}.png"
        save_path = output_dir / filename
        Image.fromarray(sample.spectrogram).save(save_path)
        
        metadata.append({
            'candidate_id': f'neg_{i:05d}',
            'spectrogram_path': filename,
            'source_file': sample.source_file,
            'start_ms': sample.start_ms,
            'end_ms': sample.end_ms,
            'label': 'noise',
            'sample_type': sample.sample_type
        })
    
    # Save metadata
    df = pd.DataFrame(metadata)
    metadata_path = output_dir / 'comprehensive_negatives_metadata.csv'
    df.to_csv(metadata_path, index=False)
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(all_samples)}")
    print(f"  - Random: {len(random_samples)}")
    print(f"  - Inter-USV gaps: {len(gap_samples)}")
    print(f"  - Low-energy: {len(low_energy_samples)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Metadata: {metadata_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive negative samples')
    parser.add_argument('--wav-dir', type=Path, required=True)
    parser.add_argument('--labels-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--n-random', type=int, default=500)
    parser.add_argument('--n-inter-usv', type=int, default=300)
    parser.add_argument('--n-low-energy', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    generate_all_negatives(
        args.wav_dir,
        args.labels_csv,
        args.output_dir,
        args.n_random,
        args.n_inter_usv,
        args.n_low_energy,
        args.seed
    )


if __name__ == '__main__':
    main()
```

### 1.2 Run Negative Generation

```powershell
# Generate comprehensive negative samples
# Total: 500 + 300 + 200 = 1000 negatives
python scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/comprehensive_negatives `
    --n-random 500 `
    --n-inter-usv 300 `
    --n-low-energy 200 `
    --seed 42
```

---

## Phase 2: Create Full Training Dataset

### 2.1 Dataset Creation Script

**File:** `scripts/create_full_training_dataset.py`

```python
"""
Create full training dataset with comprehensive negatives.

Combines:
- Original labeled USVs
- Original "Not USV" samples (energy-detector false positives)  
- New comprehensive negatives (random, inter-USV gaps, low-energy)

Splits by recording to prevent data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import argparse
from sklearn.model_selection import train_test_split


def create_full_dataset(
    original_labels_csv: Path,
    original_spectrograms_dir: Path,
    comprehensive_negatives_csv: Path,
    comprehensive_negatives_dir: Path,
    output_dir: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42
):
    """
    Create complete training dataset with proper splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    spectrograms_dir = output_dir / 'spectrograms'
    spectrograms_dir.mkdir(exist_ok=True)
    
    # Load original data
    print("Loading original labeled data...")
    orig_df = pd.read_csv(original_labels_csv)
    print(f"  Original samples: {len(orig_df)}")
    print(f"    USV: {(orig_df['label'] == 'usv').sum()}")
    print(f"    Not USV: {(orig_df['label'] != 'usv').sum()}")
    
    # Load comprehensive negatives
    print("\nLoading comprehensive negatives...")
    neg_df = pd.read_csv(comprehensive_negatives_csv)
    print(f"  Negative samples: {len(neg_df)}")
    for sample_type in neg_df['sample_type'].unique():
        count = (neg_df['sample_type'] == sample_type).sum()
        print(f"    {sample_type}: {count}")
    
    # Standardize columns
    neg_df['label'] = 'noise'
    
    # Ensure common columns
    common_cols = ['candidate_id', 'spectrogram_path', 'source_file', 'label']
    
    # Add missing columns with defaults
    for col in common_cols:
        if col not in orig_df.columns:
            orig_df[col] = 'unknown'
        if col not in neg_df.columns:
            neg_df[col] = 'unknown'
    
    # Copy spectrograms to output directory
    print("\nCopying spectrograms...")
    
    # Copy original spectrograms
    print("  Copying original spectrograms...")
    for _, row in orig_df.iterrows():
        src = original_spectrograms_dir / row['spectrogram_path']
        dst = spectrograms_dir / row['spectrogram_path']
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
    
    # Copy new negative spectrograms
    print("  Copying comprehensive negatives...")
    for _, row in neg_df.iterrows():
        src = comprehensive_negatives_dir / row['spectrogram_path']
        dst = spectrograms_dir / row['spectrogram_path']
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
    
    # Combine datasets
    print("\nCombining datasets...")
    combined_df = pd.concat([
        orig_df[common_cols + (['sample_type'] if 'sample_type' in orig_df.columns else [])],
        neg_df[common_cols + ['sample_type']]
    ], ignore_index=True)
    
    # Add sample_type for original data if missing
    if 'sample_type' not in orig_df.columns:
        combined_df.loc[combined_df['sample_type'].isna(), 'sample_type'] = 'original'
    
    print(f"  Combined total: {len(combined_df)}")
    
    # Split by recording (CRITICAL: prevents data leakage)
    print("\nSplitting by recording...")
    recordings = combined_df['source_file'].unique()
    print(f"  Unique recordings: {len(recordings)}")
    
    # First split: train vs (val + test)
    train_recordings, temp_recordings = train_test_split(
        recordings,
        test_size=test_size + val_size,
        random_state=seed
    )
    
    # Second split: val vs test
    val_recordings, test_recordings = train_test_split(
        temp_recordings,
        test_size=test_size / (test_size + val_size),
        random_state=seed
    )
    
    # Create splits
    train_df = combined_df[combined_df['source_file'].isin(train_recordings)].copy()
    val_df = combined_df[combined_df['source_file'].isin(val_recordings)].copy()
    test_df = combined_df[combined_df['source_file'].isin(test_recordings)].copy()
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save splits
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)
    
    def print_split_stats(name, df):
        n_usv = (df['label'] == 'usv').sum()
        n_not_usv = (df['label'] != 'usv').sum()
        print(f"\n{name}: {len(df)} samples")
        print(f"  USV: {n_usv} ({100*n_usv/len(df):.1f}%)")
        print(f"  Not USV: {n_not_usv} ({100*n_not_usv/len(df):.1f}%)")
        print(f"  Recordings: {df['source_file'].nunique()}")
        if 'sample_type' in df.columns:
            print(f"  By type:")
            for st in df['sample_type'].unique():
                print(f"    {st}: {(df['sample_type'] == st).sum()}")
    
    print_split_stats("Train", train_df)
    print_split_stats("Validation", val_df)
    print_split_stats("Test", test_df)
    
    # Calculate class weights for training
    n_usv = (train_df['label'] == 'usv').sum()
    n_not_usv = (train_df['label'] != 'usv').sum()
    
    # We want to PROTECT USV recall, so weight USV higher
    # Typical weight: inverse of frequency, but we boost USV extra
    usv_weight = len(train_df) / (2 * n_usv) * 1.5  # 1.5x boost for USV
    not_usv_weight = len(train_df) / (2 * n_not_usv)
    
    print(f"\nRecommended class weights:")
    print(f"  USV (class 1): {usv_weight:.3f}")
    print(f"  Not USV (class 0): {not_usv_weight:.3f}")
    print(f"  pos_weight for BCEWithLogitsLoss: {usv_weight/not_usv_weight:.3f}")
    
    # Save class weights
    weights = {
        'usv_weight': usv_weight,
        'not_usv_weight': not_usv_weight,
        'pos_weight': usv_weight / not_usv_weight
    }
    pd.DataFrame([weights]).to_csv(output_dir / 'class_weights.csv', index=False)
    
    print(f"\nOutput directory: {output_dir}")
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Create full training dataset')
    parser.add_argument('--original-labels', type=Path, required=True)
    parser.add_argument('--original-spectrograms', type=Path, required=True)
    parser.add_argument('--negatives-csv', type=Path, required=True)
    parser.add_argument('--negatives-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    create_full_dataset(
        args.original_labels,
        args.original_spectrograms,
        args.negatives_csv,
        args.negatives_dir,
        args.output_dir,
        args.test_size,
        args.val_size,
        args.seed
    )


if __name__ == '__main__':
    main()
```

### 2.2 Run Dataset Creation

```powershell
# Create full training dataset
python scripts/create_full_training_dataset.py `
    --original-labels labels.csv `
    --original-spectrograms spectrograms_training `
    --negatives-csv data/comprehensive_negatives/comprehensive_negatives_metadata.csv `
    --negatives-dir data/comprehensive_negatives `
    --output-dir data/full_training_dataset `
    --seed 42
```

---

## Phase 3: Train Full Model

### 3.1 Training Command

```powershell
# Train full model with comprehensive negatives
# - 50 epochs max (early stopping will likely trigger earlier)
# - Patience 15 for early stopping
# - Class weights to protect USV recall
python scripts/train_cnn.py `
    --train-csv data/full_training_dataset/train.csv `
    --val-csv data/full_training_dataset/val.csv `
    --spectrogram-dir data/full_training_dataset/spectrograms `
    --output-dir models/full_retrained_cnn `
    --batch-size 32 `
    --num-epochs 50 `
    --patience 15 `
    --use-class-weights `
    --learning-rate 0.001
```

### 3.2 Monitor Training

Watch for these patterns during training:

**Healthy training:**
- Train loss decreasing steadily
- Val loss decreasing (with some noise)
- Val accuracy improving
- Train-val gap < 15%

**Warning signs:**
- Val loss increasing while train loss decreases → overfitting
- Val accuracy stuck below 70% → learning rate or architecture issue
- Large oscillations in val metrics → learning rate too high

---

## Phase 4: Evaluate and Optimize Threshold

### 4.1 Run Full Evaluation

```powershell
# Evaluate on all three scenarios
python scripts/evaluate_experiment.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir analysis/full_retrained_evaluation
```

### 4.2 Threshold Optimization Script

**File:** `scripts/optimize_threshold.py`

```python
"""
Find optimal classification threshold for the retrained CNN.

Tests multiple thresholds and finds the one that:
1. Maximizes F1 score (balanced precision/recall)
2. Or achieves target recall with best precision
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN


def load_model(model_path: Path, device: str = 'auto'):
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = USVClassifierCNN()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def get_predictions(model, device, test_csv: Path, spec_dir: Path):
    """Get predictions for all test samples."""
    from PIL import Image
    
    df = pd.read_csv(test_csv)
    
    probabilities = []
    labels = []
    
    for _, row in df.iterrows():
        spec_path = spec_dir / row['spectrogram_path']
        if not spec_path.exists():
            continue
        
        # Load and predict
        spec = np.array(Image.open(spec_path).convert('L')).astype(np.float32)
        spec_norm = (spec - spec.mean()) / (spec.std() + 1e-8)
        
        with torch.no_grad():
            x = torch.from_numpy(spec_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            prob = model(x).item()
        
        probabilities.append(prob)
        labels.append(1 if row['label'] == 'usv' else 0)
    
    return np.array(probabilities), np.array(labels)


def optimize_threshold(
    model_path: Path,
    test_csv: Path,
    spec_dir: Path,
    output_dir: Path,
    target_recall: float = 0.90
):
    """Find optimal threshold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model, device = load_model(model_path)
    
    print("Getting predictions...")
    probs, labels = get_predictions(model, device, test_csv, spec_dir)
    
    print(f"Total samples: {len(probs)}")
    print(f"  USV: {labels.sum()}")
    print(f"  Not USV: {(1-labels).sum()}")
    
    # Test multiple thresholds
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / len(labels)
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal thresholds
    best_f1_idx = results_df['f1'].idxmax()
    best_f1_thresh = results_df.loc[best_f1_idx, 'threshold']
    
    # Find threshold that achieves target recall with best precision
    high_recall = results_df[results_df['recall'] >= target_recall]
    if len(high_recall) > 0:
        best_recall_idx = high_recall['precision'].idxmax()
        best_recall_thresh = high_recall.loc[best_recall_idx, 'threshold']
    else:
        best_recall_thresh = results_df.loc[results_df['recall'].idxmax(), 'threshold']
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision')
    ax.plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall')
    ax.plot(results_df['threshold'], results_df['f1'], 'r-', label='F1')
    ax.axvline(x=best_f1_thresh, color='r', linestyle='--', alpha=0.5, label=f'Best F1 ({best_f1_thresh:.2f})')
    ax.axvline(x=best_recall_thresh, color='g', linestyle='--', alpha=0.5, label=f'High Recall ({best_recall_thresh:.2f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(results_df['recall'], results_df['precision'], 'b-')
    ax.scatter([results_df.loc[best_f1_idx, 'recall']], 
               [results_df.loc[best_f1_idx, 'precision']], 
               color='red', s=100, zorder=5, label=f'Best F1 (thresh={best_f1_thresh:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_optimization.png', dpi=150)
    plt.close()
    
    # Save results
    results_df.to_csv(output_dir / 'threshold_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nBest F1 threshold: {best_f1_thresh:.2f}")
    print(f"  Precision: {results_df.loc[best_f1_idx, 'precision']:.3f}")
    print(f"  Recall: {results_df.loc[best_f1_idx, 'recall']:.3f}")
    print(f"  F1: {results_df.loc[best_f1_idx, 'f1']:.3f}")
    
    print(f"\nHigh-recall threshold (target {target_recall:.0%}): {best_recall_thresh:.2f}")
    high_recall_row = results_df[results_df['threshold'] == best_recall_thresh].iloc[0]
    print(f"  Precision: {high_recall_row['precision']:.3f}")
    print(f"  Recall: {high_recall_row['recall']:.3f}")
    print(f"  F1: {high_recall_row['f1']:.3f}")
    
    # Save recommendation
    with open(output_dir / 'recommended_threshold.txt', 'w') as f:
        f.write(f"Best F1 threshold: {best_f1_thresh:.2f}\n")
        f.write(f"High-recall threshold: {best_recall_thresh:.2f}\n")
        f.write(f"\nRecommendation: Use {best_recall_thresh:.2f} for batch detection\n")
        f.write("(prioritizes catching all USVs over avoiding false positives)\n")
    
    return best_f1_thresh, best_recall_thresh


def main():
    parser = argparse.ArgumentParser(description='Optimize classification threshold')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--spec-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-recall', type=float, default=0.90)
    
    args = parser.parse_args()
    
    optimize_threshold(
        args.model,
        args.test_csv,
        args.spec_dir,
        args.output_dir,
        args.target_recall
    )


if __name__ == '__main__':
    main()
```

### 4.3 Run Threshold Optimization

```powershell
# Find optimal threshold
python scripts/optimize_threshold.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --spec-dir data/full_training_dataset/spectrograms `
    --output-dir analysis/threshold_optimization `
    --target-recall 0.90
```

---

## Phase 5: Final Validation

### 5.1 Batch Detection Test

After threshold optimization, run the final batch detection test:

```powershell
# Final evaluation with optimized threshold
python scripts/evaluate_experiment.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir analysis/final_evaluation `
    --threshold 0.XX  # Use threshold from optimization
```

---

## Execution Summary

Run these commands in order:

```powershell
# Phase 1: Generate comprehensive negatives (~5-10 minutes)
python scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/comprehensive_negatives `
    --n-random 500 `
    --n-inter-usv 300 `
    --n-low-energy 200

# Phase 2: Create full training dataset (~2-3 minutes)
python scripts/create_full_training_dataset.py `
    --original-labels labels.csv `
    --original-spectrograms spectrograms_training `
    --negatives-csv data/comprehensive_negatives/comprehensive_negatives_metadata.csv `
    --negatives-dir data/comprehensive_negatives `
    --output-dir data/full_training_dataset

# Phase 3: Train full model (~15-30 minutes on CPU, faster on GPU)
python scripts/train_cnn.py `
    --train-csv data/full_training_dataset/train.csv `
    --val-csv data/full_training_dataset/val.csv `
    --spectrogram-dir data/full_training_dataset/spectrograms `
    --output-dir models/full_retrained_cnn `
    --batch-size 32 `
    --num-epochs 50 `
    --patience 15 `
    --use-class-weights

# Phase 4: Evaluate
python scripts/evaluate_experiment.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir analysis/full_retrained_evaluation

# Phase 5: Optimize threshold
python scripts/optimize_threshold.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --spec-dir data/full_training_dataset/spectrograms `
    --output-dir analysis/threshold_optimization `
    --target-recall 0.90
```

---

## Expected Outcomes

### Dataset Statistics

| Category | Count |
|----------|-------|
| Original USVs | ~460 |
| Original Not USV | ~470 |
| New random negatives | 500 |
| New inter-USV gap negatives | 300 |
| New low-energy negatives | 200 |
| **Total** | **~1,930** |

### Target Performance

| Metric | Target |
|--------|--------|
| USV samples mean prob | >0.85 |
| Not USV samples mean prob | <0.30 |
| Random chunks mean prob | <0.20 |
| Test set accuracy | >85% |
| Test set recall | >90% |
| Test set precision | >80% |

### Success Criteria

The retraining is successful if:

1. ✓ Random chunks → probability < 0.20 (batch detection works)
2. ✓ USV samples → probability > 0.85 (still recognizes USVs)
3. ✓ Test accuracy > 85%
4. ✓ Test recall > 90% (catches most USVs)

---

## Troubleshooting

### If USV recall is still low (<80%):

1. Increase USV class weight in training
2. Use threshold optimization to find recall-focused threshold
3. Check if low-energy negatives are accidentally removing USVs

### If random chunk probability is still high (>0.3):

1. Add more random negatives (try 1000 instead of 500)
2. Verify random negatives are being extracted correctly
3. Check normalization consistency between training and evaluation

### If training is unstable:

1. Reduce learning rate to 0.0005
2. Increase batch size if GPU memory allows
3. Add gradient clipping
