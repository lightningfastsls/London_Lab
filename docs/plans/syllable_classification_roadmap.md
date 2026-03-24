# Phase 14: USV Syllable Classification — Implementation Roadmap

> **Purpose:** This file defines Phase 14 of the USV pipeline — syllable TYPE classification
> using both supervised (Scattoni taxonomy) and unsupervised (VAE continuous space) approaches.
> It follows the same format as `ROADMAP.md` and can be merged into it when ready.
>
> **Relationship to existing phases:**
> - **Phase 7** (DONE): Unsupervised clustering via CNN features — discovers groupings without labels
> - **Phase 8** (DONE): Transformer + VQ-VAE discovers "concepts" from temporal prediction
> - **Phase 12** (FUTURE): Population comparison — currently uses CNN features and clusters
> - **Phase 14** (NEW): Traditional syllable classification + continuous VAE space — complements both
>
> **Key scientific insight:** Mouse USVs may form a continuous manifold rather than discrete
> categories (Goffinet et al., 2021). Phase 14 addresses this by running BOTH supervised
> classification (for comparability with published literature) and unsupervised VAE exploration
> (for data-driven discovery). The dual approach avoids imposing artificial boundaries while
> still enabling comparison with the standard taxonomy.
>
> **Source:** `compass_artifact_wf-9efbb114-e539-4040-b42e-660262d6248a_text_markdown.md`

---

## 14.1 Classification Spectrogram Extraction

**What:** Extract spectrogram patches from detected USVs with padding, compute high-resolution STFTs optimized for frequency contour classification (finer than the detection-stage STFT), and resize to fixed dimensions for CNN input. Optionally supports Gammatone spectrograms.
**Status:** FUTURE
**Review Tier:** 2
**Depends on:** Phase 5 (detected USVs) or Phase 13 (batch detection results)

**Key design decisions:**
- 1024-point FFT at 300 kHz → ~293 Hz frequency resolution (finer than detection's 512-point / ~586 Hz)
- Fine frequency resolution matters more than time resolution for CNN-based classification
- 25–125 kHz frequency band (slightly wider than detection's 20–120 kHz to capture full contours)
- 75% overlap for smooth temporal representation
- Fixed output size: 128×128 (default) or 224×224 (for ImageNet transfer learning)
- Optional Gammatone spectrogram alternative (BootSnap found these outperform standard STFTs)

/implement Classification Spectrogram Extraction (Phase 14.1)

Extract high-resolution spectrogram patches from detected USV segments for syllable classification. These patches are the input for both the supervised classifier (14.2) and the unsupervised VAE (14.3).

**Context:** Per ADR-001, sample rate is 300 kHz. The detection pipeline uses 512-point FFT (ADR-002), but classification benefits from finer frequency resolution — 1024-point FFT gives ~293 Hz bins, better for resolving frequency contour shapes that define syllable types. BootSnap (Abbasi et al., 2022) found that Gammatone spectrograms outperform standard STFTs for USV classification, so include this as an option. The ~15 ms padding captures onset/offset transitions that help distinguish syllable types.

**Files to create:**

1. `src/usv_spectrogram/classification/__init__.py` (NEW)

2. `src/usv_spectrogram/classification/config.py` (NEW) — Configuration

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ClassificationExtractionConfig:
    """Configuration for classification-stage spectrogram extraction."""
    # STFT parameters (finer than detection stage)
    n_fft: int = 1024               # 1024-point FFT at 300 kHz → ~293 Hz resolution
    hop_length: int = 256            # 75% overlap (1024 * 0.25 = 256)
    sr: int = 300_000                # ADR-001: always explicit
    freq_min_hz: int = 25_000        # Lower bound
    freq_max_hz: int = 125_000       # Upper bound

    # Padding around detected USV
    padding_ms: float = 15.0         # ~15 ms each side for onset/offset context

    # Output patch dimensions
    patch_height: int = 128          # Frequency axis (pixels)
    patch_width: int = 128           # Time axis (pixels)
    use_transfer_learning_size: bool = False  # If True, resize to 224×224

    # Spectrogram type
    use_gammatone: bool = False       # Gammatone spectrogram (BootSnap approach)
    n_gammatone_filters: int = 128    # Number of Gammatone filterbank channels

    # Input/output
    detections_dir: Path = Path("USV_Detections")
    wav_dir: Path = Path("5970 USV")
    output_dir: Path = Path("data/classification/patches")
```

3. `src/usv_spectrogram/classification/patch_extractor.py` (NEW) — Extraction logic

```python
class ClassificationPatchExtractor:
    """Extract classification-quality spectrogram patches from detected USVs."""

    def __init__(self, config: ClassificationExtractionConfig): ...

    def extract_patch(self, wav_path: Path, start_s: float, end_s: float) -> np.ndarray:
        """
        Extract a single spectrogram patch for a detected USV.

        Steps:
        1. Load audio segment [start - padding, end + padding] from WAV
        2. Compute STFT (n_fft=1024, hop=256, Hamming window)
        3. Convert to log-magnitude: 20 * log10(|S| + 1e-10)
        4. Crop frequency axis to 25-125 kHz
        5. Resize to (patch_height, patch_width)
        6. Return normalized patch as numpy array
        """
        ...

    def extract_gammatone_patch(self, wav_path: Path, start_s: float, end_s: float) -> np.ndarray:
        """
        Extract Gammatone spectrogram patch (alternative to STFT).
        Uses a Gammatone filterbank (128 channels) centered on 25-125 kHz.
        """
        ...

    def extract_all(self, detections: list[dict], progress_callback=None) -> list[dict]:
        """
        Extract patches for all detections.

        Args:
            detections: list of {wav_path, start_s, end_s, label (optional)}
        Returns:
            list of {patch_path, wav_source, start_s, end_s, label, patch_shape}
        """
        ...

    def _load_audio_segment(self, wav_path: Path, start_s: float, end_s: float) -> np.ndarray:
        """Load audio segment with padding, clamped to file bounds."""
        ...

    def _resize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Resize spectrogram patch to configured dimensions."""
        ...
```

4. `scripts/extract_classification_patches.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/extract_classification_patches.py \
      --detections-dir USV_Detections \
      --wav-dir "5970 USV" \
      --output-dir data/classification/patches \
      --n-fft 1024 \
      --patch-size 128 \
      --gammatone  # optional: use Gammatone instead of STFT
```

Output structure:
```
data/classification/patches/
├── patches/                    # Spectrogram PNG patches
│   ├── rec001_det001.png
│   ├── rec001_det002.png
│   └── ...
├── manifest.csv               # patch_path, wav_source, start_s, end_s, duration_ms, label
└── extraction_report.json     # Config used, counts, duration stats
```

5. `tests/test_classification_extraction.py` (NEW)

**Test plan:**
```
1. Extraction from synthetic WAV produces patch of correct shape (128, 128)
2. Padding extends segment by ~15 ms each side (verified with known audio)
3. Frequency axis correctly cropped to 25-125 kHz range
4. Gammatone extraction produces patch of correct shape
5. Patches from short USVs (<5 ms) are handled gracefully (padded to minimum)
6. Patches from USVs near file boundaries are clamped correctly
7. manifest.csv contains one row per extracted patch with correct columns
8. use_transfer_learning_size=True produces 224×224 patches
```

**Exit criteria:**
- [ ] Extract patches from 100+ detected USVs without errors
- [ ] Visual inspection: patches show clear frequency contour structure
- [ ] Manifest CSV is complete and all patch paths resolve to existing files
- [ ] Both STFT and Gammatone extraction modes work
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## 14.2 Supervised Syllable Classifier

**What:** Fine-tune a pretrained CNN (MobileNetV2 or ResNet-18) on spectrogram patches to classify USVs into 11 categories: the 10 Scattoni taxonomy types plus a noise class. Uses snapshot ensemble learning for improved generalization and cross-domain validation to test wild↔lab transfer.
**Status:** FUTURE
**Review Tier:** 3
**Depends on:** Phase 14.1

**Key design decisions:**
- 11 classes: flat, chevron, short, up-FM, down-FM, complex, step-up, step-down, two-syllable, composite, noise
- Transfer learning from ImageNet (MobileNetV2 as default — lightweight, proven on spectrograms)
- Snapshot ensemble (BootSnap approach): save models at learning rate cycle minima, ensemble predictions
- VocalMat provides 12,954 labeled spectrograms as starter training data
- Cross-domain validation: train on lab → test on wild (and vice versa) to measure generalization
- Classifiers trained on lab mice generalize poorly to wild mice (BootSnap key finding)

/implement Supervised Syllable Classifier (Phase 14.2)

Build a CNN-based supervised classifier for USV syllable types using the Scattoni taxonomy. This enables comparison with the published USV literature and provides named categories for each detected USV.

**Context:** VocalMat (Fonseca et al.) provides 12,954 labeled spectrograms across 11 categories freely available on GitHub — use these as starter training data. BootSnap (Abbasi et al., 2022) showed snapshot ensemble learning achieves F1 67–74.5% across wild and lab mice. Their key finding: classifiers trained on one population generalize poorly to the other. Include cross-domain validation.

**Files to create:**

1. `src/usv_spectrogram/classification/syllable_config.py` (NEW) — Configuration

```python
from dataclasses import dataclass, field
from pathlib import Path

# Scattoni et al. (2008) taxonomy + noise class
SYLLABLE_CLASSES = [
    "flat", "chevron", "short", "up_fm", "down_fm",
    "complex", "step_up", "step_down", "two_syllable",
    "composite", "noise"
]

@dataclass(frozen=True)
class SyllableClassifierConfig:
    """Configuration for supervised syllable classification."""
    n_classes: int = 11
    class_names: tuple[str, ...] = tuple(SYLLABLE_CLASSES)

    # Architecture
    backbone: str = "mobilenet_v2"    # or "resnet18"
    pretrained: bool = True           # ImageNet weights
    freeze_backbone_epochs: int = 5   # Freeze backbone initially, then fine-tune all

    # Snapshot ensemble (BootSnap approach)
    use_snapshot_ensemble: bool = True
    n_snapshots: int = 5              # Models saved at LR cycle minima
    cycle_length_epochs: int = 20     # Cosine annealing cycle length

    # Training
    learning_rate: float = 1e-3       # For classifier head
    backbone_lr: float = 1e-5         # For backbone after unfreezing
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 15

    # Data
    train_patches_dir: Path = Path("data/classification/patches")
    vocalmat_dir: Path | None = None  # Optional: VocalMat labeled data
    class_weights: str = "balanced"   # or "none", or manual dict

    # Cross-domain validation
    cross_domain_validation: bool = True  # Train lab→test wild & vice versa
```

2. `src/usv_spectrogram/classification/syllable_classifier.py` (NEW) — Model

```python
class SyllableClassifier(nn.Module):
    """
    CNN classifier for USV syllable types.

    Architecture:
    - Backbone: MobileNetV2 (pretrained on ImageNet) with final FC removed
    - Classifier head: Linear(1280 → 256) → ReLU → Dropout(0.3) → Linear(256 → 11)
    - Snapshot ensemble: save model weights at cosine annealing LR minima
    """

    def __init__(self, config: SyllableClassifierConfig): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x of shape (batch, 3, H, W) — RGB spectrogram patches
        Returns: logits of shape (batch, n_classes)
        """
        ...

    def freeze_backbone(self): ...
    def unfreeze_backbone(self): ...


class SnapshotEnsemble:
    """
    Snapshot ensemble (Huang et al., 2017).
    Saves model checkpoints at cosine annealing cycle minima.
    At inference, averages predictions from all snapshots.
    """
    def __init__(self, n_snapshots: int, cycle_length: int): ...

    def should_save_snapshot(self, epoch: int) -> bool: ...

    def predict(self, x: torch.Tensor, models: list[nn.Module]) -> torch.Tensor:
        """Average softmax predictions from all snapshot models."""
        ...
```

3. `src/usv_spectrogram/classification/syllable_trainer.py` (NEW) — Training loop

```python
class SyllableTrainer:
    """Training loop for syllable classifier with snapshot ensemble support."""

    def __init__(self, config: SyllableClassifierConfig): ...

    def train(self, train_loader, val_loader) -> dict:
        """
        Training procedure:
        1. Freeze backbone, train classifier head for freeze_backbone_epochs
        2. Unfreeze backbone, train all with differential learning rates
        3. Cosine annealing LR schedule with warm restarts
        4. Save snapshot at each cycle minimum (if snapshot ensemble enabled)
        5. Early stopping on validation loss
        6. Return training history and snapshot paths
        """
        ...

    def evaluate(self, test_loader, snapshot_paths: list[Path] | None = None) -> dict:
        """
        Evaluate model or snapshot ensemble on test set.

        Returns:
        - Per-class precision, recall, F1
        - Confusion matrix
        - Overall accuracy, macro F1, weighted F1
        """
        ...

    def cross_domain_evaluate(self, lab_loader, wild_loader) -> dict:
        """
        Cross-domain validation:
        1. Train on lab data, evaluate on wild data
        2. Train on wild data, evaluate on lab data
        3. Report transfer degradation for each direction
        """
        ...
```

4. `scripts/train_syllable_classifier.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/train_syllable_classifier.py \
      --patches-dir data/classification/patches \
      --vocalmat-dir data/vocalmat_labeled \
      --backbone mobilenet_v2 \
      --snapshot-ensemble \
      --output-dir models/syllable_classifier \
      --cross-domain-validation
```

Output structure:
```
models/syllable_classifier/
├── best_model.pt                   # Best single model
├── snapshots/                      # Snapshot ensemble models
│   ├── snapshot_epoch_20.pt
│   ├── snapshot_epoch_40.pt
│   └── ...
├── training_history.json           # Loss curves, LR schedule
├── confusion_matrix.png            # 11×11 confusion matrix
├── per_class_metrics.json          # Per-class P/R/F1
├── cross_domain_results.json       # Lab→wild and wild→lab transfer
└── training_report.md              # Summary report
```

5. `tests/test_syllable_classifier.py` (NEW)

**Test plan:**
```
1. Forward pass on dummy 3×128×128 input produces logits of shape (batch, 11)
2. Backbone freezing prevents gradient updates to backbone parameters
3. Backbone unfreezing allows gradient updates to all parameters
4. Snapshot ensemble averages predictions from N models correctly
5. Class weights correctly handle imbalanced training data
6. Training loss decreases on single-batch overfit
7. Per-class metrics computed correctly on synthetic predictions
8. Cross-domain evaluate produces results for both directions
9. Confusion matrix has correct dimensions (11×11)
```

**Exit criteria:**
- [ ] Model produces correct output shapes for both 128×128 and 224×224 inputs
- [ ] Training converges on VocalMat data (or subset) with decreasing loss
- [ ] Snapshot ensemble outperforms single model on validation set
- [ ] Cross-domain validation quantifies lab↔wild transfer gap
- [ ] Per-class F1 reported for all 11 categories
- [ ] Confusion matrix identifies which syllable types are most confused
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## 14.3 Unsupervised Continuous USV Space (VAE)

**What:** Train a convolutional VAE on spectrogram patches to learn a continuous latent representation of USV variation. This complements the supervised classifier by exploring whether USVs form discrete categories or a continuous manifold. Includes UMAP visualization, GMM clustering at various k, and interpolation tests.
**Status:** FUTURE
**Review Tier:** 3
**Depends on:** Phase 14.1

**Key design decisions:**
- Convolutional VAE following Goffinet et al. (2021) AVA approach
- 32-dimensional latent space (matching AVA's choice)
- Goffinet found only k≤2 GMM clusters supported for mouse USVs (vs. clean clusters for zebra finch)
- Smooth interpolation between USV types tests whether the space is truly continuous
- Color UMAP by supervised labels (from 14.2) to bridge both approaches

/implement Unsupervised Continuous USV Space VAE (Phase 14.3)

Build a convolutional VAE that learns a continuous latent space from USV spectrogram patches. This addresses the key scientific question: do mouse USVs form discrete syllable categories, or a continuous manifold?

**Context:** Goffinet et al. (2021, eLife) used AVA (Autoencoded Vocal Analysis) and found that mouse USVs form a continuous spectrum — GMM clustering only supported k≤2 clusters. This contrasts with zebra finch syllables which cluster cleanly. A 32-dimensional latent space captures the continuous variation. Color the UMAP by supervised labels from 14.2 to see whether named categories correspond to distinct regions.

**Files to create:**

1. `src/usv_spectrogram/classification/vae_config.py` (NEW)

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class USVVAEConfig:
    """Configuration for USV spectrogram VAE."""
    # Latent space
    latent_dim: int = 32            # Following Goffinet et al. (2021)

    # Encoder architecture
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    encoder_kernel_size: int = 4
    encoder_stride: int = 2

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 200
    early_stopping_patience: int = 20
    kl_weight: float = 1.0         # β in β-VAE (1.0 = standard VAE)
    kl_warmup_epochs: int = 10     # Gradually increase KL weight

    # Clustering evaluation
    gmm_k_range: tuple[int, ...] = (2, 3, 5, 8, 10, 15, 20)  # Test these k values
    n_umap_neighbors: int = 15

    # Input
    input_shape: tuple[int, int] = (128, 128)  # Matches 14.1 patch size
    patches_dir: Path = Path("data/classification/patches")
    output_dir: Path = Path("models/usv_vae")
```

2. `src/usv_spectrogram/classification/usv_vae.py` (NEW) — Model

```python
class USVVAE(nn.Module):
    """
    Convolutional VAE for USV spectrogram patches.

    Architecture:
    - Encoder: 4 conv blocks (32→64→128→256), each: Conv2d → BatchNorm → LeakyReLU
    - Bottleneck: Flatten → Linear → (mu, log_var) each of dim 32
    - Decoder: Linear → Reshape → 4 transposed conv blocks (256→128→64→32→1)
    - Loss: reconstruction (BCE or MSE) + KL divergence with optional β weighting
    """

    def __init__(self, config: USVVAEConfig): ...

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var) each of shape (batch, latent_dim)."""
        ...

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon."""
        ...

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed spectrogram."""
        ...

    def forward(self, x):
        """Returns (reconstruction, mu, log_var, z)."""
        ...

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 10) -> list:
        """Linear interpolation between two latent vectors. Tests continuity."""
        ...
```

3. `src/usv_spectrogram/classification/vae_analysis.py` (NEW) — Analysis tools

```python
class VAEAnalyzer:
    """Analysis tools for the trained USV VAE."""

    def __init__(self, model: USVVAE, config: USVVAEConfig): ...

    def encode_dataset(self, data_loader) -> tuple[np.ndarray, np.ndarray]:
        """Encode all patches to latent space. Returns (latent_vectors, labels)."""
        ...

    def umap_visualization(self, latent_vectors, labels=None, syllable_labels=None):
        """
        UMAP projection of latent space.
        - Color by supervised syllable labels (from 14.2) if available
        - Color by GMM cluster assignment
        - Color by recording source (wild vs lab)
        """
        ...

    def gmm_clustering(self, latent_vectors) -> dict:
        """
        Fit GMM at various k values. For each k:
        - BIC, AIC, silhouette score
        - Goffinet found only k<=2 supported for mice — test this claim
        Returns dict with scores per k and recommended k.
        """
        ...

    def interpolation_gallery(self, z1, z2, n_steps=10) -> np.ndarray:
        """Generate decoded spectrograms along interpolation path."""
        ...

    def latent_space_traversal(self, z_mean, dim_idx, n_steps=10, range_std=3.0):
        """Traverse single latent dimension, decode at each step."""
        ...

    def reconstruction_quality(self, data_loader) -> dict:
        """Compute reconstruction MSE per syllable type (if labels available)."""
        ...
```

4. `scripts/train_usv_vae.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/train_usv_vae.py \
      --patches-dir data/classification/patches \
      --latent-dim 32 \
      --beta 1.0 \
      --output-dir models/usv_vae
```

5. `scripts/analyze_usv_vae.py` (NEW) — Analysis CLI

```
Usage:
  .\.venv\Scripts\python.exe scripts/analyze_usv_vae.py \
      --model models/usv_vae/best_model.pt \
      --patches-dir data/classification/patches \
      --syllable-labels models/syllable_classifier/predictions.csv \
      --output-dir analysis/vae_results
```

Output structure:
```
analysis/vae_results/
├── umap_by_syllable_type.png       # UMAP colored by supervised labels
├── umap_by_gmm_cluster.png         # UMAP colored by GMM assignment
├── umap_by_population.png          # UMAP colored by wild/lab
├── gmm_bic_scores.png              # BIC vs k plot
├── interpolation_galleries/        # Smooth transitions between USV types
│   ├── flat_to_chevron.png
│   ├── up_fm_to_down_fm.png
│   └── ...
├── latent_traversals/              # Single-dimension traversals
│   ├── dim_00.png
│   └── ...
├── reconstruction_examples.png     # Original vs reconstructed patches
└── vae_analysis_report.md          # Summary with all metrics
```

6. `tests/test_usv_vae.py` (NEW)

**Test plan:**
```
1. Forward pass on dummy (batch, 1, 128, 128) input produces reconstruction of same shape
2. Encode produces mu, log_var of correct shape (batch, 32)
3. Reparameterize produces different z for different random seeds
4. KL divergence is non-negative
5. Interpolation produces n_steps decoded outputs
6. GMM clustering returns BIC scores for all k values
7. VAE loss decreases on single-batch overfit
8. KL warmup schedule increases weight from 0 to target over warmup epochs
9. Latent traversal produces decoded patches of correct shape
```

**Exit criteria:**
- [ ] VAE reconstruction quality: patches visually recognizable after encode/decode
- [ ] UMAP shows meaningful structure (not uniform cloud)
- [ ] GMM BIC curve tested — verify Goffinet's k≤2 finding for our data
- [ ] Interpolation between syllable types produces smooth transitions
- [ ] Latent traversals show interpretable variation (frequency shifts, duration changes)
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## 14.4 Dual Classification & Repertoire Analysis

**What:** Run both supervised (14.2) and unsupervised (14.3) on the same data, compare agreements, and compute syllable-level repertoire metrics. Extends Phase 12 with syllable-based comparison methods including per-animal profiles, transition matrices, Syntax Information Score, and distributional distances.
**Status:** FUTURE
**Review Tier:** 2
**Depends on:** Phase 14.2, Phase 14.3

**Key design decisions:**
- Dual approach: compare supervised labels vs. VAE clusters on same data
- Syntax Information Score (Hertz et al., 2020) ranks classification quality by temporal prediction
- Bigram transition matrices reveal sequential structure in syllable sequences
- Shannon entropy measures repertoire diversity per animal/group
- Distributional distances: EMD/JSD on VAE embeddings, PERMANOVA on Bray-Curtis
- This extends Phase 12 with syllable-type-aware population comparison

/implement Dual Classification & Repertoire Analysis (Phase 14.4)

Build the analysis pipeline that combines supervised and unsupervised classification results, computes repertoire metrics, and extends Phase 12's population comparison with syllable-type-aware methods.

**Context:** Hertz et al. (2020, Communications Biology) showed that different classification schemes produce no one-to-one mapping between labels and developed the Syntax Information Score (SIS) to rank schemes by temporal prediction ability. PERMANOVA on Bray-Curtis distance is the standard method for comparing USV repertoires (used across the field). Zala et al. (2020) found wild mice use 9 syllable types during interaction vs. 6 during introduction — social context matters.

**Files to create:**

1. `src/usv_spectrogram/classification/repertoire_analysis.py` (NEW) — Core analysis

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RepertoireAnalysisConfig:
    """Configuration for syllable repertoire analysis."""
    supervised_predictions_path: Path   # CSV: patch_id, predicted_class, confidence
    vae_embeddings_path: Path           # NPY: latent vectors from VAE
    vae_gmm_labels_path: Path | None    # CSV: GMM cluster assignments
    metadata_path: Path                 # CSV: animal_id, population, sex, social_context
    output_dir: Path = Path("analysis/repertoire")
    n_permutations: int = 1000


class RepertoireAnalyzer:
    """Syllable-level repertoire analysis combining supervised and unsupervised results."""

    def __init__(self, config: RepertoireAnalysisConfig): ...

    def run_full_analysis(self) -> dict:
        """Run all analyses, generate report."""
        ...

    # --- Per-animal profiles ---

    def compute_syllable_profiles(self, labels: np.ndarray, animal_ids: np.ndarray) -> pd.DataFrame:
        """
        Per-animal syllable proportions.
        Returns DataFrame: animal_id × syllable_type (proportions sum to 1.0 per animal).
        """
        ...

    # --- Sequential structure ---

    def compute_transition_matrix(self, syllable_sequence: list[str]) -> np.ndarray:
        """
        Bigram transition matrix P(type_{t+1} | type_t).
        Rows = current type, columns = next type. Rows sum to 1.0.
        """
        ...

    def syntax_information_score(self, sequences: list[list[str]]) -> float:
        """
        Syntax Information Score (Hertz et al., 2020).
        Measures how well syllable labels predict the next syllable.
        Higher SIS = labels capture more temporal structure = better classification scheme.

        SIS = H(S) - H(S | S_{t-1})
        where H(S) is marginal entropy and H(S | S_{t-1}) is conditional entropy.
        """
        ...

    # --- Repertoire diversity ---

    def shannon_entropy(self, proportions: np.ndarray) -> float:
        """Shannon entropy of syllable proportions. Higher = more diverse repertoire."""
        ...

    def repertoire_size(self, labels: np.ndarray, threshold: float = 0.01) -> int:
        """Number of syllable types used above threshold proportion."""
        ...

    # --- Population comparison ---

    def bray_curtis_permanova(self, profiles: pd.DataFrame, groups: np.ndarray) -> dict:
        """
        PERMANOVA on Bray-Curtis distance of syllable profiles.
        Standard method for comparing USV repertoires between populations.
        Returns {F_statistic, p_value, R_squared}.
        """
        ...

    def distributional_distance_vae(self, embeddings: np.ndarray, groups: np.ndarray) -> dict:
        """
        Compare VAE embedding distributions between populations.
        - Earth Mover's Distance (EMD) on latent space
        - Jensen-Shannon Divergence on discretized latent space
        Returns {emd, jsd, p_value_permutation}.
        """
        ...

    # --- Dual comparison ---

    def supervised_vs_unsupervised_agreement(
        self, supervised_labels: np.ndarray, gmm_labels: np.ndarray
    ) -> dict:
        """
        Compare supervised classification vs. VAE+GMM clustering.
        - Adjusted Rand Index
        - Normalized Mutual Information
        - Contingency table (which supervised types map to which GMM clusters)
        """
        ...

    def compare_classification_schemes(self, schemes: dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compare multiple labeling schemes using Syntax Information Score.
        schemes: {"supervised": labels, "gmm_k2": labels, "gmm_k5": labels, ...}
        Returns ranking by SIS (higher = better temporal prediction).
        """
        ...
```

2. `scripts/analyze_syllable_repertoire.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/analyze_syllable_repertoire.py \
      --supervised-predictions models/syllable_classifier/predictions.csv \
      --vae-embeddings models/usv_vae/latent_vectors.npy \
      --metadata analysis/metadata.csv \
      --output-dir analysis/repertoire
```

Output structure:
```
analysis/repertoire/
├── syllable_profiles/
│   ├── per_animal_profiles.csv         # Animal × syllable type proportions
│   ├── population_profiles.png         # Stacked bar chart by population
│   └── profile_heatmap.png             # Heatmap of animal × type
├── transitions/
│   ├── transition_matrix_all.png       # Full population transition heatmap
│   ├── transition_matrix_wild.png      # Wild-only transitions
│   ├── transition_matrix_lab.png       # Lab-only transitions
│   └── transition_comparison.json      # Chi-squared test between populations
├── diversity/
│   ├── entropy_by_population.png       # Box plot of per-animal entropy
│   ├── repertoire_size_comparison.png  # Bar chart of syllable types used
│   └── diversity_tests.json            # Mann-Whitney U test results
├── population_comparison/
│   ├── permanova_results.json          # Bray-Curtis PERMANOVA
│   ├── vae_distance_results.json       # EMD + JSD on latent space
│   └── comparison_summary.md           # Plain-language interpretation
├── scheme_comparison/
│   ├── sis_ranking.csv                 # Syntax Information Score per scheme
│   ├── agreement_metrics.json          # ARI, NMI between schemes
│   └── contingency_table.png           # Supervised × GMM cross-tabulation
└── repertoire_report.md                # Full report with all figures
```

3. `tests/test_repertoire_analysis.py` (NEW)

**Test plan:**
```
1. Syllable profiles sum to 1.0 per animal
2. Transition matrix rows sum to ~1.0 and dimensions are (n_types, n_types)
3. Syntax Information Score is non-negative and <= H(S) (marginal entropy)
4. Shannon entropy returns 0 for single-type repertoire, log(n) for uniform
5. PERMANOVA p-value in [0, 1]; identical populations → p > 0.05
6. Supervised vs unsupervised agreement: ARI in [-0.5, 1.0], NMI in [0, 1]
7. Compare_classification_schemes ranks by SIS correctly on synthetic data
8. EMD is non-negative and 0 for identical distributions
```

**Exit criteria:**
- [ ] All analyses run without errors on synthetic data
- [ ] SIS correctly ranks known-better classification on synthetic Markov chain
- [ ] PERMANOVA detects significant difference between synthetic divergent populations
- [ ] Agreement metrics correctly identify identical vs. random label correspondence
- [ ] Repertoire report is self-contained with plain-language interpretation
- [ ] Transition matrix heatmaps are readable and correctly labeled
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 14 Dependencies

```
Phase 5 (CNN detection) ──────────→ Phase 14.1 (patch extraction)
Phase 13 (batch detection) ───────→ Phase 14.1 (batch patches)
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                    Phase 14.2                Phase 14.3
                (supervised CNN)            (unsupervised VAE)
                              │                   │
                              └─────────┬─────────┘
                                        ▼
                                  Phase 14.4
                          (dual analysis + repertoire)
                                        │
                                        ▼
                                  Phase 12
                          (population comparison,
                           extended with syllable metrics)
```

## Phase 14 External Dependencies

```
torch >= 2.0
torchvision            # MobileNetV2/ResNet-18 pretrained models
librosa                # STFT computation
gammatone              # Optional: Gammatone filterbank (pip install gammatone)
scikit-learn           # GMM, metrics
scikit-bio             # PERMANOVA, Bray-Curtis (pip install scikit-bio)
scipy                  # Earth Mover's Distance, statistical tests
umap-learn             # UMAP projections
matplotlib, seaborn    # Visualization
```

## Phase 14 Gate

Before extending Phase 12 with syllable metrics:
- [ ] Classification patches extracted from 1000+ detected USVs (14.1)
- [ ] Supervised classifier achieves macro F1 > 50% on test set (14.2)
- [ ] Cross-domain validation quantifies lab↔wild transfer gap (14.2)
- [ ] VAE UMAP shows meaningful structure, not uniform cloud (14.3)
- [ ] GMM BIC curve tested for optimal k (14.3)
- [ ] Dual agreement (ARI, NMI) computed between supervised and unsupervised (14.4)
- [ ] SIS computed for at least 3 classification schemes (14.4)
- [ ] PERMANOVA computed on syllable profiles (14.4)
- [ ] All Phase 14 tests pass
- [ ] py_compile passes on all new files

---

## Recommended Implementation Order

| Priority | Module | Why |
|----------|--------|-----|
| **1** | 14.1 (Patch Extraction) | Foundation — all other sub-modules depend on this |
| **2** | 14.2 (Supervised Classifier) | VocalMat provides free training data to start immediately |
| **3** | 14.3 (VAE) | Can train in parallel with 14.2 on same patches |
| **4** | 14.4 (Dual Analysis) | Requires outputs from both 14.2 and 14.3 |

---

## Key References

- Scattoni et al. (2008) — Standard 10-type USV taxonomy
- Goffinet et al. (2021, eLife) — AVA: continuous USV manifold, GMM k≤2 for mice
- Hertz et al. (2020, Communications Biology) — Syntax Information Score
- Abbasi et al. / BootSnap (2022, PLOS Comp Bio) — Snapshot ensemble, wild↔lab transfer
- Fonseca et al. / VocalMat — 12,954 labeled USV spectrograms (free training data)
- Zala et al. (2020, Frontiers in Zoology) — Wild mouse USV modulation by social context
- Holy & Guo (2005) — Original USV syllable classification (may impose artificial boundaries)
