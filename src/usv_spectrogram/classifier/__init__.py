"""Lab-cleaned USV syllable classifier (Phase 18).

This package hosts the cleaning-validation gate (Module 18.1) and the
downstream classifier (Modules 18.2-18.5). The production detection
pipeline (`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`,
`postprocessing/`) is intentionally untouched.

Constants
---------
- ``TARGET_SAMPLE_RATE_HZ``: 250 kHz. The classifier pipeline is
  VocalMat-aligned (NOT the canonical 300 kHz corpus rate). Resampling is
  performed by ``classifier/resample.py`` (Module 18.2 owns that file;
  the constant lives here for Module 18.1's spectrogram-level configuration).
- ``RESAMPLE_UP`` / ``RESAMPLE_DOWN``: rational-resampling factors used with
  ``scipy.signal.resample_poly`` (300 kHz x 5 / 6 = 250 kHz).

Layer constants are duplicated in ``CleaningConfig`` for the sample-rate
validation branch.
"""
from __future__ import annotations

__version__ = "0.1.0"

# 18.2 will own resample.py; keep canonical constants here so 18.1 can
# validate against them and a future 18.2 import will not collide.
TARGET_SAMPLE_RATE_HZ: int = 250_000
RESAMPLE_UP: int = 5
RESAMPLE_DOWN: int = 6

# Module 18.1 + 18.2b public API. Re-exported so external callers (Modules
# 18.3+ downstream) can ``from usv_spectrogram.classifier import ...``
# without knowing the submodule layout.
from .cleaning_pipeline import CleaningConfig, clean_spectrogram
from .dataset import GRIMSLEY_12_CLASSES, DatasetSplit, build_stratified_split
from .diagnostics import (
    DiagnosticResult,
    notch_injection_test,
    per_band_cohens_d,
    knn_same_cohort_rate,
    raw_pixel_pca_d,
    train_diagnostic_vae,
)
from .resample import SOURCE_SAMPLE_RATE_HZ, resample_to_vocalmat

# Module 18.3 public API — model factory, augmentation, loss, training loop.
from .model import NUM_CLASSES, build_resnet18_classifier
from .augmentation import AugmentationConfig, inject_cage_noise, specaugment
from .losses import focal_loss
from .training import TrainingConfig, train_classifier

# Module 18.4 public API — DANN cage-adversarial components + cage probe.
from .dann import (
    DomainHead,
    GradientReversal,
    LambdaSchedule,
    ResNet18DANN,
    grad_reverse,
)
from .cage_probe import linear_cage_probe

__all__ = [
    "TARGET_SAMPLE_RATE_HZ",
    "SOURCE_SAMPLE_RATE_HZ",
    "RESAMPLE_UP",
    "RESAMPLE_DOWN",
    "CleaningConfig",
    "clean_spectrogram",
    "DiagnosticResult",
    "notch_injection_test",
    "per_band_cohens_d",
    "knn_same_cohort_rate",
    "raw_pixel_pca_d",
    "train_diagnostic_vae",
    "resample_to_vocalmat",
    "GRIMSLEY_12_CLASSES",
    "DatasetSplit",
    "build_stratified_split",
    # Module 18.3
    "NUM_CLASSES",
    "build_resnet18_classifier",
    "AugmentationConfig",
    "inject_cage_noise",
    "specaugment",
    "focal_loss",
    "TrainingConfig",
    "train_classifier",
    # Module 18.4
    "GradientReversal",
    "grad_reverse",
    "DomainHead",
    "LambdaSchedule",
    "ResNet18DANN",
    "linear_cage_probe",
]
