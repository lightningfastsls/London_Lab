"""Lab-cleaned USV syllable classifier (Phase 18).

This package hosts the downstream lab classifier (Modules 18.2b - 18.3).
The production detection pipeline (`scripts/run_batch_detection.py`,
`app/core/sliding_inference.py`, `postprocessing/`) is intentionally
untouched.

Archived 2026-05-28
-------------------
Module 18.1 (the cleaning-validation gate) — ``cleaning_pipeline.py`` and
``diagnostics.py`` — was archived to
``archive/cleaning_legacy/stack1/`` because its analysis family
(`CleaningConfig` / `clean_spectrogram` + the four-diagnostic gate) hit
a dead end (Module 18.4 DANN shelved, VocalMat re-render verified dead).
Trained ``lab_classifier_v1`` weights remain usable for inference via
``build_resnet18_classifier``. See
``docs/modules/cleaning-subsystems.md`` for the canonical cleaning
pipeline (Stack 4: DeepSqueak focus-STFT port).

Constants
---------
- ``TARGET_SAMPLE_RATE_HZ``: 250 kHz. The classifier pipeline is
  VocalMat-aligned (NOT the canonical 300 kHz corpus rate). Resampling is
  performed by ``classifier/resample.py``.
- ``RESAMPLE_UP`` / ``RESAMPLE_DOWN``: rational-resampling factors used
  with ``scipy.signal.resample_poly`` (300 kHz x 5 / 6 = 250 kHz).
"""
from __future__ import annotations

__version__ = "0.1.0"

TARGET_SAMPLE_RATE_HZ: int = 250_000
RESAMPLE_UP: int = 5
RESAMPLE_DOWN: int = 6

from .dataset import GRIMSLEY_12_CLASSES, DatasetSplit, build_stratified_split
from .resample import SOURCE_SAMPLE_RATE_HZ, resample_to_vocalmat

# Module 18.3 public API — model factory, augmentation, loss, training loop.
from .model import NUM_CLASSES, build_resnet18_classifier
from .augmentation import AugmentationConfig, inject_cage_noise, specaugment
from .losses import focal_loss
from .training import TrainingConfig, train_classifier

__all__ = [
    "TARGET_SAMPLE_RATE_HZ",
    "SOURCE_SAMPLE_RATE_HZ",
    "RESAMPLE_UP",
    "RESAMPLE_DOWN",
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
]
