"""USV bout data preparation pipeline (v2 architecture).

Public API for the data pipeline that converts CNN detection results
into PyTorch-ready datasets for autoregressive transformer training.
"""

from usv_language.data.bout_extractor import Bout, BoutExtractionConfig, BoutExtractor
from usv_language.data.dataset import (
    AugmentationConfig,
    BucketedBatchSampler,
    TransformerDataConfig,
    USVBoutDataset,
    split_recordings,
)
from usv_language.data.normalization import (
    NormalizationStats,
    compute_normalization_stats,
    load_normalization_stats,
    normalize,
    save_normalization_stats,
)


def __getattr__(name: str):
    """Lazy-load spectrogram symbols to avoid heavy deps (soundfile, scipy,
    src.usv_spectrogram) when only the training/dataset path is needed."""
    if name in ("BoutSpectrogramConfig", "compute_bout_spectrogram"):
        from usv_language.data.spectrogram import (
            BoutSpectrogramConfig,
            compute_bout_spectrogram,
        )
        return {"BoutSpectrogramConfig": BoutSpectrogramConfig,
                "compute_bout_spectrogram": compute_bout_spectrogram}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Bout",
    "BoutExtractionConfig",
    "BoutExtractor",
    "BoutSpectrogramConfig",
    "compute_bout_spectrogram",
    "NormalizationStats",
    "compute_normalization_stats",
    "normalize",
    "save_normalization_stats",
    "load_normalization_stats",
    "TransformerDataConfig",
    "USVBoutDataset",
    "BucketedBatchSampler",
    "AugmentationConfig",
    "split_recordings",
]
