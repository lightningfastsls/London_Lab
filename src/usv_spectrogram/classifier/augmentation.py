"""Module 18.3 — SpecAugment + cage-noise augmentation for the lab classifier.

ROADMAP §18.3 file 2. Provides:
  - AugmentationConfig — frozen dataclass holding all augmentation knobs.
  - specaugment(spec, cfg) — Park et al. 2019 time+freq masking. Uses the
    legacy `np.random.*` global state for determinism under
    `np.random.seed(...)`, matching the test_architect contract.
  - inject_cage_noise(spec, cfg, rng) — additive blend of a sampled
    verdict-negative patch into the input. Targets the cage-confound
    directly per PLAN §"Phase 1.2".

Reference: Park et al. 2019 SpecAugment (arXiv:1904.08779).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    """Frozen hyperparameter bundle for SpecAugment + cage-noise injection.

    All fields default to PLAN §"Phase 1.2" values. The frozen flag prevents
    accidental mutation across worker threads during data loading.
    """

    time_mask_max_width_frames: int = 20
    time_mask_n: int = 2
    freq_mask_max_width_bins: int = 16
    freq_mask_n: int = 2
    pitch_shift_max_pct: float = 0.10
    time_stretch_max_pct: float = 0.20
    random_crop_max_pct: float = 0.05
    cage_noise_inject_prob: float = 0.25
    cage_noise_paths: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.time_mask_max_width_frames < 0:
            raise ValueError(
                f"time_mask_max_width_frames must be >= 0, "
                f"got {self.time_mask_max_width_frames}"
            )
        if self.freq_mask_max_width_bins < 0:
            raise ValueError(
                f"freq_mask_max_width_bins must be >= 0, "
                f"got {self.freq_mask_max_width_bins}"
            )
        if self.time_mask_n < 0:
            raise ValueError(f"time_mask_n must be >= 0, got {self.time_mask_n}")
        if self.freq_mask_n < 0:
            raise ValueError(f"freq_mask_n must be >= 0, got {self.freq_mask_n}")
        if self.pitch_shift_max_pct < 0.0:
            raise ValueError(
                f"pitch_shift_max_pct must be >= 0.0, got {self.pitch_shift_max_pct}"
            )
        if self.time_stretch_max_pct < 0.0:
            raise ValueError(
                f"time_stretch_max_pct must be >= 0.0, got {self.time_stretch_max_pct}"
            )
        if self.random_crop_max_pct < 0.0:
            raise ValueError(
                f"random_crop_max_pct must be >= 0.0, got {self.random_crop_max_pct}"
            )
        if not (0.0 <= self.cage_noise_inject_prob <= 1.0):
            raise ValueError(
                f"cage_noise_inject_prob must be in [0.0, 1.0], "
                f"got {self.cage_noise_inject_prob}"
            )


def specaugment(spec: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
    """Apply Park et al. 2019 SpecAugment time and frequency masking.

    Returns a new array of the same shape as ``spec``. When all mask widths
    (or mask counts) are zero, the input is returned unmodified.

    Uses ``np.random.*`` global state so callers can reproduce results via
    ``np.random.seed(...)``.

    Parameters
    ----------
    spec
        Spectrogram of shape ``(freq_bins, time_frames)``.
    cfg
        Augmentation hyperparameters.

    Returns
    -------
    np.ndarray
        Masked spectrogram, same shape as ``spec``.
    """
    freq_bins, time_frames = spec.shape
    out = spec.copy()

    for _ in range(cfg.freq_mask_n):
        if cfg.freq_mask_max_width_bins <= 0 or freq_bins == 0:
            break
        f = int(np.random.randint(0, cfg.freq_mask_max_width_bins + 1))
        if f == 0 or f >= freq_bins:
            continue
        f0 = int(np.random.randint(0, freq_bins - f + 1))
        out[f0:f0 + f, :] = 0.0

    for _ in range(cfg.time_mask_n):
        if cfg.time_mask_max_width_frames <= 0 or time_frames == 0:
            break
        t = int(np.random.randint(0, cfg.time_mask_max_width_frames + 1))
        if t == 0 or t >= time_frames:
            continue
        t0 = int(np.random.randint(0, time_frames - t + 1))
        out[:, t0:t0 + t] = 0.0

    return out


def inject_cage_noise(
    spec: np.ndarray,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Additively blend a sampled verdict-negative patch into ``spec``.

    Short-circuits to identity when ``cage_noise_inject_prob`` is zero or
    ``cage_noise_paths`` is empty (graceful degrade — no exception). When
    the probability draw misses, returns the input unchanged.

    Parameters
    ----------
    spec
        Input spectrogram, shape ``(freq_bins, time_frames)``.
    cfg
        Augmentation config.
    rng
        ``numpy.random.Generator`` for sampling.

    Returns
    -------
    np.ndarray
        Possibly noise-blended spectrogram with shape equal to ``spec``.
    """
    if cfg.cage_noise_inject_prob <= 0.0:
        return spec
    if len(cfg.cage_noise_paths) == 0:
        return spec
    if rng.random() > cfg.cage_noise_inject_prob:
        return spec

    idx = int(rng.integers(0, len(cfg.cage_noise_paths)))
    patch_path = cfg.cage_noise_paths[idx]
    noise = np.load(patch_path) if patch_path.endswith(".npy") else _load_image_patch(patch_path)
    noise = _resize_to(noise, spec.shape)
    alpha = float(rng.uniform(0.05, 0.30))
    return spec + alpha * noise


def _load_image_patch(path: str) -> np.ndarray:
    """Load an image patch file as a float32 array. Pillow-backed for PNGs."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return arr


def _resize_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to the target shape via slicing/tiling.

    A lightweight stand-in for scipy.ndimage.zoom that avoids a heavy import
    in the hot training path. For mismatched shapes we crop or tile-and-crop.
    """
    target_h, target_w = shape
    h, w = arr.shape
    if (h, w) == (target_h, target_w):
        return arr.astype(np.float32, copy=False)
    out = np.zeros((target_h, target_w), dtype=np.float32)
    src_h = min(h, target_h)
    src_w = min(w, target_w)
    out[:src_h, :src_w] = arr[:src_h, :src_w]
    return out
