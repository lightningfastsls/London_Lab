"""Pre-CNN spectrogram denoising primitives.

Exposes ``subtract_temporal_baseline`` — per-frequency-bin temporal-noise-floor
subtraction in linear magnitude. Two methods are supported:

- ``"percentile"``: subtract a single per-bin percentile (default 10th) of
  magnitude over time. The textbook Boll 1979 formulation. Works perfectly on
  *stationary* equipment bands. For amplitude-modulated bands (where the
  band's brightness varies across the chunk), only the band's *floor* gets
  stripped — the variance survives, leaving residual horizontal-line texture
  that the wild-trained CNN can still misread as USVs.

- ``"median_envelope"``: per-bin sliding median filter over a kernel wider
  than the longest expected USV. Captures the band's *time-varying envelope*
  instead of just its floor, so amplitude-modulated bands get tracked and
  removed. USV bursts are robust to median filtering as long as the kernel
  is wider than the burst duration — the median of a window containing one
  bright burst is unaffected by that burst (it's the lower-half value).

Design notes
------------
- Operates in **linear magnitude**, not dB. Caller is responsible for the
  dB↔linear conversion. Subtraction in dB is mathematically wrong; subtraction
  in linear magnitude is the textbook formulation (Boll 1979).
- Baseline is computed **per chunk**, not per recording. Matches the existing
  2-second chunked pipeline architecture and keeps the function stateless.
- USVs are 10-300 ms; the default ``median_envelope`` kernel is 0.5 s
  (>1.5× longest USV) so a single burst contributes far fewer than half the
  samples in its smoothing window — the median ignores it. Stationary or
  slowly-modulating bands fill ≥50% of the kernel — captured.

Validated visually against
``results/batch_lab_131204_full/spectral_subtraction_preview/`` (percentile
method) and against ``results/batch_lab_131204_subtracted_pilot/`` paired
comparisons.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from ...corpus import SAMPLE_RATE_HZ, STFT_HOP

DEFAULT_BASELINE_PERCENTILE = 10.0
DEFAULT_EPSILON = 1e-10

# The kernel's correctness depends on being wider than the longest USV
# (~300 ms) so a USV burst can never dominate the per-bin median window.
# Express the invariant in seconds, derive frames from canonical corpus
# values — that way, any future change to STFT_HOP or SAMPLE_RATE_HZ in
# corpus.py automatically updates the frame count without silently
# breaking the kernel-vs-USV-width invariant.
DEFAULT_ENVELOPE_KERNEL_SEC = 0.5
DEFAULT_ENVELOPE_KERNEL_FRAMES = int(
    round(DEFAULT_ENVELOPE_KERNEL_SEC * SAMPLE_RATE_HZ / STFT_HOP)
)


def subtract_temporal_baseline(
    spec_linear: np.ndarray,
    method: str = "percentile",
    percentile: float = DEFAULT_BASELINE_PERCENTILE,
    envelope_kernel_frames: int = DEFAULT_ENVELOPE_KERNEL_FRAMES,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Subtract a per-frequency-bin temporal noise floor from a magnitude spectrogram.

    Parameters
    ----------
    spec_linear:
        Linear-magnitude spectrogram, shape ``(n_freq, n_time)``.
    method:
        ``"percentile"`` (default, Boll 1979 floor subtraction) or
        ``"median_envelope"`` (per-bin sliding median, captures slow band
        amplitude modulation).
    percentile:
        Used when ``method="percentile"`` — temporal percentile per bin used
        as the noise floor (default 10).
    envelope_kernel_frames:
        Used when ``method="median_envelope"`` — width of the per-bin temporal
        median filter, in frames. Default 1170 (~0.5 s at hop=128, sr=300 kHz)
        which is wider than any USV (longest ~300 ms = ~700 frames) so USV
        bursts cannot dominate the median.
    epsilon:
        Floor applied after subtraction so downstream ``log`` is safe.

    Returns
    -------
    cleaned:
        Linear-magnitude spectrogram with the per-bin baseline removed,
        floored at ``epsilon``. Same shape and dtype as ``spec_linear``.
    """
    if spec_linear.ndim != 2:
        raise ValueError(
            f"spec_linear must be 2D (n_freq, n_time); got shape {spec_linear.shape}"
        )
    if spec_linear.shape[1] == 0:
        return spec_linear.copy()

    if method == "percentile":
        baseline = np.percentile(spec_linear, percentile, axis=1, keepdims=True)
    elif method == "median_envelope":
        # Median filter only along the time axis (size=1 on freq axis).
        # 'reflect' boundary mode keeps the envelope smooth at chunk edges
        # without leaking energy from a wrap-around opposite edge.
        kernel = max(3, int(envelope_kernel_frames) | 1)  # force odd, ≥3
        # Don't let the kernel exceed the number of time frames available.
        kernel = min(kernel, spec_linear.shape[1] // 2 * 2 - 1) if spec_linear.shape[1] >= 5 else 3
        if kernel < 3:
            kernel = 3
        baseline = median_filter(spec_linear, size=(1, kernel), mode="reflect")
    else:
        raise ValueError(
            f"Unknown method {method!r}; expected 'percentile' or 'median_envelope'."
        )

    cleaned = np.maximum(spec_linear - baseline, epsilon)
    return cleaned
