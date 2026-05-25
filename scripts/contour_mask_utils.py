"""Contour-mask application primitives for Phase 3 of the
contour-masked VAE pipeline.

Given a 2D power spectrogram ``S_pow`` (shape ``(F, T)``) and per-time-bin
contour data (ridge frequencies + tonality), these utilities return a
*masked* copy of the spectrogram where energy near the tonal ridge is
preserved and everything else is zero (hard mask) or attenuated
(Gaussian soft mask).

Two mask flavours:

  - ``apply_hard_bandwidth_mask``: keeps S_pow[f in (f_ridge +/- bandwidth), t]
    for every column ``t`` whose contour bin has ``tonality >= threshold``.
    Columns failing the threshold OR with no contour bin are zeroed out
    entirely.

  - ``apply_soft_gaussian_mask``: multiplies S_pow by a column-wise Gaussian
    weight ``exp(-0.5 * ((f - f_ridge[t]) / sigma)^2)``, again only on
    columns whose contour bin has ``tonality >= threshold``. Other columns
    are zeroed.

Both functions are vectorised along the time axis (no per-column Python
loop): we build a sparse ``(F, T)`` weight array via NumPy broadcasting
and multiply once.

Frequencies are kHz throughout (matching the contour parquet schema).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _validate_inputs(
    S_pow: np.ndarray,
    contour_t_bins: Sequence[int],
    contour_freqs_kHz: Sequence[float],
    contour_tonalities: Sequence[float],
    freqs_kHz_axis: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coerce, validate, and return clean numpy arrays for the inputs.

    Raises ``ValueError`` if S_pow is not 2D, if the contour-arrays have
    mismatched lengths, or if the frequency axis does not match S_pow's
    first dimension.
    """
    if S_pow.ndim != 2:
        raise ValueError(
            f"S_pow must be 2D (F, T); got shape {S_pow.shape}"
        )
    freqs_kHz_axis = np.asarray(freqs_kHz_axis, dtype=np.float64)
    if freqs_kHz_axis.shape[0] != S_pow.shape[0]:
        raise ValueError(
            f"freqs_kHz_axis length {freqs_kHz_axis.shape[0]} != S_pow.shape[0] "
            f"{S_pow.shape[0]}"
        )

    t_bins = np.asarray(contour_t_bins, dtype=np.int64)
    f_ridge = np.asarray(contour_freqs_kHz, dtype=np.float64)
    tonalities = np.asarray(contour_tonalities, dtype=np.float64)

    if not (t_bins.shape == f_ridge.shape == tonalities.shape):
        raise ValueError(
            "contour_t_bins, contour_freqs_kHz, and contour_tonalities must "
            f"have identical shape; got {t_bins.shape}, {f_ridge.shape}, "
            f"{tonalities.shape}"
        )
    return freqs_kHz_axis, t_bins, f_ridge, tonalities


def _qualifying_contour_per_column(
    t_bins: np.ndarray,
    f_ridge: np.ndarray,
    tonalities: np.ndarray,
    tonality_threshold: float,
    n_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (active_columns, f_ridge_per_column).

    For each column index ``t``, picks the contour bin with the *highest*
    tonality that ``>= tonality_threshold`` AND has ``0 <= t_bin < n_time``.
    Columns with no qualifying contour bin are returned as inactive
    (``active_columns[t] == False``); their ``f_ridge_per_column[t]``
    value is undefined (NaN) and should not be read.
    """
    active = np.zeros(n_time, dtype=bool)
    f_per_col = np.full(n_time, np.nan, dtype=np.float64)
    best_ton = np.full(n_time, -np.inf, dtype=np.float64)

    if t_bins.size == 0:
        return active, f_per_col

    in_range = (t_bins >= 0) & (t_bins < n_time)
    pass_thr = tonalities >= tonality_threshold
    keep = in_range & pass_thr

    # Iterate only over qualifying contour rows. This is O(num_contour_bins)
    # and does not loop over the (F, T) grid.
    for idx in np.flatnonzero(keep):
        t = int(t_bins[idx])
        ton = float(tonalities[idx])
        if ton > best_ton[t]:
            best_ton[t] = ton
            f_per_col[t] = f_ridge[idx]
            active[t] = True

    return active, f_per_col


def apply_hard_bandwidth_mask(
    S_pow: np.ndarray,
    contour_t_bins: Sequence[int],
    contour_freqs_kHz: Sequence[float],
    contour_tonalities: Sequence[float],
    freqs_kHz_axis: Sequence[float],
    bandwidth_kHz: float,
    tonality_threshold: float,
) -> np.ndarray:
    """Apply a hard +/- ``bandwidth_kHz`` mask centred on the ridge.

    For every time column ``t`` whose qualifying contour bin has
    ``tonality >= tonality_threshold``:
      - keep S_pow[freq in (f_ridge - bandwidth, f_ridge + bandwidth), t]
      - zero everything else in that column.

    Columns with no contour bin or with tonality below threshold are
    fully zeroed.

    Returns
    -------
    np.ndarray
        Same shape and dtype as ``S_pow``. A fresh array (no in-place
        modification of ``S_pow``).
    """
    freqs_axis, t_bins, f_ridge, ton = _validate_inputs(
        S_pow,
        contour_t_bins,
        contour_freqs_kHz,
        contour_tonalities,
        freqs_kHz_axis,
    )
    n_freq, n_time = S_pow.shape

    active, f_per_col = _qualifying_contour_per_column(
        t_bins, f_ridge, ton, tonality_threshold, n_time
    )

    # Build (F, T) boolean keep-mask via broadcasting. Inactive columns
    # set f_per_col = NaN so the |...| comparison evaluates to False
    # there — guaranteeing those columns end up fully zero.
    freqs_col = freqs_axis[:, None]                # (F, 1)
    f_per_col_b = f_per_col[None, :]              # (1, T)
    with np.errstate(invalid="ignore"):
        within_band = np.abs(freqs_col - f_per_col_b) <= bandwidth_kHz
    within_band &= active[None, :]

    out = np.zeros_like(S_pow)
    out[within_band] = S_pow[within_band]
    return out


def apply_soft_gaussian_mask(
    S_pow: np.ndarray,
    contour_t_bins: Sequence[int],
    contour_freqs_kHz: Sequence[float],
    contour_tonalities: Sequence[float],
    freqs_kHz_axis: Sequence[float],
    sigma_kHz: float,
    tonality_threshold: float,
) -> np.ndarray:
    """Apply a multiplicative Gaussian mask centred on the ridge.

    For every time column ``t`` whose qualifying contour bin has
    ``tonality >= tonality_threshold``, multiplies S_pow[:, t] by
    ``exp(-0.5 * ((freqs_kHz_axis - f_ridge[t]) / sigma_kHz) ** 2)``.

    Columns with no contour bin or with tonality below threshold are
    fully zeroed.

    Returns
    -------
    np.ndarray
        Same shape as ``S_pow``, dtype promoted as needed by the multiply
        (typically float64 or float32 depending on inputs). A fresh
        array.
    """
    if sigma_kHz <= 0:
        raise ValueError(f"sigma_kHz must be > 0; got {sigma_kHz}")

    freqs_axis, t_bins, f_ridge, ton = _validate_inputs(
        S_pow,
        contour_t_bins,
        contour_freqs_kHz,
        contour_tonalities,
        freqs_kHz_axis,
    )
    n_freq, n_time = S_pow.shape

    active, f_per_col = _qualifying_contour_per_column(
        t_bins, f_ridge, ton, tonality_threshold, n_time
    )

    # Replace NaN entries (inactive columns) with 0.0 — they get zeroed
    # by the active mask below, so the value doesn't matter as long as
    # the math is finite.
    f_per_col_safe = np.where(active, f_per_col, 0.0)[None, :]  # (1, T)
    freqs_col = freqs_axis[:, None]                              # (F, 1)
    z = (freqs_col - f_per_col_safe) / sigma_kHz
    weight = np.exp(-0.5 * (z ** 2))                              # (F, T)
    weight = weight * active[None, :]                             # zero inactive cols

    return S_pow * weight
