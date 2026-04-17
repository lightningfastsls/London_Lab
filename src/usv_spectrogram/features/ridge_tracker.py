"""Viterbi-style dynamic-programming ridge tracker for USV spectrograms.

Extracts the MAP sequence of frequency-bin indices through a magnitude
spectrogram under the objective

    score = sum_t  magnitude[f_t, t]  -  lambda * sum_t |f_t - f_{t-1}|

subject to a hard local-jump constraint ``|f_t - f_{t-1}| <= max_jump_bins``.
The solution gives FM (pitch) and AM (amplitude) trajectories that are robust
to harmonic jumps and noise transients in a way that naive argmax is not.

Part of Phase 17 SIS-benchmark shared infrastructure (ROADMAP §17.3). Feeds
17.4 (iMSA pitch-jump classifier) and 17.5 (Oren 80-D vectorisation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeConfig:
    """DP ridge-tracker parameters (Pattern 1: frozen dataclass).

    Parameters
    ----------
    transition_penalty:
        ``lambda`` — cost added per bin of frequency jump between columns.
        Must be >= 0. A value of 0 reduces the tracker to per-column argmax.
    max_jump_bins:
        ``W`` — hard window radius on the Viterbi transition. The path at
        column t can only come from bins ``[f - W, f + W]`` at column t-1.
        Must be >= 1.
    silence_threshold:
        Columns whose per-column max magnitude is strictly below this value
        are treated as silent; the tracker emits NaN for FM and AM there.
    """

    transition_penalty: float = 0.1
    max_jump_bins: int = 10
    silence_threshold: float = 1e-6

    def __post_init__(self) -> None:
        if self.transition_penalty < 0:
            raise ValueError("transition_penalty must be >= 0")
        if self.max_jump_bins < 1:
            raise ValueError("max_jump_bins must be >= 1")


def track_ridge(
    magnitude: np.ndarray,
    freqs_hz: np.ndarray,
    cfg: RidgeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fm_hz, am)`` — shape ``(n_time_cols,)`` each.

    Pipeline
    --------
    1. Silence mask: ``is_silent[t] = magnitude[:, t].max() < silence_threshold``.
    2. Split active columns into contiguous non-silent runs (silent columns
       break the DP chain — each run is solved independently, seeded from
       argmax at its first column).
    3. Per-run Viterbi forward pass in O(F * W * run_len):
           cur_cost[f, t] = magnitude[f, t]
                          + max_{shift in [-W, +W]}
                              (cur_cost[f + shift, t-1] - lambda * |shift|)
       with back-pointer ``argmax_shift`` stored for traceback.
    4. Back-trace from ``argmax(cur_cost[:, r_end - 1])`` to recover
       ``ridge_idx`` for the run.
    5. Assemble outputs: ``fm_hz[active] = freqs_hz[ridge_idx[active]]``,
       ``am[active] = magnitude[ridge_idx[active], active]``. Silent columns
       remain NaN.

    Parameters
    ----------
    magnitude:
        Linear magnitude spectrogram, shape ``(n_freq_bins, n_time_cols)``,
        non-negative. Typically produced by :func:`prefilter_spectrogram`.
    freqs_hz:
        1-D array, shape ``(n_freq_bins,)``, giving the frequency of each row.
    cfg:
        :class:`RidgeConfig` with validated parameters.

    Returns
    -------
    fm_hz:
        Frequency trajectory in Hz, shape ``(n_time_cols,)``, with NaN on
        silent columns.
    am:
        Amplitude trajectory (= magnitude at the ridge bin), shape
        ``(n_time_cols,)``, with NaN on silent columns.
    """
    n_bins, n_cols = magnitude.shape
    fm_hz = np.full(n_cols, np.nan, dtype=float)
    am = np.full(n_cols, np.nan, dtype=float)

    col_max = magnitude.max(axis=0)
    is_silent = col_max < cfg.silence_threshold
    if is_silent.all():
        return fm_hz, am

    ridge_idx = np.empty(n_cols, dtype=np.int64)

    for r_start, r_end in _non_silent_runs(is_silent):
        _track_run(magnitude, r_start, r_end, cfg, ridge_idx)

    active = ~is_silent
    active_cols = np.nonzero(active)[0]
    active_bins = ridge_idx[active_cols]
    fm_hz[active_cols] = freqs_hz[active_bins]
    am[active_cols] = magnitude[active_bins, active_cols]
    return fm_hz, am


def _non_silent_runs(is_silent: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open ``[start, end)`` intervals of contiguous non-silent columns."""
    runs: list[tuple[int, int]] = []
    n = is_silent.shape[0]
    start: int | None = None
    for t in range(n):
        if not is_silent[t] and start is None:
            start = t
        elif is_silent[t] and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, n))
    return runs


def _track_run(
    magnitude: np.ndarray,
    r_start: int,
    r_end: int,
    cfg: RidgeConfig,
    ridge_idx: np.ndarray,
) -> None:
    """Forward-pass + back-trace for one contiguous non-silent run.

    Mutates ``ridge_idx[r_start:r_end]`` in place with the MAP bin indices.
    """
    n_bins = magnitude.shape[0]
    run_len = r_end - r_start
    penalty = cfg.transition_penalty
    w = cfg.max_jump_bins

    # Seed the run from the first column's raw reward (no prior path).
    cur_cost = magnitude[:, r_start].astype(float, copy=True)

    if run_len == 1:
        ridge_idx[r_start] = int(np.argmax(cur_cost))
        return

    # backtrace[f, local_t] = source bin g at column (r_start + local_t - 1)
    # for the best path arriving at bin f at column (r_start + local_t).
    # local_t ranges over [1, run_len); column 0 has no back-pointer.
    backtrace = np.zeros((n_bins, run_len), dtype=np.int64)
    bin_indices = np.arange(n_bins, dtype=np.int64)

    for local_t in range(1, run_len):
        t = r_start + local_t
        best = np.full(n_bins, -np.inf, dtype=float)
        best_src = np.zeros(n_bins, dtype=np.int64)

        for shift in range(-w, w + 1):
            # Candidate: the path ending at bin f at column t came from bin
            # g = f + shift at column t-1.  In-bounds f range ensures 0 <= g < F.
            f_lo = max(0, -shift)
            f_hi = min(n_bins, n_bins - shift)
            if f_lo >= f_hi:
                continue
            candidate = cur_cost[f_lo + shift : f_hi + shift] - penalty * abs(shift)
            slice_best = best[f_lo:f_hi]
            improved = candidate > slice_best
            if improved.any():
                slice_best[improved] = candidate[improved]
                best[f_lo:f_hi] = slice_best
                best_src[f_lo:f_hi] = np.where(
                    improved,
                    bin_indices[f_lo:f_hi] + shift,
                    best_src[f_lo:f_hi],
                )

        cur_cost = magnitude[:, t] + best
        backtrace[:, local_t] = best_src

    # Back-trace from the best terminal bin.
    end_bin = int(np.argmax(cur_cost))
    ridge_idx[r_end - 1] = end_bin
    for local_t in range(run_len - 1, 0, -1):
        end_bin = int(backtrace[end_bin, local_t])
        ridge_idx[r_start + local_t - 1] = end_bin
