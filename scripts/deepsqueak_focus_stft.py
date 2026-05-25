"""Faithful Python port of DeepSqueak's CreateFocusSpectrogram + CalculateStats.

Source files (snapshots in reference/deepsqueak_source/):
    CreateFocusSpectrogram.m   — github.com/DrCoffey/DeepSqueak/Functions/
    CalculateStats.m           — same repo

Why this exists:
    The earlier Python port (scripts/ridge_tracker.py) ran a single global STFT
    at the canonical hop=128 grid and indexed into it per call. DeepSqueak does
    NOT do that — it computes a *per-call* focus STFT whose window size is
    adaptive to the call's own duration and frequency range. That produces a
    coarser, call-specific time grid (effective hop typically 2-3x the canonical
    hop), which is the root cause of the prior 2.39x density discrepancy.

This module replicates the algorithm line-by-line. Every non-trivial function
carries a MATLAB-line reference comment of the form ``DS:CalculateStats.m:L8``.

Output convention:
    Contour bins are emitted in *physical units* (time_s, frequency_kHz). The
    downstream extract_contours_python.py wrapper quantizes time_s onto the
    canonical hop=128 grid for the time_bin_index parquet column, matching the
    MATLAB export's behavior at deepsqueak_export_contours.m line 145-148.

Dependencies: numpy, scipy, statsmodels (for robust LOWESS / rlowess).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from statsmodels.nonparametric.smoothers_lowess import lowess


# ---------------------------------------------------------------------------
# DeepSqueak defaults — sourced from CalculateStats.m comments + the MATLAB
# wrapper deepsqueak_export_contours.m lines 33-34.
# ---------------------------------------------------------------------------
DS_ENTROPY_THRESHOLD: float = 0.215
DS_AMPLITUDE_THRESHOLD: float = 0.825
DS_NOVERLAP_FRACTION: float = 0.5      # CreateFocusSpectrogram.m:L11
DS_ENTROPY_SMOOTH_SPAN: float = 0.1    # CalculateStats.m:L11
DS_RIDGE_SMOOTH_SPAN: float = 0.025    # CalculateStats.m:L54
DS_RLOWESS_ITERATIONS: int = 5         # MATLAB default for 'rlowess'
DS_LOWERING_FACTOR: float = 1.1        # CalculateStats.m:L35
DS_MIN_RIDGE_POINTS: int = 5           # CalculateStats.m:L30
DS_MAX_LOWERING_ITERS: int = 10        # CalculateStats.m:L42

# Defensive bounds — NOT in DeepSqueak source. These guard against pathological
# CSV inputs that would otherwise drive the adaptive window formula into
# multi-gigabyte STFT allocations.
#
# Root cause: optimalWindow = sqrt(duration / (2000 * freq_range_kHz)) * 1.5.
# When freq_range_kHz is very small (e.g. 0.001 kHz from a clamped or near-
# degenerate detection box), the window grows without bound. With duration
# 50 ms and freq_range 0.001 kHz, optimalWindow is 0.24 s -> 71,000-sample
# window at SR=300 kHz. For a 2 s lab clip this leads to an internal STFT
# allocation in the hundreds of GB.
#
# We require freq_range >= 1 kHz (calls narrower than that are below this
# corpus's freq resolution and almost always artifacts of the detection
# pipeline) and cap windowsize at 4096 samples (~13.6 ms at SR=300 kHz —
# wider than the longest sensible USV-grade focus window).
MIN_FREQ_RANGE_KHZ: float = 1.0
MAX_WINDOWSIZE_SAMPLES: int = 4096


@dataclass(frozen=True)
class CallBox:
    """A DeepSqueak Box: [time_s, freq_kHz, duration_s, freq_range_kHz].

    Matches MATLAB ``Calls.Box(i, :)``. Indexing in the field names follows
    MATLAB's 1-based convention used throughout the source files; the actual
    Python attribute access is by name to avoid off-by-one errors.
    """

    time_start_s: float          # Box(1) — call start time, seconds
    freq_start_kHz: float        # Box(2) — call low frequency, kHz
    duration_s: float            # Box(3) — call duration, seconds
    freq_range_kHz: float        # Box(4) — call frequency range, kHz

    @property
    def time_end_s(self) -> float:
        return self.time_start_s + self.duration_s

    @property
    def freq_end_kHz(self) -> float:
        return self.freq_start_kHz + self.freq_range_kHz


@dataclass(frozen=True)
class FocusSpectrogramResult:
    """Output bundle from create_focus_spectrogram.

    Attributes
    ----------
    I : ndarray, shape (n_freq_bins, n_time_bins)
        Magnitude STFT of the cropped audio segment, cropped in freq to the
        Box's frequency range.
    windowsize : int
        Hamming window length in samples (DS:CreateFocusSpectrogram.m:L29).
    noverlap : int
        Overlap between consecutive frames in samples (DS:L30).
    nfft : int
        FFT length in samples (DS:L31).
    fr_hz : ndarray
        Full pre-crop frequency vector (Hz) returned by the STFT.
    ti_s : ndarray
        Pre-crop time vector (seconds, relative to start of audio segment).
    freq_lo_idx : int
        Index into fr_hz of the first frequency row kept after the freq crop.
    freq_hi_idx : int
        Index into fr_hz of the last frequency row kept (inclusive).
    """

    I: np.ndarray
    windowsize: int
    noverlap: int
    nfft: int
    fr_hz: np.ndarray
    ti_s: np.ndarray
    freq_lo_idx: int
    freq_hi_idx: int


# ---------------------------------------------------------------------------
# CreateFocusSpectrogram port
# ---------------------------------------------------------------------------
def compute_optimal_window(duration_s: float, freq_range_kHz: float,
                           noverlap_fraction: float = DS_NOVERLAP_FRACTION
                           ) -> tuple[float, float, float]:
    """DS:CreateFocusSpectrogram.m:L9-L18.

    Adaptive STFT window in *seconds*. The formula is:

        optimalWindow = sqrt(duration / (2000 * freq_range_kHz))
        optimalWindow = optimalWindow + optimalWindow * noverlap   # × 1.5

    The factor of 2000 comes from MATLAB's unit convention. ``optimalWindow``
    is then used as both ``windowsize`` AND ``nfft``; ``overlap`` is
    ``optimalWindow * noverlap_fraction``.

    Returns
    -------
    (windowsize_s, overlap_s, nfft_s)
    """
    yRange = float(freq_range_kHz)  # DS:L9
    xRange = float(duration_s)      # DS:L10
    optimal = float(np.sqrt(xRange / (2000.0 * yRange)))  # DS:L12
    optimal = optimal + optimal * noverlap_fraction       # DS:L13
    return optimal, optimal * noverlap_fraction, optimal  # DS:L15-17


def create_focus_spectrogram(audio_full: np.ndarray, sr: int,
                             call_box: CallBox,
                             frequency_padding_kHz: float = 0.0,
                             ) -> FocusSpectrogramResult:
    """DS:CreateFocusSpectrogram.m.

    Computes the per-call focus STFT. The audio is cropped to the call's
    time range; the STFT params are computed from the call's duration and
    frequency range via compute_optimal_window; the spectrogram is then
    cropped in frequency to the call's band.

    Parameters
    ----------
    audio_full : 1-D ndarray
        The full audio waveform for the WAV file containing this call.
    sr : int
        Sample rate in Hz.
    call_box : CallBox
        The DeepSqueak Box for this call.
    frequency_padding_kHz : float, optional
        Extra frequency margin (kHz) below ``freq_start_kHz`` and above
        ``freq_end_kHz`` when cropping. Default 0 matches CreateFocusSpectrogram
        line 18 (options.frequency_padding = 0).

    Returns
    -------
    FocusSpectrogramResult
    """
    # Defensive: tiny freq_range -> runaway window. Floor to 1 kHz.
    safe_freq_range_kHz = max(call_box.freq_range_kHz, MIN_FREQ_RANGE_KHZ)
    win_s, ovl_s, nf_s = compute_optimal_window(
        call_box.duration_s, safe_freq_range_kHz
    )
    windowsize = int(round(sr * win_s))   # DS:L29
    noverlap = int(round(sr * ovl_s))     # DS:L30
    nfft = int(round(sr * nf_s))          # DS:L31

    # Defensive: cap windowsize at MAX_WINDOWSIZE_SAMPLES so a bad CSV row
    # can never trigger a multi-GB STFT allocation. The cap is wider than
    # any sensible USV focus window, so well-formed calls are untouched.
    if windowsize > MAX_WINDOWSIZE_SAMPLES:
        scale = MAX_WINDOWSIZE_SAMPLES / windowsize
        windowsize = MAX_WINDOWSIZE_SAMPLES
        noverlap = int(round(noverlap * scale))
        nfft = MAX_WINDOWSIZE_SAMPLES
    # noverlap must be strictly less than windowsize for scipy.signal.stft.
    if noverlap >= windowsize:
        noverlap = max(0, windowsize - 1)

    # Crop audio to [call_start_s, call_start_s + duration_s]. DS:L34
    i_lo = int(round(call_box.time_start_s * sr))
    i_hi = int(round(call_box.time_end_s * sr))
    i_lo = max(i_lo, 0)
    i_hi = min(i_hi, audio_full.shape[0])
    audio_seg = audio_full[i_lo:i_hi]

    if audio_seg.shape[0] < windowsize:
        # Pad to at least one window so STFT returns one column.
        audio_seg = np.pad(audio_seg, (0, windowsize - audio_seg.shape[0]),
                            mode="constant")

    # MATLAB spectrogram(audio, windowsize, noverlap, nfft, rate) with a scalar
    # windowsize uses a Hamming window of length `windowsize`. scipy.signal.stft
    # with boundary=None, padded=False matches that — no pre/post padding.
    hamming_win = scipy_signal.windows.hamming(windowsize, sym=False)
    fr_hz, ti_s, Zxx = scipy_signal.stft(
        audio_seg,
        fs=sr,
        window=hamming_win,
        nperseg=windowsize,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    I_full = np.abs(Zxx)  # DS:L46 (= abs(s) cropped later)

    # Frequency crop. DS:L41-44, on fr in Hz / 1000.
    fr_kHz = fr_hz / 1000.0
    lo_target = call_box.freq_start_kHz - frequency_padding_kHz
    hi_target = call_box.freq_end_kHz + frequency_padding_kHz

    lo_candidates = np.where(fr_kHz >= lo_target)[0]
    freq_lo_idx = int(lo_candidates[0]) if lo_candidates.size > 0 else 0
    freq_lo_idx = max(freq_lo_idx, 0)

    hi_candidates = np.where(fr_kHz <= hi_target)[0]
    freq_hi_idx = int(hi_candidates[-1]) if hi_candidates.size > 0 else fr_kHz.size - 1
    freq_hi_idx = min(freq_hi_idx, fr_kHz.size - 1)

    if freq_hi_idx < freq_lo_idx:
        # Defensive: don't return an empty I.
        freq_hi_idx = freq_lo_idx

    I = I_full[freq_lo_idx:freq_hi_idx + 1, :]

    return FocusSpectrogramResult(
        I=I,
        windowsize=windowsize,
        noverlap=noverlap,
        nfft=nfft,
        fr_hz=fr_hz,
        ti_s=ti_s,
        freq_lo_idx=freq_lo_idx,
        freq_hi_idx=freq_hi_idx,
    )


# ---------------------------------------------------------------------------
# CalculateStats port — internal primitives
# ---------------------------------------------------------------------------
def _spectral_flatness(I: np.ndarray) -> np.ndarray:
    """DS:CalculateStats.m:L8.

    Spectral flatness = geomean(I, axis=0) / mean(I, axis=0). Output is a
    1-D vector of length n_time_bins. Range is [0, 1]; near 0 = tonal,
    near 1 = noise-like.

    For numerical stability against log(0): small positive epsilon added.
    """
    if I.shape[0] == 0:
        return np.empty(I.shape[1], dtype=np.float64)
    eps = 1e-30
    safe = I + eps
    geo = np.exp(np.mean(np.log(safe), axis=0))
    arith = np.mean(safe, axis=0)
    return geo / arith


def _rlowess_smooth(y: np.ndarray, x: Optional[np.ndarray] = None,
                    span: float = 0.1,
                    n_iter: int = DS_RLOWESS_ITERATIONS) -> np.ndarray:
    """MATLAB ``smooth(y, span, 'rlowess')`` or ``smooth(x, y, span, 'rlowess')``.

    Robust locally-weighted scatterplot smoothing with bisquare weights.
    statsmodels' lowess with it>=1 implements robust iterations; setting
    it=5 matches MATLAB's rlowess default.

    Parameters
    ----------
    y : 1-D ndarray
        Values to smooth.
    x : 1-D ndarray, optional
        Independent variable. If None, defaults to ``np.arange(len(y))``,
        matching MATLAB's single-arg smooth() behavior.
    span : float
        Fraction of points used in each local fit, in (0, 1].
    n_iter : int
        Robust iterations.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return y.copy()
    if x is None:
        x = np.arange(y.size, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64).ravel()
    if y.size < 4 or span <= 0:
        return y.copy()
    span = float(min(max(span, 1.0 / y.size), 1.0))
    out = lowess(y, x, frac=span, it=int(n_iter), return_sorted=False)
    return np.asarray(out, dtype=np.float64).ravel()


@dataclass(frozen=True)
class CalculateStatsResult:
    """Subset of CalculateStats.m output needed for contour extraction.

    Mirrors MATLAB's ``stats`` struct. Only the fields used downstream by
    the contour-export wrapper are retained — sinuosity / slope / power
    are computed by MATLAB but not consumed by the contour pipeline.
    """

    ridge_time_col_idx: np.ndarray   # stats.ridgeTime  — STFT column indices
    ridge_freq_row_idx: np.ndarray   # stats.ridgeFreq  — raw freq row indices
    ridge_freq_smooth_row_idx: np.ndarray  # stats.ridgeFreq_smooth (rlowess)
    entropy_smoothed: np.ndarray     # stats.Entropy after smooth(..., 0.1, 'rlowess')
    time_scale_s: float              # TimeScale = (windowsize - noverlap)/sr
    freq_scale_kHz: float            # FreqScale = (sr/2000) / (1 + floor(nfft/2))


def calculate_stats(I: np.ndarray, windowsize: int, noverlap: int, nfft: int,
                    sr: int, call_box: CallBox,
                    entropy_threshold: float = DS_ENTROPY_THRESHOLD,
                    amplitude_threshold: float = DS_AMPLITUDE_THRESHOLD,
                    ) -> CalculateStatsResult:
    """DS:CalculateStats.m.

    Returns ridge detection bins in the focus-STFT grid (column indices and
    raw frequency row indices), plus the scales needed to convert to
    physical units. Conversion to (time_s, freq_kHz) is done by the caller
    so this function stays close to the MATLAB algorithm.

    Parameters mirror MATLAB's signature 1:1 except the call_box (used only
    so this function carries the same context the MATLAB function does via
    the ``Box`` arg).
    """
    if I.size == 0 or I.shape[1] == 0:
        return _empty_stats(windowsize, noverlap, nfft, sr)

    # DS:L8 — Entropy = geomean / mean (spectral flatness per column).
    entropy = _spectral_flatness(I)

    # DS:L11 — smooth entropy with rlowess, span=0.1.
    entropy_smoothed = _rlowess_smooth(entropy, span=DS_ENTROPY_SMOOTH_SPAN)

    # DS:L13-18 — brightThreshold is a global percentile of I.
    if 0.001 < amplitude_threshold < 0.999:
        bright_threshold = float(np.percentile(I.ravel(), amplitude_threshold * 100))
    else:
        bright_threshold = float(np.percentile(I.ravel(), 82.5))

    # DS:L20-23 — clip entropy_threshold to (0.001, 0.999) range.
    if entropy_threshold < 0.001 or entropy_threshold > 0.999:
        entropy_threshold = 0.215

    # DS:L27 — per-column max amplitude AND its row index (1-based in MATLAB).
    amplitude = I.max(axis=0)
    ridge_freq_row = I.argmax(axis=0).astype(np.int64)

    # DS:L28-46 — adaptive-lowering loop. Initial threshold pair is
    # (bright_threshold, entropy_threshold); iter k>=2 lowers BOTH by 1.1^iter.
    n_cols = I.shape[1]
    greater = np.zeros(n_cols, dtype=bool)
    iteration = 1
    while greater.sum() < DS_MIN_RIDGE_POINTS:
        if iteration == 1:
            amp_ok = amplitude > bright_threshold
            ent_ok = (1.0 - entropy_smoothed) > entropy_threshold
            greater = amp_ok & ent_ok
        else:
            factor = DS_LOWERING_FACTOR ** iteration
            amp_ok = amplitude > (bright_threshold / factor)
            ent_ok = (1.0 - entropy_smoothed) > (entropy_threshold / factor)
            greater = amp_ok & ent_ok
        iteration += 1
        if iteration > DS_MAX_LOWERING_ITERS:
            # DS:L42-46 — extreme failure case: keep every column.
            greater = np.ones(n_cols, dtype=bool)
            break

    # DS:L50-51 — ridge bins are the surviving column indices.
    ridge_time_col_idx = np.nonzero(greater)[0]
    ridge_freq_row_idx = ridge_freq_row[greater]

    # DS:L53-58 — smoothed frequency trajectory (rlowess on (time, freq)).
    if ridge_time_col_idx.size >= 4:
        try:
            ridge_freq_smooth = _rlowess_smooth(
                ridge_freq_row_idx.astype(np.float64),
                x=ridge_time_col_idx.astype(np.float64),
                span=DS_RIDGE_SMOOTH_SPAN,
            )
        except Exception:
            ridge_freq_smooth = ridge_freq_row_idx.astype(np.float64)
    else:
        # MATLAB falls back to ridgeFreq' when smoothing fails. DS:L58.
        ridge_freq_smooth = ridge_freq_row_idx.astype(np.float64)

    # DS:L62-65 — scaling factors.
    spectrange_kHz = sr / 2000.0
    freq_scale_kHz_per_pixel = spectrange_kHz / (1.0 + int(np.floor(nfft / 2)))
    time_scale_s_per_pixel = (windowsize - noverlap) / float(sr)

    return CalculateStatsResult(
        ridge_time_col_idx=ridge_time_col_idx,
        ridge_freq_row_idx=ridge_freq_row_idx,
        ridge_freq_smooth_row_idx=ridge_freq_smooth,
        entropy_smoothed=entropy_smoothed,
        time_scale_s=time_scale_s_per_pixel,
        freq_scale_kHz=freq_scale_kHz_per_pixel,
    )


def _empty_stats(windowsize: int, noverlap: int, nfft: int,
                 sr: int) -> CalculateStatsResult:
    spectrange_kHz = sr / 2000.0
    return CalculateStatsResult(
        ridge_time_col_idx=np.zeros(0, dtype=np.int64),
        ridge_freq_row_idx=np.zeros(0, dtype=np.int64),
        ridge_freq_smooth_row_idx=np.zeros(0, dtype=np.float64),
        entropy_smoothed=np.zeros(0, dtype=np.float64),
        time_scale_s=(windowsize - noverlap) / float(max(sr, 1)),
        freq_scale_kHz=spectrange_kHz / (1.0 + int(np.floor(max(nfft, 1) / 2))),
    )


# ---------------------------------------------------------------------------
# Convenience: end-to-end one call -> contour rows in physical units.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContourBins:
    """Per-call contour rows in physical units.

    The arrays are aligned: bin k has time_s[k], freq_kHz[k], tonality[k].
    Length is the number of surviving ridge columns (== ridge_time_col_idx
    size).
    """

    time_s: np.ndarray         # absolute seconds within the WAV
    freq_kHz: np.ndarray       # absolute kHz (call_box.freq_start_kHz + scale * row)
    tonality: np.ndarray       # 1 - entropy_smoothed at the column, in [0,1]


def extract_contour_for_call(audio_full: np.ndarray, sr: int,
                             call_box: CallBox,
                             entropy_threshold: float = DS_ENTROPY_THRESHOLD,
                             amplitude_threshold: float = DS_AMPLITUDE_THRESHOLD,
                             ) -> ContourBins:
    """End-to-end: WAV samples + Box -> ridge bins in physical units.

    Mirrors deepsqueak_export_contours.m lines 118-148. The only difference
    is that this function returns physical-unit arrays directly rather than
    writing to a parquet — that's the caller's job (extract_contours_python.py).

    Tonality is set to ``1 - entropy_smoothed`` at the surviving columns,
    matching DeepSqueak's own per-column tonality measure.
    """
    if call_box.duration_s <= 0.0 or call_box.freq_range_kHz <= 0.0:
        return ContourBins(np.zeros(0), np.zeros(0), np.zeros(0))

    focus = create_focus_spectrogram(audio_full, sr, call_box)
    stats = calculate_stats(focus.I, focus.windowsize, focus.noverlap,
                            focus.nfft, sr, call_box,
                            entropy_threshold=entropy_threshold,
                            amplitude_threshold=amplitude_threshold)

    if stats.ridge_time_col_idx.size == 0:
        return ContourBins(np.zeros(0), np.zeros(0), np.zeros(0))

    # DS:deepsqueak_export_contours.m:L144-146
    time_s = (stats.ridge_time_col_idx.astype(np.float64) * stats.time_scale_s
              + call_box.time_start_s)
    # Freq offset = physical center of first kept FFT bin (0-based Python),
    # NOT call_box.freq_start_kHz — using Box(2) reproduces MATLAB's 1-based
    # off-by-one in 0-based Python. See 2026-05-19 path-c handoff.
    freq_kHz = (stats.ridge_freq_smooth_row_idx * stats.freq_scale_kHz
                + focus.fr_hz[focus.freq_lo_idx] / 1000.0)

    # Tonality at surviving columns. entropy_smoothed has length n_cols; we
    # index it with the surviving column indices.
    tonality = 1.0 - stats.entropy_smoothed[stats.ridge_time_col_idx]
    # Clip into [0, 1] — smoothing can push slightly past either bound.
    tonality = np.clip(tonality, 0.0, 1.0)

    return ContourBins(time_s=time_s, freq_kHz=freq_kHz, tonality=tonality)


__all__ = [
    "CallBox",
    "FocusSpectrogramResult",
    "CalculateStatsResult",
    "ContourBins",
    "DS_ENTROPY_THRESHOLD",
    "DS_AMPLITUDE_THRESHOLD",
    "compute_optimal_window",
    "create_focus_spectrogram",
    "calculate_stats",
    "extract_contour_for_call",
]
