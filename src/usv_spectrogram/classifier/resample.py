"""Polyphase resampling from 300 kHz (corpus) to 250 kHz (VocalMat-aligned).

The lab CNN classifier (Phase 18) operates internally at 250 kHz to match the
VocalMat reference corpus, while our hardware records at 300 kHz
(``corpus.SAMPLE_RATE_HZ``). A rational 5/6 resample bridges the two with
built-in anti-aliasing via the Kaiser-window FIR filter inside
``scipy.signal.resample_poly``.

Module 18.1 already placed ``TARGET_SAMPLE_RATE_HZ``, ``RESAMPLE_UP``, and
``RESAMPLE_DOWN`` in ``classifier/__init__.py`` so the cleaning-gate could
reason about the target rate without importing this file. This module imports
those constants and re-exports them, plus binds ``SOURCE_SAMPLE_RATE_HZ`` to
``corpus.SAMPLE_RATE_HZ`` (ADR-001 — never redeclare the canonical sample rate).
"""
# VAULT: [[project-cnn-retrain-matched-windows]] [[feedback-cnn-inference-global-mad]]
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, resample_poly

from usv_spectrogram.corpus import SAMPLE_RATE_HZ as SOURCE_SAMPLE_RATE_HZ

from . import RESAMPLE_DOWN, RESAMPLE_UP, TARGET_SAMPLE_RATE_HZ

__all__ = [
    "SOURCE_SAMPLE_RATE_HZ",
    "TARGET_SAMPLE_RATE_HZ",
    "RESAMPLE_UP",
    "RESAMPLE_DOWN",
    "resample_to_vocalmat",
]

# Anti-alias FIR designed at the polyphase intermediate rate (up * source = 1.5 MHz).
# The default scipy `resample_poly` Kaiser FIR (121 taps, β=5.0) gives only ~32 dB
# rejection 15 kHz above the new Nyquist (125 kHz) — insufficient for ROADMAP test 5
# which demands ≥40 dB at 110 kHz for a 140 kHz aggressor tone. A longer Kaiser FIR
# with β=14 (≈125 dB sidelobe) and ~480 taps tightens the transition band so the
# anti-aliasing requirement is comfortably met without sacrificing in-band fidelity.
_FIR_NUMTAPS: int = 481
_FIR_CUTOFF_HZ: float = 120_000.0  # 5 kHz below new Nyquist as guard band
_FIR_KAISER_BETA: float = 14.0
_ANTIALIAS_FIR: np.ndarray = firwin(
    _FIR_NUMTAPS,
    cutoff=_FIR_CUTOFF_HZ,
    fs=float(SOURCE_SAMPLE_RATE_HZ * RESAMPLE_UP),
    window=("kaiser", _FIR_KAISER_BETA),
).astype(np.float64)


def resample_to_vocalmat(samples: np.ndarray) -> np.ndarray:
    """Resample mono audio from 300 kHz to 250 kHz.

    Uses ``scipy.signal.resample_poly`` with ``up=5, down=6``. The Kaiser-window
    FIR low-pass filter inside ``resample_poly`` handles anti-aliasing — any
    energy above the new Nyquist (125 kHz) is attenuated before decimation,
    preventing fold-back artifacts.

    Parameters
    ----------
    samples : np.ndarray
        Mono 1-D audio at 300 kHz. ``float32`` or ``float64`` accepted; the
        output is always coerced to ``float32`` because the downstream STFT
        and CNN code assume that dtype.

    Returns
    -------
    np.ndarray
        Mono 1-D audio at 250 kHz, ``float32``. Length ≈ ``len(samples) * 5/6``
        (within one sample due to polyphase boundary handling).

    Raises
    ------
    ValueError
        If ``samples`` is not 1-D. Stereo inputs and accidental ``(1, N)`` row
        vectors are both rejected — convert to mono before calling.
    """
    if samples.ndim != 1:
        raise ValueError(
            "resample_to_vocalmat requires mono (1-D) audio, got shape "
            f"{samples.shape}. Convert stereo or 2-D row vectors to mono "
            "before resampling."
        )
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)
    out = resample_poly(
        samples,
        up=RESAMPLE_UP,
        down=RESAMPLE_DOWN,
        window=_ANTIALIAS_FIR,
    )
    return out.astype(np.float32, copy=False)
