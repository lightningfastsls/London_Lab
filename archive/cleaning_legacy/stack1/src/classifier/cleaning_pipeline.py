"""4-layer spectrogram cleaning stack for the classifier validation gate.

Orchestrates the four cleaning layers used pre-CNN, with each layer
independently toggleable so Module 18.1's diagnostic can ablate. Wraps
existing implementations where possible (Boll baseline subtraction) and
reproduces the global-MAD-then-crop pattern from production
``app/core/sliding_inference.py`` without importing the heavy CNN runner.

Layer order is FIXED::

    soft-notch -> baseline subtraction -> global MAD -> per-recording Z-score

Re-ordering is a behaviour change and must not happen silently
(`test_clean_spectrogram_layer_order_*`). Module-level private functions
``_apply_soft_notch``, ``_apply_baseline_subtraction``, ``_apply_global_mad``
and ``_apply_per_recording_zscore`` are the patch boundary for the
order-verification test — DO NOT rename or inline them.

Cross-phase constraints (ROADMAP Phase 18)
------------------------------------------
- C1: Default ``sample_rate_hz`` is 250_000 (VocalMat-aligned). The 300_000
  rate is also accepted for cross-cohort runs but never used as default.
- C2: Global MAD computes on the WHOLE spectrogram once; downstream crop
  windows are sampled AFTER normalization. The math here reproduces
  ``sliding_inference.py:_apply_mad_normalization`` byte-for-byte, including
  the ``vmax > vmin`` divide-by-zero guard.
- C3: Soft-notch, baseline subtraction and per-recording Z-score are wrappers
  around existing implementations. Global MAD is reproduced rather than
  imported because the upstream lives inside a heavy class with PyQt6 and
  CNN dependencies.
- C4: When ``apply_soft_notch=True`` and ``tonal_library_path=None``, the
  soft-notch layer no-ops. This is the only valid configuration for the
  VocalMat and wild-5970 cohorts (we have no calibrated tonal library for
  either — only lab_131204 has one).
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MAD normalization scale factors — must match
# ``app/core/sliding_inference.py:_apply_mad_normalization`` exactly
# (training-grid invariant; per ``feedback-cnn-inference-global-mad``).
_MAD_VMIN_SCALE: float = 2.0
_MAD_VMAX_SCALE: float = 4.0
_MAD_EPS: float = 1e-12

# Z-score epsilon — protects against division by zero on a constant spectrogram.
_ZSCORE_EPS: float = 1e-12

# Boll 1979 baseline subtraction operates in linear magnitude, not dB.
# Matches upstream ``app/core/denoise.py:DEFAULT_EPSILON`` so the
# fallback path produces numerically identical output to the upstream
# implementation when the upstream module is not importable.
_DB_TO_LINEAR_EPS: float = 1e-10

# Module-default baseline percentile for the "percentile" mode wrapper.
# Matches the production default in ``app/core/denoise.py``.
_DEFAULT_BASELINE_PERCENTILE: float = 10.0

# Canonical STFT hop, sourced from ``corpus.STFT_HOP`` when available so
# the fallback baseline kernel honours any future corpus update. The
# hardcoded 128 mirrors the current corpus value and only fires on
# minimal worktree checkouts where ``corpus`` is not importable —
# matches default STFT hop; only used if upstream module unavailable.
try:
    from ..corpus import STFT_HOP as _FALLBACK_BASELINE_KERNEL_HOP  # type: ignore[no-redef]
except Exception:  # pragma: no cover — corpus missing in minimal checkouts
    _FALLBACK_BASELINE_KERNEL_HOP: int = 128  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_CleaningConfigBase = namedtuple(
    "_CleaningConfigBase",
    [
        "apply_soft_notch",
        "apply_baseline_subtraction",
        "apply_global_mad",
        "apply_per_recording_zscore",
        "baseline_mode",
        "tonal_library_path",
        "sample_rate_hz",
    ],
)


class CleaningConfig(_CleaningConfigBase):
    """Configuration for the 4-layer pre-CNN cleaning stack.

    Each layer is independently toggleable so the diagnostic can perform
    leave-one-in / leave-one-out ablation.

    Implementation note
    -------------------
    This is implemented as a ``namedtuple`` subclass (not a frozen
    dataclass) so the test ``test_cleaning_config_is_immutable_after_creation``
    -- which uses ``object.__setattr__`` to probe immutability -- raises
    AttributeError as expected. A standard frozen dataclass would let
    ``object.__setattr__`` succeed because the slot descriptor is
    settable via the C-level path. Namedtuples have ``__slots__ = ()`` and
    expose fields via property descriptors with no setter; this is the
    only Python construct that satisfies *both* the equality / kw-args
    contract AND ``object.__setattr__`` immutability.

    Notes
    -----
    - ``sample_rate_hz`` defaults to **250_000** (VocalMat-aligned). The
      canonical corpus rate (300_000, ADR-001) is also accepted for
      cross-cohort comparisons but is NOT the default for this pipeline.
    - ``tonal_library_path=None`` with ``apply_soft_notch=True`` is valid
      and silently no-ops. This is the only valid configuration for cohorts
      without a calibrated tonal library (VocalMat, wild-5970).
    """

    __slots__ = ()

    def __new__(
        cls,
        apply_soft_notch: bool = True,
        apply_baseline_subtraction: bool = True,
        apply_global_mad: bool = True,
        apply_per_recording_zscore: bool = True,
        baseline_mode: str = "median_envelope",
        tonal_library_path: Optional[Path] = None,
        sample_rate_hz: int = 250_000,
    ) -> "CleaningConfig":
        if baseline_mode not in {"percentile", "median_envelope"}:
            raise ValueError(
                "baseline_mode must be 'percentile' or 'median_envelope', "
                f"got {baseline_mode!r}"
            )
        if sample_rate_hz not in {250_000, 300_000}:
            raise ValueError(
                "sample_rate_hz must be 250000 (VocalMat-aligned, default) "
                f"or 300000 (corpus canonical, ADR-001); got {sample_rate_hz}"
            )
        # apply_soft_notch + tonal_library_path=None is intentionally allowed
        # (silent no-op). Caller documents the choice in CLI logs.
        return super().__new__(
            cls,
            apply_soft_notch,
            apply_baseline_subtraction,
            apply_global_mad,
            apply_per_recording_zscore,
            baseline_mode,
            tonal_library_path,
            sample_rate_hz,
        )


# ---------------------------------------------------------------------------
# Layer functions (private — patched by test_clean_spectrogram_layer_order_*)
# ---------------------------------------------------------------------------


def _apply_soft_notch(
    spec_db: np.ndarray,
    cfg: CleaningConfig,
    recording_id: str,
) -> np.ndarray:
    """Soft-notch tonal removal (Layer 1).

    The production soft-notch in ``app/core/notch.py`` operates on
    time-domain audio. Here we receive a dB-scale spectrogram and have no
    audio access, so the layer is a no-op when no tonal library is
    available. When a library is supplied, the tonal bins are attenuated
    in-place at ``cut_depth_db`` matched to the calibrated entry.

    For Module 18.1 the no-op path is exercised by the ablation matrix on
    VocalMat and wild-5970 cohorts. The lab_131204 cohort uses the
    calibrated library at ``data/lab_tonal_lines/lab_131204.json``.
    """
    if cfg.tonal_library_path is None:
        # Documented no-op (C4). DO NOT raise — the caller has explicitly
        # opted into the no-op by leaving the library path None.
        return spec_db

    # Library-mode spectrogram-level notch. Load the library, map each
    # entry's centre frequency to the closest freq bin and attenuate the
    # bin (and its ±half-width neighbours) by the calibrated cut depth.
    try:
        from ..app.core.notch import TonalLibrary
    except Exception:
        # Optional dependency — if the notch module is not importable in
        # this worktree, no-op silently rather than crash the diagnostic.
        return spec_db

    library = TonalLibrary.load(cfg.tonal_library_path)
    if not library.entries:
        return spec_db

    n_freq = spec_db.shape[0]
    # Frequency axis assumed linear from 0 to nyquist over n_freq bins.
    nyq = cfg.sample_rate_hz / 2.0
    freqs_hz = np.linspace(0.0, nyq, n_freq, endpoint=True)

    out = spec_db.copy()
    for entry in library.entries:
        lo = entry.center_hz - entry.width_hz / 2.0
        hi = entry.center_hz + entry.width_hz / 2.0
        mask = (freqs_hz >= lo) & (freqs_hz <= hi)
        if not mask.any():
            continue
        # Library entries don't carry a cut depth — use mean_above_median_db
        # as the calibrated attenuation budget.
        cut_db = max(0.0, float(entry.mean_above_median_db))
        out[mask, :] -= cut_db
    return out


def _apply_baseline_subtraction(
    spec_db: np.ndarray,
    cfg: CleaningConfig,
    recording_id: str,
) -> np.ndarray:
    """Per-bin temporal baseline subtraction (Layer 2, Boll 1979).

    Wraps ``app/core/denoise.subtract_temporal_baseline`` when available.
    The upstream operates in linear magnitude, so we convert dB->linear,
    subtract, then linear->dB. When the upstream is not importable, fall
    back to an equivalent in-module implementation so the diagnostic stays
    runnable on minimal checkouts.
    """
    spec_lin = np.power(10.0, spec_db / 20.0)

    try:
        from ..app.core.denoise import subtract_temporal_baseline
        cleaned_lin = subtract_temporal_baseline(
            spec_lin,
            method=cfg.baseline_mode,
            percentile=_DEFAULT_BASELINE_PERCENTILE,
        )
    except Exception:
        # Local fallback — keep math equivalent to upstream defaults.
        cleaned_lin = _local_baseline_subtract(
            spec_lin,
            method=cfg.baseline_mode,
            sample_rate_hz=cfg.sample_rate_hz,
        )

    cleaned_lin = np.maximum(cleaned_lin, _DB_TO_LINEAR_EPS)
    return 20.0 * np.log10(cleaned_lin)


def _local_baseline_subtract(
    spec_lin: np.ndarray,
    method: str,
    sample_rate_hz: int = 250_000,
) -> np.ndarray:
    """In-module fallback for ``subtract_temporal_baseline``.

    Reproduces the upstream defaults; used only when the upstream module
    is not importable (e.g. minimal worktree checkout). Behaviour matches
    ``app/core/denoise.py:subtract_temporal_baseline`` for the default
    parameters used by the diagnostic.

    The ``median_envelope`` kernel follows upstream's "0.5 s of audio"
    rule: ``int(0.5 * sample_rate_hz / hop)`` (rounded to odd, floor at
    3). The hop is sourced from ``_FALLBACK_BASELINE_KERNEL_HOP`` which
    in turn imports ``corpus.STFT_HOP`` when available.
    """
    if method == "percentile":
        baseline = np.percentile(
            spec_lin, _DEFAULT_BASELINE_PERCENTILE, axis=1, keepdims=True
        )
        return np.maximum(spec_lin - baseline, _DB_TO_LINEAR_EPS)

    if method == "median_envelope":
        from scipy.ndimage import median_filter
        n_time = spec_lin.shape[1]
        # Upstream rule: 0.5 s of audio worth of STFT columns.
        kernel = max(
            3,
            int(0.5 * sample_rate_hz / _FALLBACK_BASELINE_KERNEL_HOP) | 1,
        )
        # Never exceed the actual time-axis length; force odd.
        if kernel > n_time:
            kernel = max(3, n_time | 1 if n_time >= 3 else 3)
        baseline = median_filter(spec_lin, size=(1, kernel), mode="reflect")
        return np.maximum(spec_lin - baseline, _DB_TO_LINEAR_EPS)

    raise ValueError(
        f"Unknown baseline method {method!r}; expected 'percentile' or "
        "'median_envelope'."
    )


def _apply_global_mad(
    spec_db: np.ndarray,
    cfg: CleaningConfig,
    recording_id: str,
) -> np.ndarray:
    """Global MAD normalisation (Layer 3).

    Reproduces ``app/core/sliding_inference.py:_apply_mad_normalization``
    (lines 388-424 at HEAD). The whole-spectrogram-then-crop pattern is
    cross-phase constraint C2 — per-window MAD is the known regression
    (``feedback-cnn-inference-global-mad``).

    Output is in [0, 1] with a documented guard for the degenerate
    ``vmax == vmin`` case (constant-input spectrogram returns all zeros).
    """
    median = np.median(spec_db)
    mad = np.median(np.abs(spec_db - median))

    vmin = median - _MAD_VMIN_SCALE * mad
    vmax = median + _MAD_VMAX_SCALE * mad

    # CRITICAL: clip BEFORE normalising (matches training pipeline).
    spec_clipped = np.clip(spec_db, vmin, vmax)

    if vmax > vmin:
        return (spec_clipped - vmin) / (vmax - vmin + _MAD_EPS)
    # Degenerate constant-input case (vmax == vmin). Returning zeros is
    # what production does — checked by
    # ``test_clean_spectrogram_mad_only_produces_finite_output``.
    return np.zeros_like(spec_db)


def _apply_per_recording_zscore(
    spec_db: np.ndarray,
    cfg: CleaningConfig,
    recording_id: str,
) -> np.ndarray:
    """Per-recording Z-score on the spectrogram (Layer 4).

    The dormant ``postprocessing/normalization.normalize_scores_per_recording``
    operates on 1-D probability arrays. For spectrograms we apply a 2-D
    analogue: subtract the recording-level median and divide by the
    recording-level MAD. The recording identifier is intentionally unused
    here (the function is called per-spectrogram and the relevant stats
    are intrinsic to the array passed in); the parameter exists so a
    future implementation can fold in cross-call statistics keyed by
    ``recording_id``.

    NOTE: This 2D analogue diverges from the upstream 1D
    ``normalize_scores_per_recording`` (postprocessing/normalization.py),
    which uses a bottom-50%-percentile noise slice to isolate the noise
    floor. The 2D version uses global median + MAD, assuming USVs occupy
    <50% of pixel area (true for typical USV spectrograms where calls
    are <1% of pixels). For dense-babble regimes where USVs occupy a
    large fraction of pixels, this will inflate the noise estimate and
    depress the normalized contrast. The upstream 1D semantics could be
    ported by sorting all pixels and taking median+MAD on the bottom
    half.
    """
    median = np.median(spec_db)
    mad = np.median(np.abs(spec_db - median))
    if mad < _ZSCORE_EPS:
        # Constant-spectrogram fallback — return a centred copy.
        return spec_db - median
    return (spec_db - median) / mad


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def clean_spectrogram(
    spec: np.ndarray,
    cfg: CleaningConfig,
    recording_id: str,
) -> np.ndarray:
    """Apply the 4-layer pre-CNN cleaning stack in fixed order.

    Layer order is FIXED::

        soft-notch -> baseline subtraction -> global MAD -> per-recording Z-score

    Re-ordering is a behaviour change and is enforced by
    ``test_clean_spectrogram_layer_order_*``.

    Parameters
    ----------
    spec:
        Spectrogram in dB, shape ``(n_freq_bins, n_time_frames)``.
    cfg:
        Cleaning configuration (frozen).
    recording_id:
        Identifier used by Layer 4 for cross-call statistics (currently a
        no-op pass-through; reserved for future per-recording stats).

    Returns
    -------
    cleaned:
        Cleaned spectrogram, same shape and floating dtype as ``spec``.

    Notes
    -----
    When every layer is disabled, the input is returned UNCHANGED
    (``test_clean_spectrogram_all_layers_disabled_returns_input_unchanged``
    asserts ``np.array_equal``). Do not insert sneaky dtype casts on the
    short-circuit path.
    """
    if spec.ndim != 2:
        raise ValueError(
            f"clean_spectrogram expects a 2-D spectrogram (n_freq, n_time); "
            f"got shape {spec.shape}"
        )

    out = spec
    if cfg.apply_soft_notch:
        out = _apply_soft_notch(out, cfg, recording_id)
    if cfg.apply_baseline_subtraction:
        out = _apply_baseline_subtraction(out, cfg, recording_id)
    if cfg.apply_global_mad:
        out = _apply_global_mad(out, cfg, recording_id)
    if cfg.apply_per_recording_zscore:
        out = _apply_per_recording_zscore(out, cfg, recording_id)

    return out
