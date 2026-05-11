"""Adaptive soft-notch filter for pre-CNN equipment-tonal suppression.

See ``docs/handoffs/2026-05-11_adaptive-soft-notch.md`` for the full design
rationale. Two operating modes:

- **Library mode** (preferred for batch detection): per-rig
  :class:`TonalLibrary` loaded at batch start. For each WAV chunk the
  library's frequencies/widths drive the filter; cut depth is measured
  per-chunk against the local PSD median.
- **Audit mode** (always-on companion to library mode; primary path when no
  library is supplied): per-chunk :func:`discover_tonals` via Welch PSD +
  local-median baseline + peak clustering. Reconciled against the library
  to surface drift. **Unmatched detections are logged, not filtered** —
  the library is the source of truth, the audit is a stale-library alarm.

Filter formulation
------------------
Each soft-notch is a complementary-bandpass subtraction::

    alpha = 1.0 - 10**(-cut_depth_db / 20.0)
    audio = audio - alpha * sosfiltfilt(bandpass, audio)

``alpha == 1`` recovers a hard band-stop (``cut_depth_db -> inf``).
``alpha < 1`` leaves a finite-depth dip whose stop-band attenuation equals
``cut_depth_db``. Mathematically identical to a parametric peaking-EQ cut,
but reuses the existing scipy Butterworth machinery the rest of the
pipeline already trusts.

Default-off invariant
---------------------
This module modifies the audio sample buffer the CNN sees. Wild-mouse
batches (5970, 3452, 9252) MUST produce byte-identical detections without
``--soft-notch`` — see GATE C of the spec handoff for the byte-equivalence
test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from ...corpus import USV_FREQ_MAX_HZ, USV_FREQ_MIN_HZ

__all__ = [
    "DetectedTonal",
    "LibraryEntry",
    "TonalLibrary",
    "ReconciliationResult",
    "discover_tonals",
    "reconcile",
    "apply_soft_notches",
    "auto_soft_notch",
]


# ----------------------------------------------------------------------------
# Data contracts
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedTonal:
    """A tonal measured from a single PSD (audit-mode output).

    Fields are populated by :func:`discover_tonals`:

    - ``center_hz``: argmax frequency of the peak (after local-median subtraction).
    - ``width_hz``: measured -3 dB width of the peak.
    - ``peak_db``: PSD value at the peak frequency, in dB.
    - ``local_median_db``: median PSD in the surrounding ``median_window_hz``,
      excluding the peak itself.
    - ``above_median_db``: ``peak_db - local_median_db``. Sign-positive.
    """
    center_hz: float
    width_hz: float
    peak_db: float
    local_median_db: float
    above_median_db: float


@dataclass(frozen=True)
class LibraryEntry:
    """A calibrated tonal that should always be filtered for this rig.

    Aggregate stats across the calibration sample. ``mean_above_median_db``
    and ``stdev_above_median_db`` together define the intensity-drift
    detector: a per-chunk measurement that deviates by more than
    ``intensity_drift_sigma`` from the mean triggers a warning.

    ``detection_rate`` is the fraction of calibration chunks in which this
    tonal was discovered. Entries below ``min_detection_rate`` are rejected
    during calibration (transient noise vs. recurrent equipment line).
    """
    center_hz: float
    width_hz: float
    mean_above_median_db: float
    stdev_above_median_db: float
    n_chunks_seen: int
    detection_rate: float


@dataclass(frozen=True)
class TonalLibrary:
    """Calibrated tonal library for a single rig.

    Layer-2 corpus fact stored at ``data/lab_tonal_lines/<rig_id>.json``.
    See ``docs/modules/corpus-constants.md`` for the layer architecture.
    """
    rig_id: str
    calibrated_at: str          # ISO 8601 timestamp
    n_chunks_sampled: int
    sample_files: list[str]
    entries: list[LibraryEntry]

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TonalLibrary":
        """Load and validate a tonal library from JSON.

        Raises ``ValueError`` on malformed JSON or invalid field values
        (e.g. ``detection_rate`` outside ``[0, 1]``).
        """
        raise NotImplementedError("TonalLibrary.load not yet implemented")

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the library to JSON. Round-trips with :meth:`load`."""
        raise NotImplementedError("TonalLibrary.save not yet implemented")


@dataclass(frozen=True)
class ReconciliationResult:
    """Audit output: how per-chunk detections reconcile against the library.

    - ``matched``: list of ``(library_entry, detected_tonal)`` pairs whose
      centers agree within ``freq_tolerance_hz``.
    - ``unmatched_detections``: detections that have no library counterpart
      (drift — possible stale library).
    - ``unmatched_library_entries``: library entries that did NOT appear in
      this chunk (informational; entries may be absent on quiet chunks).
    - ``intensity_drifts``: ``(library_entry, sigma)`` pairs where the
      measured ``above_median_db`` deviates from the library mean by more
      than ``intensity_drift_sigma`` standard deviations.
    """
    matched: list[tuple[LibraryEntry, DetectedTonal]]
    unmatched_detections: list[DetectedTonal]
    unmatched_library_entries: list[LibraryEntry]
    intensity_drifts: list[tuple[LibraryEntry, float]]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def discover_tonals(
    audio: np.ndarray,
    fs_hz: float,
    *,
    usv_band_min_hz: float = float(USV_FREQ_MIN_HZ),
    usv_band_max_hz: float = float(USV_FREQ_MAX_HZ),
    discovery_threshold_db: float = 10.0,
    median_window_hz: float = 4_000.0,
    nperseg: int = 8192,
) -> list[DetectedTonal]:
    """Discover stationary tonal peaks in ``audio`` via Welch PSD.

    Algorithm:
        1. Welch PSD with ``nperseg`` (default 8192 → ~37 Hz/bin at 300 kHz).
        2. For each freq bin in ``[usv_band_min_hz, usv_band_max_hz]``,
           compute a rolling median over a ``median_window_hz`` neighborhood.
        3. Find bins where ``pxx_db - rolling_median > discovery_threshold_db``.
        4. Cluster contiguous bins; for each cluster, center = argmax bin,
           width = contiguous span at peak - 3 dB.

    For multi-channel audio, PSD is computed on channel 0; tonals describe
    the rig, not channel-specific artefacts.
    """
    raise NotImplementedError("discover_tonals not yet implemented")


def reconcile(
    library: TonalLibrary,
    detections: Sequence[DetectedTonal],
    *,
    freq_tolerance_hz: float = 200.0,
    intensity_drift_sigma: float = 2.0,
) -> ReconciliationResult:
    """Reconcile per-chunk detections against the library.

    Matching is by center-frequency proximity (``freq_tolerance_hz``).
    Intensity drift fires when a matched detection's ``above_median_db``
    deviates from ``library_entry.mean_above_median_db`` by more than
    ``intensity_drift_sigma`` standard deviations of the library entry.
    """
    raise NotImplementedError("reconcile not yet implemented")


def apply_soft_notches(
    audio: np.ndarray,
    fs_hz: float,
    tonals: Sequence[Union[LibraryEntry, DetectedTonal]],
    *,
    min_width_hz: float = 200.0,
    width_safety_factor: float = 2.0,
    safety_margin_db: float = 0.0,
    order: int = 4,
) -> np.ndarray:
    """Apply soft-notch filters at every supplied tonal.

    Each tonal's filter is a Butterworth band-pass at
    ``[center - width/2, center + width/2]`` (order ``order``, effective
    order ``2*order`` via ``sosfiltfilt``). The complementary-bandpass
    subtraction ``audio - alpha * bandpass(audio)`` is applied, with
    ``alpha`` derived from the cut depth.

    Cut depth per tonal:

    - For :class:`DetectedTonal`: measured per-chunk as
      ``above_median_db + safety_margin_db``.
    - For :class:`LibraryEntry`: measured per-chunk from the local PSD
      (NOT from ``mean_above_median_db`` — that's a sanity reference
      only), then ``+ safety_margin_db``.

    Width:

    - For :class:`LibraryEntry`: ``width_hz`` is used directly
      (deterministic kill zone).
    - For :class:`DetectedTonal`: ``max(min_width_hz, measured *
      width_safety_factor)``.

    Multi-channel audio is filtered per-channel with the same SOS coefficients.

    Returns audio with the same shape and dtype as the input.
    """
    raise NotImplementedError("apply_soft_notches not yet implemented")


def auto_soft_notch(
    audio: np.ndarray,
    fs_hz: float,
    library: TonalLibrary | None = None,
    **kwargs,
) -> tuple[np.ndarray, ReconciliationResult]:
    """One-shot entry: discover + reconcile + filter.

    Behavior depends on whether a library is supplied:

    - ``library is None`` (pure auto-detect): every discovered tonal is
      filtered; ``reconciliation.matched`` and
      ``reconciliation.unmatched_library_entries`` are empty;
      ``reconciliation.unmatched_detections`` carries every discovered tonal
      (informational — they WERE filtered).
    - ``library is not None`` (library mode): library entries drive the
      filter; discovered tonals are reconciled against the library;
      unmatched detections are logged only (not filtered).

    ``**kwargs`` are forwarded to :func:`discover_tonals`,
    :func:`reconcile`, and :func:`apply_soft_notches` by parameter-name match.

    Returns ``(cleaned_audio, reconciliation_result)``. When no tonals are
    discovered AND no library entries exist, ``cleaned_audio`` is
    byte-identical to ``audio``.
    """
    raise NotImplementedError("auto_soft_notch not yet implemented")
