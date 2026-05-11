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

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

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
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {p}: {exc}") from exc

        for required in ("rig_id", "calibrated_at", "n_chunks_sampled",
                         "sample_files", "entries"):
            if required not in data:
                raise ValueError(f"Missing required field '{required}' in {p}")

        entries: list[LibraryEntry] = []
        for i, e in enumerate(data["entries"]):
            for required in ("center_hz", "width_hz", "mean_above_median_db",
                             "stdev_above_median_db", "n_chunks_seen", "detection_rate"):
                if required not in e:
                    raise ValueError(
                        f"Missing required field '{required}' in entry {i} of {p}"
                    )
            rate = float(e["detection_rate"])
            if not (0.0 <= rate <= 1.0):
                raise ValueError(
                    f"detection_rate {rate} out of range [0, 1] in entry {i} "
                    f"(center_hz={e['center_hz']}) of {p}"
                )
            entries.append(LibraryEntry(
                center_hz=float(e["center_hz"]),
                width_hz=float(e["width_hz"]),
                mean_above_median_db=float(e["mean_above_median_db"]),
                stdev_above_median_db=float(e["stdev_above_median_db"]),
                n_chunks_seen=int(e["n_chunks_seen"]),
                detection_rate=rate,
            ))

        return cls(
            rig_id=str(data["rig_id"]),
            calibrated_at=str(data["calibrated_at"]),
            n_chunks_sampled=int(data["n_chunks_sampled"]),
            sample_files=list(data["sample_files"]),
            entries=entries,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the library to JSON. Round-trips with :meth:`load`."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rig_id": self.rig_id,
            "calibrated_at": self.calibrated_at,
            "n_chunks_sampled": self.n_chunks_sampled,
            "sample_files": list(self.sample_files),
            "entries": [
                {
                    "center_hz": float(e.center_hz),
                    "width_hz": float(e.width_hz),
                    "mean_above_median_db": float(e.mean_above_median_db),
                    "stdev_above_median_db": float(e.stdev_above_median_db),
                    "n_chunks_seen": int(e.n_chunks_seen),
                    "detection_rate": float(e.detection_rate),
                }
                for e in self.entries
            ],
        }
        p.write_text(json.dumps(payload, indent=2))


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
# Internal helpers
# ----------------------------------------------------------------------------

_EPS_PSD = 1e-30  # numerical floor for log10 of zero-PSD bins


def _select_channel(audio: np.ndarray) -> np.ndarray:
    """For multi-channel audio, PSD is computed on channel 0 only (spec OQ-8)."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        # soundfile's always_2d shape is (samples, channels); take channel 0.
        return audio[:, 0]
    raise ValueError(f"Unsupported audio ndim={audio.ndim}; expected 1 or 2.")


def _welch_psd_db(
    audio_1d: np.ndarray, fs_hz: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_hz, pxx_db) from Welch PSD on a 1-D signal."""
    nperseg_actual = min(nperseg, len(audio_1d))
    if nperseg_actual < 16:
        # Too short for a meaningful PSD; return empty arrays.
        return np.empty(0), np.empty(0)
    f, pxx = welch(audio_1d, fs=fs_hz, nperseg=nperseg_actual)
    pxx_db = 10.0 * np.log10(pxx + _EPS_PSD)
    return f, pxx_db


def _rolling_median_db(values_db: np.ndarray, half_window_bins: int) -> np.ndarray:
    """Rolling median over ``[i - half_window, i + half_window]`` (inclusive).

    Edges use the available window (no wraparound, no reflection).
    """
    n = len(values_db)
    out = np.empty(n, dtype=values_db.dtype)
    hw = max(1, int(half_window_bins))
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        out[i] = np.median(values_db[lo:hi])
    return out


def _measure_band_psd(
    f_hz: np.ndarray,
    pxx_db: np.ndarray,
    band_lo_hz: float,
    band_hi_hz: float,
    median_window_hz: float,
) -> tuple[float, float]:
    """Return ``(peak_db, local_median_db)`` for a target band.

    The peak is the max PSD inside ``[band_lo_hz, band_hi_hz]``; the local
    median is computed over a ``median_window_hz``-wide neighborhood centered
    on the band, EXCLUDING the band itself. Used for library-mode per-chunk
    cut-depth measurement.
    """
    band_mask = (f_hz >= band_lo_hz) & (f_hz <= band_hi_hz)
    if not band_mask.any():
        return float("nan"), float("nan")
    peak_db = float(np.max(pxx_db[band_mask]))

    center_hz = 0.5 * (band_lo_hz + band_hi_hz)
    nbhd_mask = (
        (f_hz >= center_hz - median_window_hz / 2.0)
        & (f_hz <= center_hz + median_window_hz / 2.0)
        & ~band_mask
    )
    if not nbhd_mask.any():
        # Degenerate (band fills the neighborhood); fall back to band itself.
        nbhd_mask = band_mask
    local_median_db = float(np.median(pxx_db[nbhd_mask]))
    return peak_db, local_median_db


def _apply_one_band(
    audio: np.ndarray,
    fs_hz: float,
    center_hz: float,
    width_hz: float,
    cut_depth_db: float,
    order: int,
) -> np.ndarray:
    """Apply a single complementary-bandpass soft-notch.

    Returns audio with the same dtype/shape as input. Multi-channel arrays
    are filtered per-channel with the same SOS coefficients.
    """
    if cut_depth_db <= 0:
        return audio

    nyq = fs_hz / 2.0
    low = max(1.0, center_hz - width_hz / 2.0)
    high = min(nyq * 0.999, center_hz + width_hz / 2.0)
    if low >= high or low >= nyq:
        return audio  # invalid band; skip rather than raise

    sos = butter(order, [low, high], btype="bandpass", fs=fs_hz, output="sos")
    alpha = 1.0 - 10.0 ** (-cut_depth_db / 20.0)
    # Clip alpha to [0, 1]. Very-deep cuts (cut_depth_db -> inf) saturate
    # alpha -> 1 (hard band-stop). Tiny / negative cuts saturate alpha -> 0
    # (no-op). This prevents over-shoot for pathological cut depths.
    alpha = max(0.0, min(1.0, alpha))
    if alpha == 0.0:
        return audio

    if audio.ndim == 1:
        bp = sosfiltfilt(sos, audio)
        return (audio - alpha * bp).astype(audio.dtype, copy=False)

    if audio.ndim == 2:
        out = np.empty_like(audio)
        for ch in range(audio.shape[1]):
            bp = sosfiltfilt(sos, audio[:, ch])
            out[:, ch] = (audio[:, ch] - alpha * bp).astype(audio.dtype, copy=False)
        return out

    raise ValueError(f"Unsupported audio ndim={audio.ndim}; expected 1 or 2.")


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
        1. Welch PSD with ``nperseg`` (default 8192 -> ~37 Hz/bin at 300 kHz).
        2. For each freq bin in ``[usv_band_min_hz, usv_band_max_hz]``,
           compute a rolling median over a ``median_window_hz`` neighborhood.
        3. Find bins where ``pxx_db - rolling_median > discovery_threshold_db``.
        4. Cluster contiguous bins; for each cluster, center = argmax bin,
           width = contiguous span at peak - 3 dB.

    For multi-channel audio, PSD is computed on channel 0; tonals describe
    the rig, not channel-specific artefacts.
    """
    a = _select_channel(audio)
    f, pxx_db = _welch_psd_db(a, fs_hz, nperseg)
    if len(f) < 3:
        return []

    band_mask = (f >= usv_band_min_hz) & (f <= usv_band_max_hz)
    if band_mask.sum() < 3:
        return []
    band_f = f[band_mask]
    band_db = pxx_db[band_mask]

    # Welch returns uniformly-spaced frequencies; compute bin resolution.
    bin_res_hz = float(band_f[1] - band_f[0])
    half_window_bins = max(1, int(round((median_window_hz / 2.0) / bin_res_hz)))
    rolling_median = _rolling_median_db(band_db, half_window_bins)

    excess = band_db - rolling_median
    above = excess > discovery_threshold_db
    if not above.any():
        return []

    tonals: list[DetectedTonal] = []
    n = len(above)
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        # Cluster runs from `start` (inclusive) to `end` (exclusive).
        start = i
        while i < n and above[i]:
            i += 1
        end = i
        cluster_db = band_db[start:end]
        peak_idx_local = int(np.argmax(cluster_db))
        peak_idx = start + peak_idx_local
        peak_db = float(band_db[peak_idx])
        local_median_db = float(rolling_median[peak_idx])

        # -3 dB width: walk outward while PSD stays within 3 dB of peak.
        threshold_3db = peak_db - 3.0
        left = peak_idx
        while left > 0 and band_db[left - 1] >= threshold_3db:
            left -= 1
        right = peak_idx
        while right < n - 1 and band_db[right + 1] >= threshold_3db:
            right += 1
        width_hz = float(band_f[right] - band_f[left])
        if width_hz <= 0:
            width_hz = bin_res_hz  # at least one bin wide

        tonals.append(DetectedTonal(
            center_hz=float(band_f[peak_idx]),
            width_hz=width_hz,
            peak_db=peak_db,
            local_median_db=local_median_db,
            above_median_db=peak_db - local_median_db,
        ))

    return tonals


def reconcile(
    library: TonalLibrary,
    detections: Sequence[DetectedTonal],
    *,
    freq_tolerance_hz: float = 200.0,
    intensity_drift_sigma: float = 2.0,
) -> ReconciliationResult:
    """Reconcile per-chunk detections against the library.

    Matching is by center-frequency proximity (``freq_tolerance_hz``).
    Each detection can match at most one library entry (greedy nearest);
    each entry can match at most one detection.

    Intensity drift fires when a matched detection's ``above_median_db``
    deviates from ``library_entry.mean_above_median_db`` by more than
    ``intensity_drift_sigma`` standard deviations of the library entry.
    """
    detection_used = [False] * len(detections)
    matched: list[tuple[LibraryEntry, DetectedTonal]] = []
    unmatched_library_entries: list[LibraryEntry] = []
    intensity_drifts: list[tuple[LibraryEntry, float]] = []

    for entry in library.entries:
        best_idx = -1
        best_dist = math.inf
        for i, det in enumerate(detections):
            if detection_used[i]:
                continue
            dist = abs(det.center_hz - entry.center_hz)
            if dist <= freq_tolerance_hz and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            det = detections[best_idx]
            detection_used[best_idx] = True
            matched.append((entry, det))
            if entry.stdev_above_median_db > 0:
                sigma = abs(det.above_median_db - entry.mean_above_median_db) \
                    / entry.stdev_above_median_db
                if sigma > intensity_drift_sigma:
                    intensity_drifts.append((entry, sigma))
        else:
            unmatched_library_entries.append(entry)

    unmatched_detections = [
        det for i, det in enumerate(detections) if not detection_used[i]
    ]

    return ReconciliationResult(
        matched=matched,
        unmatched_detections=unmatched_detections,
        unmatched_library_entries=unmatched_library_entries,
        intensity_drifts=intensity_drifts,
    )


def apply_soft_notches(
    audio: np.ndarray,
    fs_hz: float,
    tonals: Sequence[Union[LibraryEntry, DetectedTonal]],
    *,
    min_width_hz: float = 200.0,
    width_safety_factor: float = 2.0,
    safety_margin_db: float = 0.0,
    order: int = 4,
    cut_depths_db: Union[Sequence[float], None] = None,
) -> np.ndarray:
    """Apply soft-notch filters at every supplied tonal.

    Each tonal's filter is a Butterworth band-pass at
    ``[center - width/2, center + width/2]`` (order ``order``, effective
    order ``2*order`` via ``sosfiltfilt``). The complementary-bandpass
    subtraction ``audio - alpha * bandpass(audio)`` is applied, with
    ``alpha`` derived from the cut depth.

    Cut depth per tonal:

    - If ``cut_depths_db`` is supplied (caller-measured, one per tonal),
      those values are used directly. This is the library-mode path:
      :func:`auto_soft_notch` measures cut depths from the local PSD around
      each library entry and passes them in.
    - Otherwise, for :class:`DetectedTonal` items, cut depth =
      ``above_median_db + safety_margin_db``.
    - Otherwise, for :class:`LibraryEntry` items without explicit
      ``cut_depths_db``: raises :class:`ValueError`. Library entries do not
      carry a cut depth — it must be measured per-chunk.

    Width:

    - For :class:`LibraryEntry`: ``width_hz`` is used directly
      (deterministic kill zone).
    - For :class:`DetectedTonal`: ``max(min_width_hz, measured *
      width_safety_factor)``.

    Multi-channel audio is filtered per-channel with the same SOS
    coefficients. Returns audio with the same shape and dtype as the input.
    """
    if len(tonals) == 0:
        return audio

    if cut_depths_db is not None and len(cut_depths_db) != len(tonals):
        raise ValueError(
            f"cut_depths_db length {len(cut_depths_db)} != tonals length {len(tonals)}"
        )

    # Cascade lowest-frequency first for predictable logging (spec L1).
    indices = sorted(range(len(tonals)), key=lambda i: tonals[i].center_hz)

    result = audio
    for tonal_idx in indices:
        t = tonals[tonal_idx]

        if isinstance(t, LibraryEntry):
            width = float(t.width_hz)
            if cut_depths_db is not None:
                cut_db = float(cut_depths_db[tonal_idx]) + safety_margin_db
            else:
                raise ValueError(
                    "LibraryEntry tonals require explicit cut_depths_db "
                    "(measured per-chunk from the local PSD)."
                )
        elif isinstance(t, DetectedTonal):
            width = max(min_width_hz, float(t.width_hz) * width_safety_factor)
            if cut_depths_db is not None:
                cut_db = float(cut_depths_db[tonal_idx]) + safety_margin_db
            else:
                cut_db = float(t.above_median_db) + safety_margin_db
        else:
            raise TypeError(
                f"Unsupported tonal type {type(t).__name__}; expected "
                "LibraryEntry or DetectedTonal."
            )

        result = _apply_one_band(
            result, fs_hz,
            center_hz=float(t.center_hz),
            width_hz=width,
            cut_depth_db=cut_db,
            order=order,
        )

    return result


def auto_soft_notch(
    audio: np.ndarray,
    fs_hz: float,
    library: Union[TonalLibrary, None] = None,
    *,
    usv_band_min_hz: float = float(USV_FREQ_MIN_HZ),
    usv_band_max_hz: float = float(USV_FREQ_MAX_HZ),
    discovery_threshold_db: float = 10.0,
    median_window_hz: float = 4_000.0,
    nperseg: int = 8192,
    freq_tolerance_hz: float = 200.0,
    intensity_drift_sigma: float = 2.0,
    min_width_hz: float = 200.0,
    width_safety_factor: float = 2.0,
    safety_margin_db: float = 0.0,
    order: int = 4,
) -> tuple[np.ndarray, ReconciliationResult]:
    """One-shot entry: discover + reconcile + filter.

    Behavior depends on whether a library is supplied:

    - ``library is None`` (pure auto-detect): every discovered tonal is
      filtered; ``reconciliation.matched`` and
      ``reconciliation.unmatched_library_entries`` are empty;
      ``reconciliation.unmatched_detections`` carries every discovered tonal
      (informational - they WERE filtered).
    - ``library is not None`` (library mode): library entries drive the
      filter; discovered tonals are reconciled against the library;
      unmatched detections are logged only (not filtered).

    Returns ``(cleaned_audio, reconciliation_result)``. When no tonals are
    discovered AND no library entries exist, ``cleaned_audio`` is the same
    ndarray as ``audio`` (byte-identical, including dtype).
    """
    detections = discover_tonals(
        audio, fs_hz,
        usv_band_min_hz=usv_band_min_hz,
        usv_band_max_hz=usv_band_max_hz,
        discovery_threshold_db=discovery_threshold_db,
        median_window_hz=median_window_hz,
        nperseg=nperseg,
    )

    if library is None:
        if not detections:
            return audio, ReconciliationResult([], [], [], [])
        cleaned = apply_soft_notches(
            audio, fs_hz, detections,
            min_width_hz=min_width_hz,
            width_safety_factor=width_safety_factor,
            safety_margin_db=safety_margin_db,
            order=order,
        )
        # Pure auto-detect: matched is empty by definition. unmatched_detections
        # carries the full discovery list (they were filtered; this surfaces
        # them to log lines for audit traceability).
        return cleaned, ReconciliationResult(
            matched=[],
            unmatched_detections=list(detections),
            unmatched_library_entries=[],
            intensity_drifts=[],
        )

    # ---------- Library mode ----------
    recon = reconcile(
        library, detections,
        freq_tolerance_hz=freq_tolerance_hz,
        intensity_drift_sigma=intensity_drift_sigma,
    )

    if not library.entries:
        return audio, recon

    # Per-chunk cut-depth measurement for each library entry.
    a = _select_channel(audio)
    f, pxx_db = _welch_psd_db(a, fs_hz, nperseg)
    cut_depths_db: list[float] = []
    for entry in library.entries:
        if len(f) == 0:
            cut_depths_db.append(0.0)
            continue
        peak_db, local_median_db = _measure_band_psd(
            f, pxx_db,
            band_lo_hz=entry.center_hz - entry.width_hz / 2.0,
            band_hi_hz=entry.center_hz + entry.width_hz / 2.0,
            median_window_hz=median_window_hz,
        )
        if math.isnan(peak_db) or math.isnan(local_median_db):
            cut_depths_db.append(0.0)
            continue
        cut_db = max(0.0, peak_db - local_median_db)
        cut_depths_db.append(cut_db)

    cleaned = apply_soft_notches(
        audio, fs_hz, library.entries,
        min_width_hz=min_width_hz,
        width_safety_factor=width_safety_factor,
        safety_margin_db=safety_margin_db,
        order=order,
        cut_depths_db=cut_depths_db,
    )
    return cleaned, recon
