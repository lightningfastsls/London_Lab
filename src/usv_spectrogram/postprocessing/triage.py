"""Recording-level triage for batch USV detection.

Assigns each recording to a tier (auto_accept / auto_reject / manual_review)
based on detection confidence and QC metrics.  Designed for batch runs of
25 000+ recordings where manual review of every file is impractical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .hysteresis import USVEvent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriageConfig:
    """Thresholds for automatic triage decisions.

    Attributes
    ----------
    auto_accept_min_peak : float
        Minimum *peak_probability* every event must exceed for the recording
        to be auto-accepted.  Must be in (0, 1].
    auto_reject_max_window : float
        Maximum window probability allowed across the entire recording for
        it to be auto-rejected (i.e. "clearly empty").  Must be >= 0.
    noise_floor_p90_threshold : float
        If the 90th-percentile window probability exceeds this value a
        ``"high_noise_floor"`` QC flag is raised.
    outlier_count_zscore : float
        When batch statistics are available, recordings whose event count
        exceeds ``mean + zscore * std`` are flagged as outliers.
    max_event_duration_ms : float
        Any event longer than this is flagged for manual review.
    total_duration_review_ms : float
        Recordings whose summed detected duration exceeds this are flagged.
    high_event_count_threshold : int
        Recordings with more events than this are flagged.
    max_event_fraction_of_recording : float
        Events spanning at least this fraction of the probability timeline are flagged.
    """

    auto_accept_min_peak: float = 0.90
    auto_reject_max_window: float = 0.10
    noise_floor_p90_threshold: float = 0.4
    outlier_count_zscore: float = 2.0
    max_event_duration_ms: float = 600.0
    total_duration_review_ms: float = 600.0
    high_event_count_threshold: int = 10
    max_event_fraction_of_recording: float = 0.8

    def __post_init__(self) -> None:
        if self.auto_accept_min_peak <= 0.0:
            raise ValueError(
                f"auto_accept_min_peak must be > 0, got {self.auto_accept_min_peak}"
            )
        if self.auto_reject_max_window < 0.0:
            raise ValueError(
                f"auto_reject_max_window must be >= 0, got {self.auto_reject_max_window}"
            )
        if self.auto_reject_max_window >= self.auto_accept_min_peak:
            raise ValueError(
                f"auto_reject_max_window ({self.auto_reject_max_window}) must be "
                f"strictly less than auto_accept_min_peak ({self.auto_accept_min_peak})"
            )
        if self.max_event_duration_ms <= 0.0:
            raise ValueError(
                f"max_event_duration_ms must be > 0, got {self.max_event_duration_ms}"
            )
        if self.total_duration_review_ms <= 0.0:
            raise ValueError(
                f"total_duration_review_ms must be > 0, got {self.total_duration_review_ms}"
            )
        if self.high_event_count_threshold < 1:
            raise ValueError(
                f"high_event_count_threshold must be >= 1, got {self.high_event_count_threshold}"
            )
        if not (0.0 < self.max_event_fraction_of_recording <= 1.0):
            raise ValueError(
                "max_event_fraction_of_recording must be in (0, 1], "
                f"got {self.max_event_fraction_of_recording}"
            )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RecordingResult:
    """Triage outcome and QC metrics for a single recording.

    Attributes
    ----------
    filepath : str
        Path to the source WAV file.
    events : list[USVEvent]
        Detected USV events (after hysteresis + optional FP filtering).
    tier : str
        One of ``'auto_accept'``, ``'auto_reject'``, ``'manual_review'``.
    confidence_score : float
        Summary confidence metric (currently ``max_confidence``).
    qc_flags : list[str]
        Machine-readable QC flags (e.g. ``"high_noise_floor"``).
    n_events : int
        Number of detected events.
    max_confidence : float
        Maximum ``peak_probability`` across all events (0.0 if none).
    mean_event_confidence : float
        Mean of per-event ``peak_probability`` (0.0 if none).
    total_usv_duration_ms : float
        Sum of ``duration_ms`` across all events.
    noise_floor_p90 : float
        90th percentile of the raw window-probability array.
    """

    filepath: str
    events: List[USVEvent]
    tier: str
    confidence_score: float
    qc_flags: List[str]
    n_events: int
    max_confidence: float
    mean_event_confidence: float
    total_usv_duration_ms: float
    noise_floor_p90: float


# ---------------------------------------------------------------------------
# Triage logic
# ---------------------------------------------------------------------------

def triage_recording(
    filepath: str,
    events: List[USVEvent],
    probabilities: np.ndarray,
    config: TriageConfig | None = None,
    batch_stats: Optional[Dict[str, float]] = None,
) -> RecordingResult:
    """Assign a recording to a triage tier and compute QC metrics.

    Parameters
    ----------
    filepath : str
        Path to the WAV file (stored verbatim in the result).
    events : list[USVEvent]
        Detected events from ``hysteresis_detect`` (possibly filtered).
    probabilities : np.ndarray
        1-D array of per-window probabilities from ``SlidingInference``.
    config : TriageConfig, optional
        Triage thresholds.  Defaults to ``TriageConfig()``.
    batch_stats : dict, optional
        Keys ``"event_count_mean"`` and ``"event_count_std"`` from a prior
        batch run, used for outlier flagging.

    Returns
    -------
    RecordingResult
    """
    if config is None:
        config = TriageConfig()

    # -- QC metrics ----------------------------------------------------------
    n_events = len(events)

    if n_events > 0:
        max_confidence = float(max(e.peak_probability for e in events))
        mean_event_confidence = float(
            np.mean([e.peak_probability for e in events])
        )
        total_usv_duration_ms = float(sum(e.duration_ms for e in events))
    else:
        max_confidence = 0.0
        mean_event_confidence = 0.0
        total_usv_duration_ms = 0.0

    noise_floor_p90 = float(np.percentile(probabilities, 90))

    # -- QC flags ------------------------------------------------------------
    qc_flags: List[str] = []

    # Outlier event count (only when batch statistics are available)
    if batch_stats is not None:
        std = batch_stats.get("event_count_std", 0.0)
        if std > 0.0:
            mean = batch_stats["event_count_mean"]
            z = (n_events - mean) / std
            if z > config.outlier_count_zscore:
                qc_flags.append("outlier_event_count")

    # High noise floor
    if noise_floor_p90 > config.noise_floor_p90_threshold:
        qc_flags.append("high_noise_floor")

    if n_events > config.high_event_count_threshold:
        qc_flags.append("high_event_count")

    if any(e.duration_ms > config.max_event_duration_ms for e in events):
        qc_flags.append("long_event_duration")

    if total_usv_duration_ms > config.total_duration_review_ms:
        qc_flags.append("high_total_usv_duration")

    if probabilities.size > 0 and events:
        max_fraction = max(e.window_count / probabilities.size for e in events)
        if max_fraction >= config.max_event_fraction_of_recording:
            qc_flags.append("event_spans_most_of_recording")

    # -- Tier assignment (order matters) -------------------------------------
    prob_max = float(np.max(probabilities)) if probabilities.size > 0 else 0.0

    if n_events == 0:
        # Resolved Ambiguity #6: no events = no USVs detected → auto_reject
        tier = "auto_reject"
    elif prob_max <= config.auto_reject_max_window:
        tier = "auto_reject"
    elif {"long_event_duration", "event_spans_most_of_recording"} & set(qc_flags):
        tier = "manual_review"
    elif all(e.peak_probability >= config.auto_accept_min_peak for e in events):
        tier = "auto_accept"
    else:
        tier = "manual_review"

    # -- Assemble result -----------------------------------------------------
    confidence_score = mean_event_confidence

    return RecordingResult(
        filepath=filepath,
        events=events,
        tier=tier,
        confidence_score=confidence_score,
        qc_flags=qc_flags,
        n_events=n_events,
        max_confidence=max_confidence,
        mean_event_confidence=mean_event_confidence,
        total_usv_duration_ms=total_usv_duration_ms,
        noise_floor_p90=noise_floor_p90,
    )
