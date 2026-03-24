"""Coordinate synchronization between LMT, WAV, and spectrogram domains.

LMT operates at 30 fps (1 frame = 33.333 ms). Our recordings are at 300 kHz.
Spectrograms use configurable hop_length (default 128 samples).

This module converts between these three coordinate systems and provides
event-detection alignment for correlating behavior with vocalizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .db_loader import BehavioralEvent


@dataclass(frozen=True)
class SyncConfig:
    """Configuration for LMT-to-WAV/spectrogram synchronization.

    Parameters
    ----------
    lmt_frame_rate : float
        LMT camera frame rate (fps). Standard LMT is 30.0.
    wav_sample_rate : int
        WAV recording sample rate (Hz). See ADR-001: 300 kHz canonical.
    time_offset_s : float
        Constant time offset to add when converting LMT frames to seconds.
        Accounts for synchronization delay between LMT and audio recording.
        Positive = LMT events are shifted later in WAV time.
    """

    lmt_frame_rate: float = 30.0
    wav_sample_rate: int = 300_000
    time_offset_s: float = 0.0

    def __post_init__(self) -> None:
        if self.lmt_frame_rate <= 0:
            raise ValueError("lmt_frame_rate must be positive")
        if self.wav_sample_rate <= 0:
            raise ValueError("wav_sample_rate must be positive")


# Event specificity ranking for the alignment algorithm.
# Higher rank = more specific = more informative behavioral context.
# Social/pairwise interactions are most informative for USV analysis.
_ENVIRONMENTAL_EVENTS = frozenset({
    "night", "Center", "Top", "Bottom", "Left", "Right",
})

_GENERAL_EVENTS = frozenset({
    "Contact", "Rearing", "Stop isolated", "Move isolated",
})


def _event_specificity(event: BehavioralEvent) -> int:
    """Rank event specificity (higher = more specific).

    Ranking heuristic:
        3 — Social/pairwise (partner_id is set)
        2 — Named behavioral action (not in general/environmental sets)
        1 — General behavioral state
        0 — Environmental/positional
    """
    if event.partner_id is not None:
        return 3
    if event.event_type in _ENVIRONMENTAL_EVENTS:
        return 0
    if event.event_type in _GENERAL_EVENTS:
        return 1
    return 2


class LMTSynchronizer:
    """Converts between LMT, WAV, and spectrogram coordinate systems.

    Also provides event-detection alignment: given a list of behavioral events
    and a list of USV detections, find which behaviors co-occur with each
    vocalization.

    Parameters
    ----------
    config : SyncConfig
        Synchronization parameters.
    """

    def __init__(self, config: Optional[SyncConfig] = None) -> None:
        self._config = config or SyncConfig()

    @property
    def config(self) -> SyncConfig:
        return self._config

    def lmt_frame_to_seconds(self, frame: int) -> float:
        """Convert an LMT frame number to seconds.

        Applies the configured time offset for LMT-to-WAV synchronization.
        """
        return frame / self._config.lmt_frame_rate + self._config.time_offset_s

    def seconds_to_wav_sample(self, time_s: float) -> int:
        """Convert a time in seconds to a WAV sample index."""
        return round(time_s * self._config.wav_sample_rate)

    def seconds_to_spectrogram_frame(
        self, time_s: float, hop_length: int = 128
    ) -> int:
        """Convert a time in seconds to a spectrogram frame index.

        Uses floor (int truncation) per ADR-002 convention:
        1.0s at sr=300000, hop=128 → int(300000/128) = 2343.

        Parameters
        ----------
        time_s : float
            Time in seconds.
        hop_length : int
            STFT hop length in samples. Default 128 per ADR-002.
        """
        return int(time_s * self._config.wav_sample_rate / hop_length)

    def align_events_with_detections(
        self,
        events: list[BehavioralEvent],
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Align behavioral events with USV detections by time overlap.

        For each detection, finds all behavioral events whose time range
        overlaps with the detection window, ranks them by specificity, and
        returns the enriched detection with behavioral context.

        Parameters
        ----------
        events : list[BehavioralEvent]
            Behavioral events (from LMTDatabaseLoader.get_events).
        detections : list[dict]
            USV detections. Each dict must have 'start_time_s' and 'end_time_s'
            keys (in seconds). Works with any dict — intentionally decoupled
            from DetectedUSV.

        Returns
        -------
        list[dict]
            Copy of each detection dict with two added keys:
            - 'behavioral_context': list of overlapping BehavioralEvent objects,
              sorted by specificity (most specific first)
            - 'dominant_event': the most specific overlapping event, or None
        """
        results = []
        for det in detections:
            det_start = det["start_time_s"]
            det_end = det["end_time_s"]

            # Find overlapping events.
            # Overlap condition: event starts before detection ends AND
            #                    event ends after detection starts.
            overlapping = [
                ev for ev in events
                if ev.start_time_s < det_end and ev.end_time_s > det_start
            ]

            # Sort by specificity (most specific first)
            overlapping.sort(key=_event_specificity, reverse=True)

            enriched = {
                **det,
                "behavioral_context": overlapping,
                "dominant_event": overlapping[0] if overlapping else None,
            }
            results.append(enriched)

        return results
