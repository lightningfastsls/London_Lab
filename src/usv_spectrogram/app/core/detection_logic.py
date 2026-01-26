"""Hysteresis-based USV detection logic.

Converts CNN probability predictions into discrete USV detections using
hysteresis thresholding and merging of nearby events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DetectedUSV:
    """A single detected USV event."""

    start_time_s: float  # Start time in seconds
    end_time_s: float  # End time in seconds
    start_col: int  # Start column index
    end_col: int  # End column index
    max_probability: float  # Peak probability within event
    mean_probability: float  # Mean probability within event


@dataclass
class DetectionResult:
    """Container for detection results."""

    usvs: List[DetectedUSV]  # List of detected USV events
    probabilities: np.ndarray  # Original probability array
    column_indices: np.ndarray  # Column indices for probabilities
    times: np.ndarray  # Time values for probabilities


class HysteresisDetector:
    """Detects USV events using hysteresis thresholding.

    Hysteresis prevents spurious on/off transitions:
    - Detection starts when probability exceeds high_threshold
    - Detection continues until probability drops below low_threshold
    - Nearby detections (gap < merge_gap_columns) are merged
    """

    def __init__(
        self,
        high_threshold: float = 0.40,
        low_threshold: float | None = None,
        merge_gap_columns: int = 3
    ):
        """Initialize hysteresis detector.

        Args:
            high_threshold: Threshold to start detection
            low_threshold: Threshold to end detection (default: 0.7 × high_threshold)
            merge_gap_columns: Merge detections if gap < this many columns
        """
        self.high_threshold = high_threshold

        if low_threshold is None:
            self.low_threshold = 0.7 * high_threshold
        else:
            self.low_threshold = low_threshold

        self.merge_gap_columns = merge_gap_columns

        # Validate thresholds
        if not 0.0 <= self.low_threshold < self.high_threshold <= 1.0:
            raise ValueError(
                f"Must have 0 <= low_threshold ({self.low_threshold}) < "
                f"high_threshold ({self.high_threshold}) <= 1"
            )

    def detect(
        self,
        probabilities: np.ndarray,
        column_indices: np.ndarray,
        times: np.ndarray
    ) -> DetectionResult:
        """Detect USV events from probability predictions.

        Args:
            probabilities: Probability predictions, shape (n_windows,)
            column_indices: Column index for each window, shape (n_windows,)
            times: Time in seconds for each window, shape (n_windows,)

        Returns:
            DetectionResult containing detected USV events
        """
        if len(probabilities) == 0:
            return DetectionResult(
                usvs=[],
                probabilities=probabilities,
                column_indices=column_indices,
                times=times
            )

        # Run hysteresis detection
        raw_events = self._hysteresis_detect(probabilities, column_indices, times)

        # Merge nearby events
        merged_events = self._merge_nearby(raw_events)

        return DetectionResult(
            usvs=merged_events,
            probabilities=probabilities,
            column_indices=column_indices,
            times=times
        )

    def _hysteresis_detect(
        self,
        probabilities: np.ndarray,
        column_indices: np.ndarray,
        times: np.ndarray
    ) -> List[DetectedUSV]:
        """Apply hysteresis thresholding to find events.

        Args:
            probabilities: Probability predictions
            column_indices: Column indices
            times: Time values

        Returns:
            List of detected events (before merging)
        """
        events = []
        in_event = False
        event_start_idx = 0

        for i in range(len(probabilities)):
            prob = probabilities[i]

            if not in_event:
                # Check for event start
                if prob >= self.high_threshold:
                    in_event = True
                    event_start_idx = i
            else:
                # Check for event end
                if prob < self.low_threshold:
                    # Event ended, create detection
                    event = self._create_event(
                        event_start_idx,
                        i - 1,  # Last index above low threshold
                        probabilities,
                        column_indices,
                        times
                    )
                    events.append(event)
                    in_event = False

        # Handle event that extends to end
        if in_event:
            event = self._create_event(
                event_start_idx,
                len(probabilities) - 1,
                probabilities,
                column_indices,
                times
            )
            events.append(event)

        return events

    def _create_event(
        self,
        start_idx: int,
        end_idx: int,
        probabilities: np.ndarray,
        column_indices: np.ndarray,
        times: np.ndarray
    ) -> DetectedUSV:
        """Create DetectedUSV from index range.

        Args:
            start_idx: Start index in probability array
            end_idx: End index (inclusive)
            probabilities: Probability array
            column_indices: Column index array
            times: Time array

        Returns:
            DetectedUSV object
        """
        event_probs = probabilities[start_idx:end_idx + 1]

        return DetectedUSV(
            start_time_s=times[start_idx],
            end_time_s=times[end_idx],
            start_col=column_indices[start_idx],
            end_col=column_indices[end_idx],
            max_probability=float(event_probs.max()),
            mean_probability=float(event_probs.mean())
        )

    def _merge_nearby(self, events: List[DetectedUSV]) -> List[DetectedUSV]:
        """Merge events that are close together.

        Args:
            events: List of detected events

        Returns:
            List of merged events
        """
        if len(events) <= 1:
            return events

        # Sort by start time
        events = sorted(events, key=lambda e: e.start_time_s)

        merged = []
        current = events[0]

        for next_event in events[1:]:
            gap = next_event.start_col - current.end_col

            if gap <= self.merge_gap_columns:
                # Merge with current event
                current = DetectedUSV(
                    start_time_s=current.start_time_s,
                    end_time_s=next_event.end_time_s,
                    start_col=current.start_col,
                    end_col=next_event.end_col,
                    max_probability=max(
                        current.max_probability,
                        next_event.max_probability
                    ),
                    mean_probability=(
                        current.mean_probability + next_event.mean_probability
                    ) / 2.0
                )
            else:
                # Gap too large, save current and start new
                merged.append(current)
                current = next_event

        # Add final event
        merged.append(current)

        return merged
