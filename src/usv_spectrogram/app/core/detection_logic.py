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
    # Track if user manually adjusted boundaries
    user_adjusted: bool = False
    original_start_time_s: float | None = None
    original_end_time_s: float | None = None


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
        merge_gap_columns: int = 3,
        min_duration_ms: float = 10.0,
        max_duration_ms: float = 500.0,
        min_sustained_prob: float = 0.80,
        exclude_start_sec: float = 0.1,
        exclude_end_sec: float = 0.1
    ):
        """Initialize hysteresis detector.

        Args:
            high_threshold: Threshold to start detection
            low_threshold: Threshold to end detection (default: 0.7 × high_threshold)
            merge_gap_columns: Merge detections if gap < this many columns
            min_duration_ms: Minimum USV duration in milliseconds (default: 10ms)
            max_duration_ms: Maximum USV duration in milliseconds (default: 500ms)
            min_sustained_prob: Minimum probability within detection (default: 0.80)
                               Rejects detections with jagged probability traces (false positives)
                               Set to 0.0 to disable
            exclude_start_sec: Exclude detections in first N seconds (default: 0.1s)
                              Removes recording startup artifacts. Set to 0.0 to disable
            exclude_end_sec: Exclude detections in last N seconds (default: 0.1s)
                            Removes recording shutdown artifacts. Set to 0.0 to disable
        """
        self.high_threshold = high_threshold

        if low_threshold is None:
            self.low_threshold = 0.7 * high_threshold
        else:
            self.low_threshold = low_threshold

        self.merge_gap_columns = merge_gap_columns
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.min_sustained_prob = min_sustained_prob
        self.exclude_start_sec = exclude_start_sec
        self.exclude_end_sec = exclude_end_sec

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

        # Filter by duration (reject too short/long events)
        filtered_events = self._filter_by_duration(raw_events)

        # Merge nearby events (do this BEFORE stability filter to catch gaps)
        merged_events = self._merge_nearby(filtered_events)

        # Filter by probability stability (reject jagged traces)
        # Must run AFTER merge because merging creates gaps with low probability
        merged_events = self._filter_by_probability_stability(
            merged_events, probabilities, column_indices
        )

        # Filter by temporal position (reject edge artifacts)
        merged_events = self._filter_by_temporal_position(merged_events, times)

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

    def _filter_by_duration(self, events: List[DetectedUSV]) -> List[DetectedUSV]:
        """Filter events by duration.

        Mouse USVs are typically 10-200ms. Events outside the duration range
        are likely noise artifacts or non-USV vocalizations.

        Args:
            events: List of detected events

        Returns:
            Filtered list of events within duration range
        """
        filtered = []
        rejected_too_short = 0
        rejected_too_long = 0

        for event in events:
            duration_ms = (event.end_time_s - event.start_time_s) * 1000.0
            if duration_ms < self.min_duration_ms:
                rejected_too_short += 1
            elif duration_ms > self.max_duration_ms:
                rejected_too_long += 1
            else:
                filtered.append(event)

        # Debug output
        if rejected_too_short > 0 or rejected_too_long > 0:
            print(f"[DurationFilter] Input events: {len(events)}")
            print(f"[DurationFilter] Rejected too short (< {self.min_duration_ms}ms): {rejected_too_short}")
            print(f"[DurationFilter] Rejected too long (> {self.max_duration_ms}ms): {rejected_too_long}")
            print(f"[DurationFilter] Kept: {len(filtered)}")

        return filtered

    def _filter_by_probability_stability(
        self,
        events: List[DetectedUSV],
        probabilities: np.ndarray,
        column_indices: np.ndarray
    ) -> List[DetectedUSV]:
        """Filter events by probability stability.

        Real USVs have sustained high probability throughout the event.
        False positives have jagged, intermittent probability traces.

        Args:
            events: List of detected events
            probabilities: Probability array
            column_indices: Column index array

        Returns:
            Filtered list of events with stable probability traces
        """
        if self.min_sustained_prob <= 0.0:
            return events  # Filter disabled

        filtered = []
        rejected_unstable = 0
        event_stats = []  # Store stats for first few events

        for i, event in enumerate(events):
            # Find probability slice for this event
            # Match column indices to find corresponding probability indices
            start_mask = column_indices >= event.start_col
            end_mask = column_indices <= event.end_col
            event_mask = start_mask & end_mask
            event_probs = probabilities[event_mask]

            if len(event_probs) == 0:
                # Should not happen, but skip if no probabilities found
                continue

            min_prob = event_probs.min()
            median_prob = np.median(event_probs)
            max_prob = event_probs.max()

            # Store stats for first 5 events for debugging
            if i < 5:
                event_stats.append({
                    'index': i,
                    'start_time': event.start_time_s,
                    'duration_ms': (event.end_time_s - event.start_time_s) * 1000,
                    'n_windows': len(event_probs),
                    'min_prob': min_prob,
                    'median_prob': median_prob,
                    'max_prob': max_prob
                })

            if min_prob >= self.min_sustained_prob:
                filtered.append(event)
            else:
                rejected_unstable += 1

        # Debug output (always print to track filter activity)
        if self.min_sustained_prob > 0.0:
            print(f"[ProbabilityStabilityFilter] Input events: {len(events)}")
            print(f"[ProbabilityStabilityFilter] Rejected unstable (min < {self.min_sustained_prob}): {rejected_unstable}")
            print(f"[ProbabilityStabilityFilter] Kept: {len(filtered)}")

            # Print detailed stats for first few events
            if len(event_stats) > 0:
                print(f"[ProbabilityStabilityFilter] Sample event stats (first {len(event_stats)}):")
                for stat in event_stats:
                    print(f"  Event {stat['index']}: t={stat['start_time']:.2f}s, dur={stat['duration_ms']:.1f}ms, "
                          f"windows={stat['n_windows']}, prob: min={stat['min_prob']:.3f}, "
                          f"med={stat['median_prob']:.3f}, max={stat['max_prob']:.3f}")

        return filtered

    def _filter_by_temporal_position(
        self,
        events: List[DetectedUSV],
        times: np.ndarray
    ) -> List[DetectedUSV]:
        """Filter events by temporal position.

        Reject detections too close to file boundaries to remove recording
        hardware transients (microphone startup/shutdown artifacts).

        Args:
            events: List of detected events
            times: Time array in seconds

        Returns:
            Filtered list of events away from file edges
        """
        if self.exclude_start_sec <= 0.0 and self.exclude_end_sec <= 0.0:
            return events  # Filter disabled

        if len(times) == 0:
            return events

        audio_duration = times[-1]
        filtered = []
        rejected_at_start = 0
        rejected_at_end = 0
        edge_events = []  # Track events near edges

        for event in events:
            # Check if event overlaps with excluded regions
            if event.start_time_s < self.exclude_start_sec:
                rejected_at_start += 1
            elif event.end_time_s > (audio_duration - self.exclude_end_sec):
                rejected_at_end += 1
            else:
                filtered.append(event)

            # Track events within 1 second of edges for debugging
            if event.start_time_s < 1.0 or event.end_time_s > (audio_duration - 1.0):
                edge_events.append({
                    'start_time': event.start_time_s,
                    'end_time': event.end_time_s,
                    'location': 'START' if event.start_time_s < 1.0 else 'END'
                })

        # Debug output (always print to track filter activity)
        if self.exclude_start_sec > 0.0 or self.exclude_end_sec > 0.0:
            print(f"[TemporalPositionFilter] Audio duration: {audio_duration:.2f}s")
            print(f"[TemporalPositionFilter] Exclusion zones: first {self.exclude_start_sec}s, last {self.exclude_end_sec}s")
            print(f"[TemporalPositionFilter] Input events: {len(events)}")
            print(f"[TemporalPositionFilter] Rejected at start (< {self.exclude_start_sec}s): {rejected_at_start}")
            print(f"[TemporalPositionFilter] Rejected at end (> {audio_duration - self.exclude_end_sec:.2f}s): {rejected_at_end}")
            print(f"[TemporalPositionFilter] Kept: {len(filtered)}")

            # Show events near edges (within 1 second)
            if len(edge_events) > 0:
                print(f"[TemporalPositionFilter] Events near edges (within 1s):")
                for i, evt in enumerate(edge_events[:5]):  # Show first 5
                    print(f"  Event @ {evt['start_time']:.3f}s - {evt['end_time']:.3f}s ({evt['location']})")

        return filtered

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
        merge_count = 0
        gaps = []

        for next_event in events[1:]:
            gap = next_event.start_col - current.end_col
            gaps.append(gap)

            if gap <= self.merge_gap_columns:
                # Merge with current event
                merge_count += 1
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

        # Debug output
        if len(gaps) > 0:
            gaps_array = np.array(gaps)
            print(f"[MergeNearby] Input events: {len(events)}")
            print(f"[MergeNearby] Merge threshold: {self.merge_gap_columns} columns")
            print(f"[MergeNearby] Gaps between events - min: {gaps_array.min()}, median: {np.median(gaps_array):.1f}, max: {gaps_array.max()}")
            print(f"[MergeNearby] Merges performed: {merge_count}")
            print(f"[MergeNearby] Output events: {len(merged)}")

        return merged
