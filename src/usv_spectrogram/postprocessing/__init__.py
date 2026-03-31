"""Post-processing modules for USV detection pipeline.

Batch-processing utilities that operate on CNN output probabilities
to produce discrete USV events.
"""

from .calibration import TemperatureScaler, compute_ece
from .event_features import EventFeatures, extract_event_features
from .event_scoring import EventScoringConfig, compute_f_beta, match_events_collar
from .fp_filter import FalsePositiveFilter
from .hysteresis import HysteresisConfig, USVEvent, convert_to_detection_format, hysteresis_detect
from .normalization import normalize_scores_batch, normalize_scores_per_recording
from .triage import RecordingResult, TriageConfig, triage_recording
from .batch_output import write_batch_results

__all__ = [
    "EventFeatures",
    "EventScoringConfig",
    "FalsePositiveFilter",
    "HysteresisConfig",
    "TemperatureScaler",
    "USVEvent",
    "compute_ece",
    "compute_f_beta",
    "convert_to_detection_format",
    "extract_event_features",
    "hysteresis_detect",
    "match_events_collar",
    "normalize_scores_batch",
    "normalize_scores_per_recording",
    "RecordingResult",
    "TriageConfig",
    "triage_recording",
    "write_batch_results",
]
