"""USV detection pipeline for candidate generation and processing."""

from .config import DetectionConfig
from .candidate import Candidate
from .energy_detector import EnergyDetector

__all__ = ["DetectionConfig", "Candidate", "EnergyDetector"]
