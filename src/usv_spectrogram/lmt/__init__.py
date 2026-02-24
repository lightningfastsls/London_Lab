"""LMT (Live Mouse Tracker) data access layer.

Provides Python API for loading behavioral events from LMT SQLite databases
and aligning their timestamps with WAV/spectrogram coordinate systems.

LMT records frame-by-frame behavioral annotations at 30 fps. This module
handles the coordinate conversion between LMT frames, wall-clock seconds,
WAV samples (300 kHz), and spectrogram frames (hop_length-dependent).
"""

from .db_loader import BehavioralEvent, AnimalInfo, LMTDatabaseLoader
from .synchronizer import SyncConfig, LMTSynchronizer

__all__ = [
    "BehavioralEvent",
    "AnimalInfo",
    "LMTDatabaseLoader",
    "SyncConfig",
    "LMTSynchronizer",
]
