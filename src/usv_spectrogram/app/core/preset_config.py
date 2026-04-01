"""Threshold preset configuration for progressive labeling workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class ThresholdPreset:
    """A named threshold preset."""
    name: str
    high_threshold: float
    low_threshold: float

    def validate(self) -> bool:
        """Check that thresholds are valid (high >= low, both in [0, 1])."""
        return (
            0.0 <= self.low_threshold < self.high_threshold <= 1.0
        )


# Default presets calibrated for hard-neg retrained CNN (2026-03-31).
# Derived from hysteresis optimization on 220 labeled recordings (F2=0.867).
DEFAULT_PRESETS: List[ThresholdPreset] = [
    ThresholdPreset(name="Conservative (1SD)", high_threshold=0.80, low_threshold=0.50),
    ThresholdPreset(name="Best F2", high_threshold=0.60, low_threshold=0.40),
    ThresholdPreset(name="High Recall", high_threshold=0.50, low_threshold=0.30),
]


class PresetManager:
    """Manage threshold presets with JSON persistence."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize preset manager.

        Args:
            config_dir: Directory for preset_config.json.
                        Defaults to app's core directory.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_path = config_dir / "preset_config.json"
        self.presets: List[ThresholdPreset] = self._load_presets()

    def _load_presets(self) -> List[ThresholdPreset]:
        """Load presets from JSON, falling back to defaults."""
        if not self.config_path.exists():
            return list(DEFAULT_PRESETS)

        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)

            presets = [ThresholdPreset(**item) for item in data]
            # Validate all loaded presets
            if all(p.validate() for p in presets) and len(presets) > 0:
                return presets
            else:
                print(f"[PresetManager] Invalid presets in {self.config_path}, using defaults")
                return list(DEFAULT_PRESETS)

        except Exception as e:
            print(f"[PresetManager] Could not load {self.config_path}: {e}, using defaults")
            return list(DEFAULT_PRESETS)

    def save_presets(self):
        """Persist current presets to JSON."""
        try:
            data = [asdict(p) for p in self.presets]
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[PresetManager] Error saving presets: {e}")

    def get_preset(self, name: str) -> ThresholdPreset | None:
        """Get preset by name."""
        for p in self.presets:
            if p.name == name:
                return p
        return None
