"""Hebrew-to-English vehicle name mapper.

Translates Hebrew manufacturer and model names from the Israeli government
vehicle registry to canonical English equivalents, using a curated JSON
mapping file.  Unknown names return ``None`` and are logged as warnings
so the mapping file can be extended over time.

Usage::

    mapper = NameMapper(Path("data/hebrew_names.json"))
    enriched = mapper.enrich_vehicle(vehicle_record)
    print(enriched.make_english)   # "Toyota"
    print(enriched.model_english)  # "Corolla"
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

from parts_finder.models import VehicleRecord

logger = logging.getLogger(__name__)


class NameMapper:
    """Translate Hebrew vehicle names to canonical English using a JSON map.

    The mapping file must contain ``"makes"`` and ``"models"`` top-level keys,
    each mapping Hebrew strings to their English equivalents.  Keys are
    normalized (stripped of leading/trailing whitespace) at load time for
    reliable matching.

    Parameters
    ----------
    mapping_path:
        Path to the JSON mapping file.
    """

    def __init__(self, mapping_path: Path) -> None:
        with open(mapping_path, encoding="utf-8") as fh:
            data = json.load(fh)

        try:
            self._makes: dict[str, str] = {
                k.strip(): v for k, v in data["makes"].items()
            }
            self._models: dict[str, str] = {
                k.strip(): v for k, v in data["models"].items()
            }
        except KeyError as exc:
            raise ValueError(
                f"Mapping file {mapping_path} missing required key {exc}. "
                "Expected top-level keys: 'makes' and 'models'."
            ) from exc

    def translate_make(self, hebrew_make: str) -> Optional[str]:
        """Return the English equivalent for a Hebrew manufacturer name.

        Returns ``None`` if no mapping exists (and logs a WARNING so the
        mapping file can be extended).  Empty or whitespace-only input
        returns ``None`` with no warning.
        """
        if not hebrew_make or not hebrew_make.strip():
            return None

        stripped = hebrew_make.strip()
        result = self._makes.get(stripped)
        if result is None:
            logger.warning("Unmapped make: %r", stripped)
        return result

    def translate_model(self, hebrew_model: str) -> Optional[str]:
        """Return the English equivalent for a Hebrew model name.

        Returns ``None`` if no mapping exists (and logs a WARNING so the
        mapping file can be extended).  Empty or whitespace-only input
        returns ``None`` with no warning.
        """
        if not hebrew_model or not hebrew_model.strip():
            return None

        stripped = hebrew_model.strip()
        result = self._models.get(stripped)
        if result is None:
            logger.warning("Unmapped model: %r", stripped)
        return result

    def enrich_vehicle(self, record: VehicleRecord) -> VehicleRecord:
        """Return a new VehicleRecord with English names populated.

        Our curated mapping takes precedence over API-provided English names
        (which are inconsistent).  If a name is *not* in our mapping (returns
        ``None``), any existing API-provided English name is preserved.
        """
        translated_make = self.translate_make(record.make_hebrew)
        translated_model = self.translate_model(record.model_hebrew)

        # None means "no mapping found" — fall back to the API value.
        make_en = translated_make if translated_make is not None else record.make_english
        model_en = translated_model if translated_model is not None else record.model_english

        return replace(record, make_english=make_en, model_english=model_en)
