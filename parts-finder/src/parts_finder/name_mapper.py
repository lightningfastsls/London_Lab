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

import csv
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

        # fuel_types is optional — older mapping files may not have it
        self._fuel_types: dict[str, str] = {
            k.strip(): v for k, v in data.get("fuel_types", {}).items()
        }

        # Model code prefix mappings — loaded from *_model_mapping.csv files
        # next to the JSON mapping file.  Each CSV maps degem_nm prefixes to
        # canonical model/series names for a specific make.
        self._model_prefix_maps: dict[str, list[tuple[str, str]]] = {}
        self._load_model_mappings(mapping_path.parent)

    def _load_model_mappings(self, data_dir: Path) -> None:
        """Load make-specific degem_nm prefix → model CSVs from *data_dir*/seed/."""
        seed_dir = data_dir / "seed"
        if not seed_dir.is_dir():
            return
        for csv_path in seed_dir.glob("*_model_mapping.csv"):
            # Extract make from filename: "bmw_model_mapping.csv" → "BMW"
            make_key = csv_path.stem.replace("_model_mapping", "")
            # Resolve to the canonical English make name
            make_canonical = self._resolve_make_from_filename(make_key)
            if make_canonical is None:
                logger.warning("Cannot resolve make for mapping file: %s", csv_path.name)
                continue
            entries: list[tuple[str, str]] = []
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    prefix = row.get("api_prefix", "").strip()
                    # Column is 'series' for BMW, 'model' for others
                    model = (row.get("series") or row.get("model") or "").strip()
                    if prefix and model:
                        entries.append((prefix, model))
            # Sort by prefix length descending so longer prefixes match first
            entries.sort(key=lambda e: len(e[0]), reverse=True)
            self._model_prefix_maps[make_canonical] = entries
            logger.info(
                "Loaded %d model prefix mappings for %s from %s",
                len(entries), make_canonical, csv_path.name,
            )

    def _resolve_make_from_filename(self, filename_stem: str) -> Optional[str]:
        """Map a filename stem like 'bmw' to the canonical English make 'BMW'.

        Tries exact (case-insensitive) match first, then prefix match to
        handle names like 'mercedes' -> 'Mercedes-Benz'.
        """
        lower_to_english = {v.lower(): v for v in self._makes.values()}
        exact = lower_to_english.get(filename_stem.lower())
        if exact:
            return exact
        # Prefix match: 'mercedes' matches 'mercedes-benz'
        stem_lower = filename_stem.lower()
        for canonical_lower, canonical in lower_to_english.items():
            if canonical_lower.startswith(stem_lower):
                return canonical
        return None

    def translate_make(self, hebrew_make: str) -> Optional[str]:
        """Return the English equivalent for a Hebrew manufacturer name.

        Tries exact match first, then prefix match.  The government API
        sometimes appends a country of manufacture to the brand name
        (e.g. ``"קיה סלובקיה"`` for "Kia Slovakia"), so after an exact
        miss we check if the input *starts with* any known make.

        Returns ``None`` if no mapping exists (and logs a WARNING so the
        mapping file can be extended).  Empty or whitespace-only input
        returns ``None`` with no warning.
        """
        if not hebrew_make or not hebrew_make.strip():
            return None

        stripped = hebrew_make.strip()

        # Exact match (fast path)
        result = self._makes.get(stripped)
        if result is not None:
            return result

        # Prefix match — handles "brand + country" variants
        for hebrew_key, english_val in self._makes.items():
            if stripped.startswith(hebrew_key):
                logger.info(
                    "Prefix-matched make: %r -> %r (key=%r)",
                    stripped, english_val, hebrew_key,
                )
                return english_val

        logger.warning("Unmapped make: %r", stripped)
        return None

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

    def resolve_model_from_code(self, make_english: str, degem_nm: str) -> Optional[str]:
        """Resolve a model name from the internal degem_nm code using prefix maps.

        For makes like BMW and Volvo, the gov API returns internal model codes
        (e.g. ``"31xxxx"`` for BMW 3 Series, ``"MV..."`` for Volvo XC60).
        This method checks the first N characters against known prefix tables.

        Returns the canonical model/series name, or ``None`` if no mapping.
        """
        if not make_english or not degem_nm:
            return None
        entries = self._model_prefix_maps.get(make_english)
        if not entries:
            return None
        code = degem_nm.strip().upper()
        for prefix, model in entries:
            if code.startswith(prefix.upper()):
                logger.info(
                    "Resolved model from degem_nm: %r prefix=%r -> %r (make=%s)",
                    degem_nm, prefix, model, make_english,
                )
                return model
        return None

    def translate_fuel_type(self, hebrew_fuel: str) -> Optional[str]:
        """Return the English equivalent for a Hebrew fuel type.

        Returns ``None`` if no mapping exists.  Empty or whitespace-only
        input returns ``None`` with no warning.
        """
        if not hebrew_fuel or not hebrew_fuel.strip():
            return None
        stripped = hebrew_fuel.strip()
        result = self._fuel_types.get(stripped)
        if result is None and not stripped.isascii():
            logger.warning("Unmapped fuel type: %r", stripped)
        return result

    def enrich_vehicle(self, record: VehicleRecord) -> VehicleRecord:
        """Return a new VehicleRecord with English names populated.

        Our curated mapping takes precedence over API-provided English names
        (which are inconsistent).  If a name is *not* in our mapping (returns
        ``None``), any existing API-provided English name is preserved.

        Also translates the Hebrew fuel_type to its English equivalent
        (e.g. ``"בנזין"`` → ``"petrol"``).
        """
        translated_make = self.translate_make(record.make_hebrew)
        translated_model = self.translate_model(record.model_hebrew)

        # None means "no mapping found" — fall back to the API value.
        make_en = translated_make if translated_make is not None else record.make_english
        model_en = translated_model if translated_model is not None else record.model_english

        # If model is still unknown, try resolving from degem_nm prefix maps
        if model_en is None and make_en and record.degem_nm:
            model_en = self.resolve_model_from_code(make_en, record.degem_nm)

        # Translate fuel type (Hebrew → English) if mapping exists
        translated_fuel = self.translate_fuel_type(record.fuel_type)
        fuel = translated_fuel if translated_fuel is not None else record.fuel_type

        return replace(
            record,
            make_english=make_en,
            model_english=model_en,
            fuel_type=fuel,
        )
