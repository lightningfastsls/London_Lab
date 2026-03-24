"""Coolant specification lookup with three-tier cascade matching.

Given a VehicleRecord (from the government API), resolves coolant
specifications using a cascade:

1. **Exact match** — make + model + year + engine_code
2. **Model-year fallback** — make + model + year (any engine)
3. **Brand default** — hardcoded brand-level coolant spec

Unlike oil (which aggregates brand defaults from the DB), coolant
specs come from a hardcoded brand knowledge base because coolant
technology standards are manufacturer-wide conventions (e.g., all
VW group vehicles use G13 Si-OAT).

The unique feature is a **compatibility matrix** that generates
mixing warnings — incorrect coolant mixing causes gel formation,
corrosion, and engine damage.

Usage::

    with PartsDatabase("parts.db") as db:
        coolant = CoolantLookup(db)
        result = coolant.lookup(vehicle)
        if result:
            print(f"{result.spec} ({result.color})")
            print(result.mixing_warning)
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple, Optional

from parts_finder.db import PartsDatabase
from parts_finder.lookup._shared import resolve_vehicle_names
from parts_finder.models import CoolantResult, VehicleRecord, VehicleSpecs

logger = logging.getLogger(__name__)


class CoolantSpec(NamedTuple):
    """Internal representation of a coolant specification."""

    spec_name: str
    technology: str
    color: str
    aftermarket_match: str


# Brand knowledge base — maps spec abbreviations to full details.
# These are manufacturer-wide standards, not per-vehicle data.
_COOLANT_SPECS: dict[str, CoolantSpec] = {
    "G13":   CoolantSpec("TL 774 J (G13)", "Si-OAT", "purple", "GLYSANTIN G40/G30"),
    "LC-18": CoolantSpec("BMW LC-18", "HOAT", "blue/green", "GLYSANTIN G48"),
    "325.6": CoolantSpec("MB 325.6", "OAT", "blue", "GLYSANTIN G30"),
    "SLLC":  CoolantSpec("Super Long Life Coolant", "OAT", "pink", "Any OAT pink coolant"),
    "MS591": CoolantSpec("MS 591-08", "P-OAT", "green", "P-OAT green coolant"),
    "FL22":  CoolantSpec("FL22", "P-OAT", "green", "FL22-compatible only"),
    "TYPE2": CoolantSpec("Type 2 / e-Coolant", "OAT", "blue", "Honda-spec OAT"),
    "L250":  CoolantSpec("L250 / L248", "Si-OAT", "blue/green", "Si-OAT coolant"),
    "TYPED": CoolantSpec("Type D (Glaceol RX)", "OAT", "yellow", "GLYSANTIN G33"),
}

# Maps car makes to their default coolant spec key.
_BRAND_DEFAULTS: dict[str, str] = {
    "Volkswagen": "G13", "Skoda": "G13", "SEAT": "G13", "Audi": "G13",
    "BMW": "LC-18", "MINI": "LC-18",
    "Mercedes-Benz": "325.6",
    "Toyota": "SLLC", "Lexus": "SLLC",
    "Hyundai": "MS591", "Kia": "MS591",
    "Mazda": "FL22",
    "Honda": "TYPE2",
    "Nissan": "L250", "Infiniti": "L250",
    "Renault": "TYPED", "Peugeot": "TYPED", "Citroen": "TYPED",
}

# Compatibility matrix: technology -> set of technologies safe to mix with.
# Mixing incompatible types causes gel formation, corrosion, or seal damage.
COMPATIBILITY_MATRIX: dict[str, set[str]] = {
    "IAT":    {"IAT"},
    "OAT":    {"OAT"},
    "HOAT":   {"HOAT", "IAT"},       # HOAT contains IAT inhibitors
    "P-OAT":  {"P-OAT"},
    "Si-OAT": {"Si-OAT", "OAT"},     # Si-OAT is OAT + silicate
}

# All known coolant technology types.
_ALL_TECHNOLOGIES = frozenset(COMPATIBILITY_MATRIX.keys())


def _build_mixing_warning(technology: str) -> str:
    """Generate a human-readable mixing warning for a coolant technology.

    Lists all incompatible technologies.  Returns a generic warning
    for unknown technology types.
    """
    compatible = COMPATIBILITY_MATRIX.get(technology)
    if compatible is None:
        return f"Unknown coolant technology '{technology}' — verify compatibility before mixing"

    incompatible = sorted(_ALL_TECHNOLOGIES - compatible)
    if not incompatible:
        return f"{technology} is compatible with all known coolant types"

    if len(incompatible) == 1:
        return f"Do NOT mix with {incompatible[0]} coolant"
    return f"Do NOT mix with {', '.join(incompatible[:-1])} or {incompatible[-1]} coolant"


def _resolve_coolant_type(raw: str) -> str | None:
    """Map DB coolant_type string to a _COOLANT_SPECS key.

    The DB might store "G13", "TL 774 J", "SLLC", etc.  This helper
    normalizes common variations to the canonical key.

    Returns ``None`` if the raw value cannot be mapped.
    """
    if not raw or not raw.strip():
        return None

    normalized = raw.strip().upper()

    # Direct key match (most common case).
    for key in _COOLANT_SPECS:
        if key.upper() == normalized:
            return key

    # Partial match: check if the raw value appears in the spec name.
    for key, spec in _COOLANT_SPECS.items():
        if normalized in spec.spec_name.upper():
            return key

    return None


class CoolantLookup:
    """Three-tier coolant specification resolver.

    Wraps :class:`PartsDatabase` queries with a cascade that
    progressively relaxes matching criteria until a result is found
    or all tiers are exhausted.  Enriches DB records with a hardcoded
    brand knowledge base and mixing-compatibility warnings.
    """

    def __init__(self, db: PartsDatabase) -> None:
        self._db = db

    def lookup(self, vehicle: VehicleRecord) -> Optional[CoolantResult]:
        """Resolve coolant specs for a vehicle, cascading through tiers.

        Returns ``None`` only when all three tiers fail to match.
        """
        make, model = self._resolve_names(vehicle)
        if make is None:
            logger.warning(
                "Cannot resolve make name for plate %s — skipping coolant lookup",
                vehicle.plate,
            )
            return None

        engine_code = vehicle.engine_code

        # Tier 1: exact match (requires model + engine_code)
        if model and engine_code:
            result = self._try_exact(make, model, vehicle.year, engine_code)
            if result is not None:
                return result

        # Tier 2: model-year fallback (requires model, ignores engine)
        if model:
            result = self._try_model_year(make, model, vehicle.year)
            if result is not None:
                return result

        # Tier 3: brand default
        return self._try_brand_default(make)

    def _resolve_names(
        self, vehicle: VehicleRecord,
    ) -> tuple[str | None, str | None]:
        """Resolve make/model names, preferring English over Hebrew."""
        return resolve_vehicle_names(vehicle, logger)

    def _try_exact(
        self,
        make: str,
        model: str,
        year: int,
        engine_code: str,
    ) -> Optional[CoolantResult]:
        """Tier 1: exact match on all identity fields."""
        specs = self._db.find_specs(make, model, year, engine_code)
        return self._specs_to_result(specs, source="exact")

    def _try_model_year(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[CoolantResult]:
        """Tier 2: match on make + model + year, preferring coolant data."""
        specs = self._db.find_specs_by_model_year_for_coolant(make, model, year)
        return self._specs_to_result(specs, source="model_year")

    def _try_brand_default(self, make: str) -> Optional[CoolantResult]:
        """Tier 3: hardcoded brand-level default."""
        spec_key = _BRAND_DEFAULTS.get(make)
        if spec_key is None:
            return None
        spec = _COOLANT_SPECS[spec_key]
        warning = _build_mixing_warning(spec.technology)
        return CoolantResult.from_brand_default(
            make=make,
            spec_info=(spec.spec_name, spec.technology, spec.color, spec.aftermarket_match),
            mixing_warning=warning,
        )

    def _specs_to_result(
        self,
        specs: Optional[VehicleSpecs],
        source: Literal["exact", "model_year"],
    ) -> Optional[CoolantResult]:
        """Convert a VehicleSpecs to a CoolantResult via the knowledge base.

        Returns ``None`` if specs is None or coolant_type cannot be resolved.
        """
        if specs is None:
            return None

        spec_key = _resolve_coolant_type(specs.coolant_type)
        if spec_key is None:
            return None

        spec = _COOLANT_SPECS[spec_key]
        warning = _build_mixing_warning(spec.technology)
        return CoolantResult.from_vehicle_specs(
            specs=specs,
            spec_info=(spec.spec_name, spec.technology, spec.color, spec.aftermarket_match),
            mixing_warning=warning,
            source=source,
        )
