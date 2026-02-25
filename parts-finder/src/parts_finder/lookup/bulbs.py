"""Bulb type lookup with two-tier cascade matching.

Given a VehicleRecord (from the government API), resolves bulb types
for all lamp positions from the database using a cascade:

1. **Exact match** — make + model + year + engine_code
2. **Model-year fallback** — make + model + year (any engine)

Unlike oil (which has a brand-default tier), bulbs vary too much by
model to aggregate at brand level.  Different engine variants on the
same chassis typically share the same light housings, which is why
the model-year fallback relaxes only the engine_code constraint.

Usage::

    with PartsDatabase("parts.db") as db:
        bulbs = BulbLookup(db)
        result = bulbs.lookup(vehicle)
        if result:
            for pos, bulb_type in result.replaceable_positions.items():
                print(f"{pos}: {bulb_type}")
"""

from __future__ import annotations

import logging
from typing import Optional

from parts_finder.db import PartsDatabase
from parts_finder.lookup._shared import resolve_vehicle_names
from parts_finder.models import BulbResult, VehicleRecord, VehicleSpecs

logger = logging.getLogger(__name__)

# Bulb field names on VehicleSpecs — used to check "has data".
_BULB_SPEC_FIELDS = (
    "low_beam_bulb", "high_beam_bulb", "front_turn_bulb",
    "rear_turn_bulb", "tail_brake_bulb", "reverse_bulb",
    "fog_bulb", "license_plate_bulb",
)


def _has_bulb_data(specs: VehicleSpecs) -> bool:
    """Return True if at least one bulb field is non-empty."""
    return any(getattr(specs, f) for f in _BULB_SPEC_FIELDS)


class BulbLookup:
    """Two-tier bulb type resolver.

    Wraps :class:`PartsDatabase` queries with a cascade that
    progressively relaxes matching criteria until a result is found
    or both tiers are exhausted.
    """

    def __init__(self, db: PartsDatabase) -> None:
        self._db = db

    def lookup(self, vehicle: VehicleRecord) -> Optional[BulbResult]:
        """Resolve bulb types for a vehicle, cascading through tiers.

        Returns ``None`` only when both tiers fail to match.
        """
        make, model = self._resolve_names(vehicle)
        if make is None:
            logger.warning(
                "Cannot resolve make name for plate %s — skipping bulb lookup",
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

        return None

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
    ) -> Optional[BulbResult]:
        """Tier 1: exact match on all identity fields."""
        specs = self._db.find_specs(make, model, year, engine_code)
        if specs is None or not _has_bulb_data(specs):
            return None
        return BulbResult.from_vehicle_specs(specs, source="exact")

    def _try_model_year(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[BulbResult]:
        """Tier 2: match on make + model + year, ignoring engine_code."""
        specs = self._db.find_specs_by_model_year(make, model, year)
        if specs is None or not _has_bulb_data(specs):
            return None
        return BulbResult.from_vehicle_specs(specs, source="model_year")
