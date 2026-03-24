"""Orchestration layer for the full Parts Finder lookup pipeline.

Coordinates the complete flow: license plate → government API resolution →
Hebrew-to-English name enrichment → category-specific database lookups.

Usage::

    config = AppConfig(db_path="parts_finder.db")
    with PartsDatabase(config.db_path) as db:
        mapper = NameMapper(Path("data/hebrew_names.json"))
        engine = LookupEngine(config, db, mapper)
        result = await engine.lookup("12-345-67")
        print(result.oil)  # OilResult or None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from parts_finder.config import AppConfig
from parts_finder.db import PartsDatabase
from parts_finder.lookup.brakes import BrakeLookup
from parts_finder.lookup.bulbs import BulbLookup
from parts_finder.lookup.coolant import CoolantLookup
from parts_finder.lookup.filters import FilterLookup
from parts_finder.models import (
    BrakeResult,
    BulbResult,
    CoolantResult,
    FilterResult,
    OilResult,
    VehicleRecord,
)
from parts_finder.name_mapper import NameMapper
from parts_finder.oil_lookup import OilLookup
from parts_finder.plate_client import PlateClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LookupResult:
    """Complete lookup result across all implemented categories.

    Each category field is ``None`` when the lookup found no match or
    when the category module raised an unexpected error (logged as a
    warning rather than crashing the entire pipeline).
    """

    vehicle: VehicleRecord
    oil: Optional[OilResult] = None
    coolant: Optional[CoolantResult] = None
    brakes: Optional[BrakeResult] = None
    bulbs: Optional[BulbResult] = None
    filters: Optional[FilterResult] = None


class LookupEngine:
    """Single entry point for the license-plate-to-parts pipeline.

    Wraps :class:`PlateClient` (async), :class:`NameMapper`, and all
    five category lookup modules into one ``await engine.lookup(plate)``
    call.

    Parameters
    ----------
    config:
        Application configuration (API URL, timeouts, DB path).
    db:
        Open database connection.  The caller is responsible for
        lifecycle (open/close).
    mapper:
        Loaded Hebrew→English name mapper.
    """

    def __init__(
        self,
        config: AppConfig,
        db: PartsDatabase,
        mapper: NameMapper,
    ) -> None:
        self._config = config
        self._db = db
        self._mapper = mapper

        # Instantiate category lookups once
        self._oil = OilLookup(db)
        self._coolant = CoolantLookup(db)
        self._bulbs = BulbLookup(db)
        self._brakes = BrakeLookup(db)
        self._filters = FilterLookup(db)

    async def lookup(self, plate: str) -> LookupResult:
        """Run the full pipeline for a license plate.

        Parameters
        ----------
        plate:
            Raw plate string (hyphens/spaces stripped automatically).

        Returns
        -------
        LookupResult
            Vehicle info plus per-category results (``None`` where no
            match was found or the lookup failed gracefully).

        Raises
        ------
        GovApiError
            If the government API is unreachable or returns an error.
        PlateNotFoundError
            If the plate is not in the registry.
        ValueError
            If the plate string contains no digits.
        """
        # Step 1: Resolve plate via government API
        async with PlateClient(self._config) as client:
            vehicle = await client.lookup(plate)

        # Step 2: Enrich with English names
        vehicle = self._mapper.enrich_vehicle(vehicle)

        # Step 3: Run category lookups (each isolated)
        oil = self._safe_lookup("oil", self._oil.lookup, vehicle)
        coolant = self._safe_lookup("coolant", self._coolant.lookup, vehicle)
        bulbs = self._safe_lookup("bulbs", self._bulbs.lookup, vehicle)
        brakes = self._safe_lookup("brakes", self._brakes.lookup, vehicle)
        filters = self._safe_lookup("filters", self._filters.lookup, vehicle)

        return LookupResult(
            vehicle=vehicle,
            oil=oil,
            coolant=coolant,
            brakes=brakes,
            bulbs=bulbs,
            filters=filters,
        )

    @staticmethod
    def _safe_lookup(
        category: str,
        fn: object,
        vehicle: VehicleRecord,
    ) -> object:
        """Call a lookup function, catching and logging any exception.

        Returns the lookup result on success, or ``None`` on failure.
        This ensures that one broken category doesn't take down the
        entire pipeline.
        """
        try:
            return fn(vehicle)  # type: ignore[operator]
        except Exception:
            logger.warning(
                "Lookup failed for category '%s' (plate %s)",
                category,
                vehicle.plate,
                exc_info=True,
            )
            return None
