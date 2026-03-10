"""Oil specification lookup with four-tier cascade matching.

Given a VehicleRecord (from the government API), resolves oil
specifications from the database using a cascade:

1. **Exact match** — make + model + year + engine_code
2. **Model-year match** — make + model + year (ignoring engine_code,
   preferring fuel_type match when multiple variants exist)
3. **Engine family fallback** — make + engine prefix + year
4. **Brand default** — most common spec for make

Usage::

    with PartsDatabase("parts.db") as db:
        oil = OilLookup(db)
        result = oil.lookup(vehicle)
        if result:
            print(f"{result.viscosity} ({result.confidence})")
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from parts_finder.db import PartsDatabase
from parts_finder.lookup._shared import resolve_vehicle_names
from parts_finder.models import OilResult, VehicleRecord

logger = logging.getLogger(__name__)

# Engine family prefixes must be alphanumeric only.  This prevents
# SQL LIKE wildcards (%, _, [) from leaking into prefix queries.
_SAFE_PREFIX_RE = re.compile(r"^[A-Za-z0-9]+$")


def extract_engine_family(engine_code: str) -> str | None:
    """Extract the family prefix from a dash-separated engine code.

    Returns the portion before the first dash, or ``None`` if the code
    has no dash (dashless codes like "G4FJ" cannot be reliably split
    without manufacturer-specific knowledge).  Also returns ``None`` if
    the prefix contains non-alphanumeric characters, which would be
    unsafe in a SQL LIKE query.

    Examples::

        >>> extract_engine_family("2ZR-FE")
        '2ZR'
        >>> extract_engine_family("1KD-FTV")
        '1KD'
        >>> extract_engine_family("G4FJ") is None
        True
    """
    if not engine_code or not engine_code.strip():
        return None
    code = engine_code.strip()
    if "-" not in code:
        return None
    prefix = code.split("-", 1)[0]
    if not prefix or not _SAFE_PREFIX_RE.match(prefix):
        return None
    return prefix


class OilLookup:
    """Three-tier oil specification resolver.

    Wraps :class:`PartsDatabase` queries with a cascade that progressively
    relaxes matching criteria until a result is found or all tiers are
    exhausted.
    """

    def __init__(self, db: PartsDatabase) -> None:
        self._db = db

    def lookup(self, vehicle: VehicleRecord) -> Optional[OilResult]:
        """Resolve oil specs for a vehicle, cascading through tiers.

        Returns ``None`` only when all three tiers fail to match.
        """
        make, model = self._resolve_names(vehicle)
        if make is None:
            logger.warning(
                "Cannot resolve make name for plate %s — skipping oil lookup",
                vehicle.plate,
            )
            return None

        engine_code = vehicle.engine_code

        # Tier 1: exact match (requires model; Tiers 2-4 relax progressively)
        if model and engine_code:
            result = self._try_exact(make, model, vehicle.year, engine_code)
            if result is not None:
                return result

        # Tier 2: model-year match (ignores engine_code, uses fuel_type)
        if model:
            result = self._try_model_year(
                make, model, vehicle.year, vehicle.fuel_type,
            )
            if result is not None:
                return result

        # Tier 3: engine family fallback
        if engine_code:
            result = self._try_engine_family(make, engine_code, vehicle.year)
            if result is not None:
                return result

        # Tier 4: brand default
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
    ) -> Optional[OilResult]:
        """Tier 1: exact match on all identity fields."""
        specs = self._db.find_specs(make, model, year, engine_code)
        if specs is None or not specs.oil_viscosity:
            return None
        return OilResult.from_vehicle_specs(specs, confidence="exact")

    def _try_model_year(
        self,
        make: str,
        model: str,
        year: int,
        fuel_type: str,
    ) -> Optional[OilResult]:
        """Tier 2: match on make + model + year, ignoring engine_code.

        This tier bridges the gap between the government API's real
        engine codes (e.g. "G4FD") and the DB's descriptive engine
        labels (e.g. "1.6T Petrol") seeded from JSX data.  When
        multiple engine variants exist for the same model-year, the
        fuel_type is used to pick the most relevant one.
        """
        specs = self._db.find_specs_by_model_year_for_oil(
            make, model, year, fuel_type=fuel_type,
        )
        if specs is None or not specs.oil_viscosity:
            return None
        return OilResult.from_vehicle_specs(specs, confidence="model_year")

    def _try_engine_family(
        self,
        make: str,
        engine_code: str,
        year: int,
    ) -> Optional[OilResult]:
        """Tier 2: match on engine family prefix."""
        prefix = extract_engine_family(engine_code)
        if prefix is None:
            return None
        specs = self._db.find_specs_by_engine_family(make, prefix, year)
        if specs is None or not specs.oil_viscosity:
            return None
        return OilResult.from_vehicle_specs(specs, confidence="engine_family")

    def _try_brand_default(self, make: str) -> Optional[OilResult]:
        """Tier 3: aggregate brand-level default."""
        result = self._db.find_brand_default_oil(make)
        if result is None:
            return None
        viscosity, spec, interval = result
        return OilResult.from_brand_default(
            make=make,
            viscosity=viscosity,
            spec=spec,
            change_interval_km=interval,
        )
