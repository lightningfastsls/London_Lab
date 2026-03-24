"""Filter parts lookup with two-tier cascade matching.

Given a VehicleRecord (from the government API), resolves filter part
OEM numbers and aftermarket cross-references using a cascade:

1. **Exact match** — make + model + year + engine_code
2. **Model-year fallback** — make + model + year (any engine)

Unlike oil or coolant (which have brand-default tiers), filter OEM
numbers are too vehicle-specific to aggregate at brand level.
Different engine variants on the same chassis sometimes share the same
filter housing, which is why the model-year fallback relaxes only the
engine_code constraint.

Cross-references are resolved at result construction time by calling
``db.find_crossrefs(oem)`` for each non-empty OEM number, mapping
to aftermarket brands (MANN-FILTER, Bosch, Mahle, etc.).

Usage::

    with PartsDatabase("parts.db") as db:
        filters = FilterLookup(db)
        result = filters.lookup(vehicle)
        if result:
            print(f"Oil filter: {result.oil_filter_oem}")
            for xref in result.oil_filter_crossrefs:
                print(f"  {xref.brand}: {xref.brand_part_number}")
"""

from __future__ import annotations

import logging
from typing import Optional

from parts_finder.db import PartsDatabase
from parts_finder.lookup._shared import resolve_vehicle_names
from parts_finder.models import (
    FilterResult,
    VehicleRecord,
    VehicleSpecs,
    _FILTER_OEM_FIELDS,
)

logger = logging.getLogger(__name__)


def _has_filter_data(specs: VehicleSpecs) -> bool:
    """Return True if at least one filter OEM field is non-empty."""
    return any(getattr(specs, f) for f in _FILTER_OEM_FIELDS)


class FilterLookup:
    """Two-tier filter parts resolver.

    Wraps :class:`PartsDatabase` queries with a cascade that
    progressively relaxes matching criteria until a result is found
    or both tiers are exhausted.  Cross-references to aftermarket
    brands are resolved during result construction.
    """

    def __init__(self, db: PartsDatabase) -> None:
        self._db = db

    def lookup(self, vehicle: VehicleRecord) -> Optional[FilterResult]:
        """Resolve filter parts for a vehicle, cascading through tiers.

        Returns ``None`` only when both tiers fail to match.
        """
        make, model = self._resolve_names(vehicle)
        if make is None:
            logger.warning(
                "Cannot resolve make name for plate %s — skipping filter lookup",
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
    ) -> Optional[FilterResult]:
        """Tier 1: exact match on all identity fields."""
        specs = self._db.find_specs(make, model, year, engine_code)
        if specs is None or not _has_filter_data(specs):
            return None
        return FilterResult.from_vehicle_specs(specs, source="exact", db=self._db)

    def _try_model_year(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[FilterResult]:
        """Tier 2: match on make + model + year, preferring filter data."""
        specs = self._db.find_specs_by_model_year_for_filters(make, model, year)
        if specs is None or not _has_filter_data(specs):
            return None
        return FilterResult.from_vehicle_specs(specs, source="model_year", db=self._db)
