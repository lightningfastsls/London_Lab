"""Translate domain models to API response schemas.

This module is the bridge between the frozen dataclasses (domain layer)
and the Pydantic models (API layer).  It centralises all translation
logic so that routes.py stays thin and testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from parts_finder.api.schemas import (
    AftermarketPartResponse,
    BrakesResponse,
    BulbsResponse,
    CategoriesResponse,
    CoolantResponse,
    FilterCategoryResponse,
    OilResponse,
    PlateLookupResponse,
    VehicleResponse,
)
from parts_finder.lookup_engine import LookupResult
from parts_finder.models import ProductCrossRef

if TYPE_CHECKING:
    from parts_finder.api.fallback import AIFallbackResult

# The 7 API categories in display order.
_ALL_CATEGORIES = (
    "oil", "oil_filter", "air_filter", "cabin_filter",
    "brakes", "bulbs", "coolant",
)


def _crossrefs_to_response(
    crossrefs: tuple[ProductCrossRef, ...],
) -> list[AftermarketPartResponse]:
    """Convert a tuple of ProductCrossRef to API response models."""
    return [
        AftermarketPartResponse(
            brand=xref.brand,
            part_number=xref.brand_part_number,
            notes=xref.notes,
        )
        for xref in crossrefs
    ]


def _build_vehicle(result: LookupResult) -> VehicleResponse:
    """Translate VehicleRecord to API vehicle response."""
    v = result.vehicle
    return VehicleResponse(
        plate=v.plate,
        make=v.make_english or v.make_hebrew,
        model=v.model_english or v.model_hebrew,
        year=v.year,
        engine_code=v.engine_code,
        fuel_type=v.fuel_type,
    )


def _build_oil(result: LookupResult) -> OilResponse | None:
    """Translate OilResult to API oil response."""
    if result.oil is None:
        return None
    o = result.oil
    return OilResponse(
        viscosity=o.viscosity,
        capacity_l=o.capacity_l,
        spec=o.spec,
        oem_approval=o.oem_approval,
        change_interval_km=o.change_interval_km,
        confidence=o.confidence,
        source=o.source,
    )


def _build_filter_category(
    oem: str,
    crossrefs: tuple[ProductCrossRef, ...],
) -> FilterCategoryResponse | None:
    """Build a single filter category response, or None if no OEM number."""
    if not oem:
        return None
    return FilterCategoryResponse(
        oem_part_number=oem,
        aftermarket=_crossrefs_to_response(crossrefs),
    )


def _build_brakes(result: LookupResult) -> BrakesResponse | None:
    """Translate BrakeResult to API brakes response."""
    if result.brakes is None:
        return None
    b = result.brakes
    return BrakesResponse(
        front_pad_oem=b.front_pad_oem,
        rear_pad_oem=b.rear_pad_oem,
        front_disc_oem=b.front_disc_oem,
        rear_disc_oem=b.rear_disc_oem,
        brake_fluid_type=b.brake_fluid_type,
        front_pad_aftermarket=_crossrefs_to_response(b.front_pad_crossrefs),
        rear_pad_aftermarket=_crossrefs_to_response(b.rear_pad_crossrefs),
        front_disc_aftermarket=_crossrefs_to_response(b.front_disc_crossrefs),
        rear_disc_aftermarket=_crossrefs_to_response(b.rear_disc_crossrefs),
    )


def _build_bulbs(result: LookupResult) -> BulbsResponse | None:
    """Translate BulbResult to API bulbs response."""
    if result.bulbs is None:
        return None
    b = result.bulbs
    return BulbsResponse(
        low_beam=b.low_beam,
        high_beam=b.high_beam,
        front_turn=b.front_turn,
        rear_turn=b.rear_turn,
        tail_brake=b.tail_brake,
        reverse=b.reverse,
        fog=b.fog,
        license_plate=b.license_plate,
    )


def _build_coolant(result: LookupResult) -> CoolantResponse | None:
    """Translate CoolantResult to API coolant response."""
    if result.coolant is None:
        return None
    c = result.coolant
    return CoolantResponse(
        spec=c.spec,
        technology=c.technology,
        color=c.color,
        capacity_l=c.capacity_l,
        aftermarket_match=c.aftermarket_match,
        mixing_warning=c.mixing_warning,
    )


def build_response(
    result: LookupResult,
    ai_result: AIFallbackResult | None = None,
) -> PlateLookupResponse:
    """Convert a domain LookupResult into the full API response.

    Computes coverage by counting how many of the 7 categories
    have non-None values.  When *ai_result* is provided, AI-generated
    specs fill in categories that the database missed.
    """
    # Build filter categories from the unified FilterResult
    oil_filter = None
    air_filter = None
    cabin_filter = None
    if result.filters is not None:
        f = result.filters
        oil_filter = _build_filter_category(
            f.oil_filter_oem, f.oil_filter_crossrefs,
        )
        air_filter = _build_filter_category(
            f.air_filter_oem, f.air_filter_crossrefs,
        )
        cabin_filter = _build_filter_category(
            f.cabin_filter_oem, f.cabin_filter_crossrefs,
        )

    categories = CategoriesResponse(
        oil=_build_oil(result),
        oil_filter=oil_filter,
        air_filter=air_filter,
        cabin_filter=cabin_filter,
        brakes=_build_brakes(result),
        bulbs=_build_bulbs(result),
        coolant=_build_coolant(result),
    )

    # Merge AI results into categories where DB had no data
    used_ai = False
    db_provided: set[str] = set()
    if ai_result is not None:
        cat_dict = categories.model_dump()
        # Track which categories the DB already provided (before AI merge)
        db_provided = {name for name in _ALL_CATEGORIES if cat_dict[name] is not None}
        merge: dict[str, object] = {}
        for cat_name in _ALL_CATEGORIES:
            ai_val = getattr(ai_result, cat_name, None)
            if cat_dict[cat_name] is None and ai_val is not None:
                merge[cat_name] = ai_val
                used_ai = True
        if merge:
            categories = categories.model_copy(update=merge)

    # Compute coverage (after AI merge)
    cat_dict = categories.model_dump()
    matched = [name for name in _ALL_CATEGORIES if cat_dict[name] is not None]
    unmatched = [name for name in _ALL_CATEGORIES if cat_dict[name] is None]
    coverage = f"{len(matched)}/{len(_ALL_CATEGORIES)} categories matched"

    # Determine data source label
    if not used_ai:
        data_source = "database"
    elif db_provided:
        data_source = "hybrid"
    else:
        data_source = "ai_fallback"

    return PlateLookupResponse(
        vehicle=_build_vehicle(result),
        categories=categories,
        coverage=coverage,
        unmatched_categories=unmatched,
        data_source=data_source,
    )
