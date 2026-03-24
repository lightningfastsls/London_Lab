"""Pydantic request/response models for the Parts Finder API.

These models form the API contract — they translate between the
internal frozen dataclasses (domain models) and the JSON wire format.
Keeping them separate from the domain models means:

1. Domain dataclasses stay frozen and dependency-free.
2. API models can evolve independently (add fields, rename, deprecate).
3. Pydantic handles validation and OpenAPI schema generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────


class PlateLookupRequest(BaseModel):
    """POST /api/plate-lookup request body."""

    plate: str = Field(..., min_length=1, description="Israeli license plate number")


# ── Response: leaf models ────────────────────────────────────────────


class AftermarketPartResponse(BaseModel):
    """A single aftermarket cross-reference for an OEM part."""

    brand: str
    part_number: str
    notes: str = ""


class VehicleResponse(BaseModel):
    """Vehicle identification from the government registry."""

    plate: str
    make: str
    model: str
    year: int
    engine_code: str
    fuel_type: str


class OilResponse(BaseModel):
    """Oil specification recommendation."""

    viscosity: str
    capacity_l: float
    spec: str
    oem_approval: str
    change_interval_km: int
    confidence: str
    source: str


class FilterCategoryResponse(BaseModel):
    """A single filter category (oil filter, air filter, etc.)."""

    oem_part_number: str
    aftermarket: list[AftermarketPartResponse] = []


class BrakesResponse(BaseModel):
    """Brake parts recommendation with OEM + aftermarket options."""

    front_pad_oem: str = ""
    rear_pad_oem: str = ""
    front_disc_oem: str = ""
    rear_disc_oem: str = ""
    brake_fluid_type: str = ""
    front_pad_aftermarket: list[AftermarketPartResponse] = []
    rear_pad_aftermarket: list[AftermarketPartResponse] = []
    front_disc_aftermarket: list[AftermarketPartResponse] = []
    rear_disc_aftermarket: list[AftermarketPartResponse] = []


class BulbsResponse(BaseModel):
    """Bulb types per lamp position."""

    low_beam: str = ""
    high_beam: str = ""
    front_turn: str = ""
    rear_turn: str = ""
    tail_brake: str = ""
    reverse: str = ""
    fog: str = ""
    license_plate: str = ""


class CoolantResponse(BaseModel):
    """Coolant specification recommendation."""

    spec: str
    technology: str
    color: str
    capacity_l: float
    aftermarket_match: str
    mixing_warning: str


# ── Response: composite models ───────────────────────────────────────


class CategoriesResponse(BaseModel):
    """All 7 product categories — None where no match was found."""

    oil: OilResponse | None = None
    oil_filter: FilterCategoryResponse | None = None
    air_filter: FilterCategoryResponse | None = None
    cabin_filter: FilterCategoryResponse | None = None
    brakes: BrakesResponse | None = None
    bulbs: BulbsResponse | None = None
    coolant: CoolantResponse | None = None


class PlateLookupResponse(BaseModel):
    """Full API response for POST /api/plate-lookup."""

    vehicle: VehicleResponse
    categories: CategoriesResponse
    coverage: str = Field(
        ..., description="Human-readable coverage summary, e.g. '5/7 categories matched'"
    )
    unmatched_categories: list[str] = Field(
        default_factory=list,
        description="Category names with no match",
    )
    data_source: str = Field(
        default="database",
        description=(
            "Source of the response data. "
            "'database' = all from local DB; "
            "'ai_fallback' = all AI-generated; "
            "'hybrid' = mix of DB and AI."
        ),
    )


# ── Error responses ──────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
