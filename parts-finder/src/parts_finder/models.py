"""Domain models for the Parts Finder service.

VehicleRecord is the canonical representation of a vehicle looked up
from the Israeli government registry (data.gov.il).  The ``engine_code``
field (sourced from the API's ``degem_manoa``) is the "golden key" that
ties a vehicle to its parts catalog.

VehicleSpecs holds the maintenance specifications for a vehicle across
model-year generations (oil, filters, brakes, bulbs, coolant, belts).

ProductCrossRef maps OEM part numbers to aftermarket equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


def _clean(value: str | None) -> str | None:
    """Strip whitespace; collapse empty strings to None."""
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _normalize_model_name(value: str) -> str:
    """Title-case an ALL-CAPS English model name for DB consistency.

    The government API's ``kinuy_mishari`` field often returns model
    names in ALL CAPS (e.g. "SPORTAGE", "COROLLA") while the DB stores
    them in Title Case ("Sportage", "Corolla").  Pure-Hebrew names are
    left unchanged (Hebrew has no case distinction).
    """
    # Only transform if all-caps ASCII
    if value.isascii() and value.isupper() and len(value) > 1:
        return value.title()
    return value


def _looks_like_model_name(value: str | None) -> bool:
    """Return True if the value looks like a human-readable model name.

    The API's ``degem_nm`` field sometimes contains internal model codes
    (e.g. "PB814B", "ZRE181L") rather than readable names (e.g. "Corolla",
    "SPORTAGE").  Internal codes are typically short alphanumeric strings
    with digits embedded mid-word — not what a user would recognise.

    Heuristic: reject values that are a single word mixing letters and
    digits (like "PB814B").  Accept multi-word values or pure-letter words.
    """
    if not value:
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    # Multi-word values are likely real names
    if " " in cleaned:
        return True
    # Single word: reject if it mixes letters and digits (internal code)
    has_alpha = any(c.isalpha() for c in cleaned)
    has_digit = any(c.isdigit() for c in cleaned)
    return not (has_alpha and has_digit)


def _require_str(record: dict, key: str) -> str:
    """Extract a required string field from an API record.

    Raises ``ValueError`` if the value is ``None`` (the API returned
    ``null``) rather than silently coercing to the string ``"None"``.
    """
    val = record[key]
    if val is None:
        raise ValueError(f"Required field '{key}' is null in API record")
    return str(val).strip()


@dataclass(frozen=True)
class VehicleRecord:
    """A single vehicle resolved from the government registry.

    Required fields come directly from the API; optional fields may be
    absent for older registrations.
    """

    # --- Required ---
    plate: str
    make_hebrew: str
    model_hebrew: str
    year: int
    engine_code: str
    fuel_type: str

    # --- Optional ---
    vin: Optional[str] = None
    trim: Optional[str] = None
    make_english: Optional[str] = None
    model_english: Optional[str] = None
    degem_nm: Optional[str] = None

    @property
    def lookup_key(self) -> str:
        """Canonical key for downstream catalog queries.

        Format: ``make|model|year|engine_code``
        """
        return f"{self.make_hebrew}|{self.model_hebrew}|{self.year}|{self.engine_code}"

    @classmethod
    def from_api_record(cls, plate: str, record: dict) -> VehicleRecord:
        """Build a VehicleRecord from a raw data.gov.il API record.

        Field mapping (Hebrew API names -> dataclass fields):
            tozeret_nm       -> make_hebrew
            kinuy_mishari    -> model_hebrew
            shnat_yitzur     -> year
            degem_manoa      -> engine_code
            sug_delek_nm     -> fuel_type
            misgeret         -> vin
            ramat_gimur      -> trim
            tozeret_cd       -> (not used)
            tozeret_eretz_nm -> make_english  (if present)
            degem_nm         -> model_english (if present)
        """
        return cls(
            plate=plate.strip(),
            make_hebrew=_require_str(record, "tozeret_nm"),
            model_hebrew=_normalize_model_name(_require_str(record, "kinuy_mishari")),
            year=int(record["shnat_yitzur"]),
            engine_code=str(record.get("degem_manoa", "")).strip(),
            fuel_type=_require_str(record, "sug_delek_nm"),
            vin=_clean(record.get("misgeret")),
            trim=_clean(record.get("ramat_gimur")),
            make_english=_clean(record.get("tozeret_eretz_nm")),
            model_english=(
                _clean(record.get("degem_nm"))
                if _looks_like_model_name(record.get("degem_nm"))
                else None
            ),
            degem_nm=_clean(record.get("degem_nm")),
        )


@dataclass(frozen=True)
class VehicleSpecs:
    """Maintenance specifications for a vehicle model-year generation.

    The identity fields (make through fuel_type) are required; all
    category fields default to empty so records can be populated
    incrementally as data is compiled.

    The year range (year_from/year_to) spans a model generation — a
    single specs record covers e.g. 2019-2023 Corolla 1ZR-FE.
    """

    # --- Identity (required) ---
    make: str
    model: str
    year_from: int
    year_to: int
    engine_code: str
    fuel_type: str

    # --- Oil ---
    oil_viscosity: str = ""
    oil_capacity_l: float = 0.0
    oil_spec: str = ""
    oil_oem_approval: str = ""
    oil_filter_oem: str = ""
    oil_drain_plug_torque_nm: int = 0
    oil_change_interval_km: int = 0

    # --- Filters ---
    air_filter_oem: str = ""
    cabin_filter_oem: str = ""
    fuel_filter_oem: str = ""

    # --- Brakes ---
    front_brake_pad_oem: str = ""
    rear_brake_pad_oem: str = ""
    front_rotor_oem: str = ""
    rear_rotor_oem: str = ""
    brake_fluid_type: str = ""

    # --- Bulbs ---
    low_beam_bulb: str = ""
    high_beam_bulb: str = ""
    front_turn_bulb: str = ""
    rear_turn_bulb: str = ""
    tail_brake_bulb: str = ""
    reverse_bulb: str = ""
    fog_bulb: str = ""
    license_plate_bulb: str = ""

    # --- Coolant ---
    coolant_type: str = ""
    coolant_capacity_l: float = 0.0
    thermostat_temp_c: int = 0

    # --- Belts & Spark ---
    timing_belt_oem: str = ""
    serpentine_belt_oem: str = ""
    spark_plug_oem: str = ""
    spark_plug_gap_mm: float = 0.0


@dataclass(frozen=True)
class OilResult:
    """Oil specification lookup result with confidence metadata.

    Produced by OilLookup's four-tier cascade.  The ``confidence`` field
    indicates which tier matched:

    - ``"exact"``          — make + model + year + engine_code
    - ``"model_year"``     — make + model + year (engine_code ignored)
    - ``"engine_family"``  — make + engine family prefix + year
    - ``"brand_default"``  — most common spec for make (capacity=0.0)

    The ``source`` field is a human-readable string describing the matched
    record (e.g. ``"Toyota Corolla 2019-2023 2ZR-FE"``).
    """

    viscosity: str
    capacity_l: float
    spec: str
    oem_approval: str
    change_interval_km: int
    confidence: Literal["exact", "model_year", "engine_family", "brand_default"]
    source: str

    @classmethod
    def from_vehicle_specs(
        cls, specs: VehicleSpecs, confidence: Literal["exact", "model_year", "engine_family", "brand_default"],
    ) -> OilResult:
        """Extract oil fields from a VehicleSpecs record."""
        source = (
            f"{specs.make} {specs.model} "
            f"{specs.year_from}-{specs.year_to} {specs.engine_code}"
        )
        return cls(
            viscosity=specs.oil_viscosity,
            capacity_l=specs.oil_capacity_l,
            spec=specs.oil_spec,
            oem_approval=specs.oil_oem_approval,
            change_interval_km=specs.oil_change_interval_km,
            confidence=confidence,
            source=source,
        )

    @classmethod
    def from_brand_default(
        cls,
        make: str,
        viscosity: str,
        spec: str,
        change_interval_km: int,
    ) -> OilResult:
        """Build from aggregated brand-level defaults.

        Capacity is set to 0.0 because it varies by engine and
        cannot be meaningfully aggregated.
        """
        return cls(
            viscosity=viscosity,
            capacity_l=0.0,
            spec=spec,
            oem_approval="",
            change_interval_km=change_interval_km,
            confidence="brand_default",
            source=f"{make} (brand default)",
        )


# Mapping from BulbResult field names to VehicleSpecs field names.
_BULB_POSITION_FIELDS = {
    "low_beam": "low_beam_bulb",
    "high_beam": "high_beam_bulb",
    "front_turn": "front_turn_bulb",
    "rear_turn": "rear_turn_bulb",
    "tail_brake": "tail_brake_bulb",
    "reverse": "reverse_bulb",
    "fog": "fog_bulb",
    "license_plate": "license_plate_bulb",
}


@dataclass(frozen=True)
class BulbResult:
    """Bulb type lookup result for all lamp positions.

    Each field holds an ECE bulb type code (e.g. ``"H7"``, ``"W5W"``),
    the string ``"LED"`` for factory-LED positions that cannot be
    replaced with aftermarket bulbs, or ``""`` for unknown positions.

    .. note:: The ``"LED"`` sentinel must be stored in all-caps.
       Mixed-case variants (``"led"``, ``"Led"``) will **not** be
       filtered by :attr:`replaceable_positions`.

    The ``source`` field indicates which lookup tier matched:

    - ``"exact"``       — make + model + year + engine_code
    - ``"model_year"``  — make + model + year (any engine)
    """

    low_beam: str = ""
    high_beam: str = ""
    front_turn: str = ""
    rear_turn: str = ""
    tail_brake: str = ""
    reverse: str = ""
    fog: str = ""
    license_plate: str = ""
    source: Literal["exact", "model_year"] = "exact"

    @property
    def replaceable_positions(self) -> dict[str, str]:
        """Positions where an aftermarket bulb can be sold.

        Filters out factory-LED positions (value ``"LED"``) and unknown
        positions (value ``""``), returning only positions with a
        concrete ECE bulb type.
        """
        return {
            pos: getattr(self, pos)
            for pos in _BULB_POSITION_FIELDS
            if getattr(self, pos) not in ("", "LED")
        }

    @classmethod
    def from_vehicle_specs(
        cls, specs: VehicleSpecs, source: Literal["exact", "model_year"],
    ) -> BulbResult:
        """Extract bulb fields from a VehicleSpecs record."""
        return cls(
            **{pos: getattr(specs, specs_field)
               for pos, specs_field in _BULB_POSITION_FIELDS.items()},
            source=source,
        )


@dataclass(frozen=True)
class CoolantResult:
    """Coolant specification lookup result with mixing-compatibility warnings.

    Produced by CoolantLookup's three-tier cascade.  The ``source`` field
    indicates which tier matched:

    - ``"exact"``         — make + model + year + engine_code
    - ``"model_year"``    — make + model + year (any engine)
    - ``"brand_default"`` — hardcoded brand-level spec (capacity=0.0)

    The ``mixing_warning`` field provides safety guidance about which
    coolant technologies must not be mixed (based on IAT/OAT/HOAT/P-OAT/Si-OAT
    compatibility).
    """

    spec: str               # e.g., 'TL 774 J (G13)'
    technology: str         # 'OAT', 'HOAT', 'P-OAT', 'Si-OAT', 'IAT'
    color: str              # e.g., 'purple'
    capacity_l: float       # 0.0 when unknown (brand_default tier)
    aftermarket_match: str  # e.g., 'GLYSANTIN G40/G30'
    mixing_warning: str     # e.g., 'Do NOT mix with IAT or P-OAT coolant'
    source: Literal["exact", "model_year", "brand_default"]

    @classmethod
    def from_vehicle_specs(
        cls,
        specs: VehicleSpecs,
        spec_info: tuple[str, str, str, str],
        mixing_warning: str,
        source: Literal["exact", "model_year", "brand_default"],
    ) -> CoolantResult:
        """Build from a VehicleSpecs record enriched with brand knowledge base.

        ``spec_info`` is ``(spec_name, technology, color, aftermarket_match)``
        from the coolant specs knowledge base.
        """
        spec_name, technology, color, aftermarket_match = spec_info
        return cls(
            spec=spec_name,
            technology=technology,
            color=color,
            capacity_l=specs.coolant_capacity_l,
            aftermarket_match=aftermarket_match,
            mixing_warning=mixing_warning,
            source=source,
        )

    @classmethod
    def from_brand_default(
        cls,
        make: str,
        spec_info: tuple[str, str, str, str],
        mixing_warning: str,
    ) -> CoolantResult:
        """Build from brand-level default.

        Capacity is set to 0.0 because it varies by engine and
        cannot be meaningfully aggregated.
        """
        spec_name, technology, color, aftermarket_match = spec_info
        return cls(
            spec=spec_name,
            technology=technology,
            color=color,
            capacity_l=0.0,
            aftermarket_match=aftermarket_match,
            mixing_warning=mixing_warning,
            source="brand_default",
        )


# Mapping from FilterResult field names to VehicleSpecs field names.
_FILTER_SPEC_FIELDS = {
    "oil_filter_oem": "oil_filter_oem",
    "air_filter_oem": "air_filter_oem",
    "cabin_filter_oem": "cabin_filter_oem",
    "fuel_filter_oem": "fuel_filter_oem",
}

# VehicleSpecs field names checked by _has_filter_data() in filters.py.
_FILTER_OEM_FIELDS = (
    "oil_filter_oem", "air_filter_oem", "cabin_filter_oem", "fuel_filter_oem",
)


@dataclass(frozen=True)
class FilterResult:
    """Filter parts lookup result with OEM numbers and aftermarket cross-references.

    Each OEM field maps to a part number from the ``vehicle_specs`` table.
    Cross-reference tuples hold aftermarket equivalents (MANN-FILTER, Bosch,
    etc.) resolved from the ``product_crossref`` table at construction time.

    The ``source`` field indicates which lookup tier matched:

    - ``"exact"``       — make + model + year + engine_code
    - ``"model_year"``  — make + model + year (any engine)
    """

    oil_filter_oem: str = ""
    air_filter_oem: str = ""
    cabin_filter_oem: str = ""
    fuel_filter_oem: str = ""
    oil_filter_crossrefs: tuple[ProductCrossRef, ...] = ()
    air_filter_crossrefs: tuple[ProductCrossRef, ...] = ()
    cabin_filter_crossrefs: tuple[ProductCrossRef, ...] = ()
    fuel_filter_crossrefs: tuple[ProductCrossRef, ...] = ()
    source: Literal["exact", "model_year"] = "exact"

    @property
    def has_data(self) -> bool:
        """True if at least one filter OEM number is known."""
        return bool(
            self.oil_filter_oem or self.air_filter_oem
            or self.cabin_filter_oem or self.fuel_filter_oem
        )

    @classmethod
    def from_vehicle_specs(
        cls,
        specs: VehicleSpecs,
        source: Literal["exact", "model_year"],
        db: object,
    ) -> FilterResult:
        """Extract filter fields from a VehicleSpecs and resolve cross-refs.

        Parameters
        ----------
        specs:
            The matched vehicle specification record.
        source:
            Which lookup tier produced this match.
        db:
            A :class:`~parts_finder.db.PartsDatabase` instance used to
            resolve OEM numbers to aftermarket cross-references.  Typed
            as ``object`` to avoid a circular import.
        """
        oem_values = {
            result_field: getattr(specs, specs_field)
            for result_field, specs_field in _FILTER_SPEC_FIELDS.items()
        }

        # Resolve cross-references for each non-empty OEM number.
        crossref_map: dict[str, tuple[ProductCrossRef, ...]] = {}
        oem_to_crossref = {
            "oil_filter_oem": "oil_filter_crossrefs",
            "air_filter_oem": "air_filter_crossrefs",
            "cabin_filter_oem": "cabin_filter_crossrefs",
            "fuel_filter_oem": "fuel_filter_crossrefs",
        }
        for oem_field, crossref_field in oem_to_crossref.items():
            oem = oem_values[oem_field]
            if oem:
                crossref_map[crossref_field] = tuple(db.find_crossrefs(oem))
            else:
                crossref_map[crossref_field] = ()

        return cls(
            **oem_values,
            **crossref_map,
            source=source,
        )


# Mapping from BrakeResult field names to VehicleSpecs field names.
_BRAKE_SPEC_FIELDS = {
    "front_pad_oem": "front_brake_pad_oem",
    "rear_pad_oem": "rear_brake_pad_oem",
    "front_disc_oem": "front_rotor_oem",
    "rear_disc_oem": "rear_rotor_oem",
    "brake_fluid_type": "brake_fluid_type",
}

# VehicleSpecs field names checked by _has_brake_data() in brakes.py to
# decide whether a specs record contains useful brake data for cascade
# decisions.  Excludes brake_fluid_type because fluid alone is not
# actionable.  See also BrakeResult.has_data which mirrors this check
# using BrakeResult's own field names.
_BRAKE_OEM_FIELDS = (
    "front_brake_pad_oem", "rear_brake_pad_oem",
    "front_rotor_oem", "rear_rotor_oem",
)


@dataclass(frozen=True)
class BrakeResult:
    """Brake parts lookup result with OEM numbers and aftermarket cross-references.

    Each OEM field maps to a part number from the ``vehicle_specs`` table.
    Cross-reference tuples hold aftermarket equivalents (Brembo, TRW, Textar,
    etc.) resolved from the ``product_crossref`` table at construction time.

    The ``source`` field indicates which lookup tier matched:

    - ``"exact"``       — make + model + year + engine_code
    - ``"model_year"``  — make + model + year (any engine)

    .. note:: Disc diameter and brake type (ventilated/solid/drum) are not
       yet available — the DB schema has no columns for these fields.
    """

    front_pad_oem: str = ""
    rear_pad_oem: str = ""
    front_disc_oem: str = ""       # "disc" in user terminology = "rotor" in DB
    rear_disc_oem: str = ""
    brake_fluid_type: str = ""
    front_pad_crossrefs: tuple[ProductCrossRef, ...] = ()
    rear_pad_crossrefs: tuple[ProductCrossRef, ...] = ()
    front_disc_crossrefs: tuple[ProductCrossRef, ...] = ()
    rear_disc_crossrefs: tuple[ProductCrossRef, ...] = ()
    source: Literal["exact", "model_year"] = "exact"

    @property
    def has_data(self) -> bool:
        """True if at least one brake part OEM number is known.

        Brake fluid type alone is not sufficient — without pad or disc
        OEM numbers there's nothing actionable to cross-reference.
        """
        # Mirrors _BRAKE_OEM_FIELDS check on VehicleSpecs — keep in sync
        # if brake OEM fields are added or removed.
        return bool(
            self.front_pad_oem or self.rear_pad_oem
            or self.front_disc_oem or self.rear_disc_oem
        )

    @classmethod
    def from_vehicle_specs(
        cls,
        specs: VehicleSpecs,
        source: Literal["exact", "model_year"],
        db: object,
    ) -> BrakeResult:
        """Extract brake fields from a VehicleSpecs and resolve cross-refs.

        Parameters
        ----------
        specs:
            The matched vehicle specification record.
        source:
            Which lookup tier produced this match.
        db:
            A :class:`~parts_finder.db.PartsDatabase` instance used to
            resolve OEM numbers to aftermarket cross-references.  Typed
            as ``object`` to avoid a circular import.
        """
        oem_values = {
            result_field: getattr(specs, specs_field)
            for result_field, specs_field in _BRAKE_SPEC_FIELDS.items()
        }

        # Resolve cross-references for each non-empty OEM number.
        crossref_map: dict[str, tuple[ProductCrossRef, ...]] = {}
        oem_to_crossref = {
            "front_pad_oem": "front_pad_crossrefs",
            "rear_pad_oem": "rear_pad_crossrefs",
            "front_disc_oem": "front_disc_crossrefs",
            "rear_disc_oem": "rear_disc_crossrefs",
        }
        for oem_field, crossref_field in oem_to_crossref.items():
            oem = oem_values[oem_field]
            if oem:
                crossref_map[crossref_field] = tuple(db.find_crossrefs(oem))
            else:
                crossref_map[crossref_field] = ()

        return cls(
            **oem_values,
            **crossref_map,
            source=source,
        )


@dataclass(frozen=True)
class ProductCrossRef:
    """Maps an OEM part number to an aftermarket equivalent.

    Multiple records can exist per OEM number (one per brand), enabling
    the user to compare alternatives across aftermarket suppliers.
    """

    oem_part_number: str
    category: str
    brand: str
    brand_part_number: str
    notes: str = ""
