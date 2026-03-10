"""SQLite database layer for vehicle specs and part cross-references.

Wraps stdlib ``sqlite3`` to provide a simple read/write interface for
offline-populated vehicle specifications and OEM-to-aftermarket part
mappings.  The database is populated via import scripts (Phase 2b) and
queried at runtime by the API layer (Phase 3).

Usage::

    with PartsDatabase("parts.db") as db:
        db.insert_specs(specs)
        result = db.find_specs("Toyota", "Corolla", 2021, "1ZR-FE")
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from parts_finder.models import ProductCrossRef, VehicleSpecs

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS vehicle_specs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    make TEXT NOT NULL,
    model TEXT NOT NULL,
    year_from INTEGER NOT NULL,
    year_to INTEGER NOT NULL,
    engine_code TEXT NOT NULL,
    fuel_type TEXT NOT NULL,
    oil_viscosity TEXT NOT NULL DEFAULT '',
    oil_capacity_l REAL NOT NULL DEFAULT 0.0,
    oil_spec TEXT NOT NULL DEFAULT '',
    oil_oem_approval TEXT NOT NULL DEFAULT '',
    oil_filter_oem TEXT NOT NULL DEFAULT '',
    oil_drain_plug_torque_nm INTEGER NOT NULL DEFAULT 0,
    oil_change_interval_km INTEGER NOT NULL DEFAULT 0,
    air_filter_oem TEXT NOT NULL DEFAULT '',
    cabin_filter_oem TEXT NOT NULL DEFAULT '',
    fuel_filter_oem TEXT NOT NULL DEFAULT '',
    front_brake_pad_oem TEXT NOT NULL DEFAULT '',
    rear_brake_pad_oem TEXT NOT NULL DEFAULT '',
    front_rotor_oem TEXT NOT NULL DEFAULT '',
    rear_rotor_oem TEXT NOT NULL DEFAULT '',
    brake_fluid_type TEXT NOT NULL DEFAULT '',
    low_beam_bulb TEXT NOT NULL DEFAULT '',
    high_beam_bulb TEXT NOT NULL DEFAULT '',
    front_turn_bulb TEXT NOT NULL DEFAULT '',
    rear_turn_bulb TEXT NOT NULL DEFAULT '',
    tail_brake_bulb TEXT NOT NULL DEFAULT '',
    reverse_bulb TEXT NOT NULL DEFAULT '',
    fog_bulb TEXT NOT NULL DEFAULT '',
    license_plate_bulb TEXT NOT NULL DEFAULT '',
    coolant_type TEXT NOT NULL DEFAULT '',
    coolant_capacity_l REAL NOT NULL DEFAULT 0.0,
    thermostat_temp_c INTEGER NOT NULL DEFAULT 0,
    timing_belt_oem TEXT NOT NULL DEFAULT '',
    serpentine_belt_oem TEXT NOT NULL DEFAULT '',
    spark_plug_oem TEXT NOT NULL DEFAULT '',
    spark_plug_gap_mm REAL NOT NULL DEFAULT 0.0,
    UNIQUE(make, model, year_from, year_to, engine_code)
);

CREATE INDEX IF NOT EXISTS idx_specs_lookup
    ON vehicle_specs (make, model, engine_code);

CREATE TABLE IF NOT EXISTS product_crossref (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    oem_part_number TEXT NOT NULL,
    category TEXT NOT NULL,
    brand TEXT NOT NULL,
    brand_part_number TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE(oem_part_number, brand, brand_part_number)
);

CREATE INDEX IF NOT EXISTS idx_crossref_oem
    ON product_crossref (oem_part_number);
"""

# Field names matching the VehicleSpecs dataclass (order must match schema).
_SPECS_FIELDS = [
    "make", "model", "year_from", "year_to", "engine_code", "fuel_type",
    "oil_viscosity", "oil_capacity_l", "oil_spec", "oil_oem_approval",
    "oil_filter_oem", "oil_drain_plug_torque_nm", "oil_change_interval_km",
    "air_filter_oem", "cabin_filter_oem",
    "fuel_filter_oem", "front_brake_pad_oem", "rear_brake_pad_oem",
    "front_rotor_oem", "rear_rotor_oem", "brake_fluid_type",
    "low_beam_bulb", "high_beam_bulb", "front_turn_bulb",
    "rear_turn_bulb", "tail_brake_bulb", "reverse_bulb",
    "fog_bulb", "license_plate_bulb", "coolant_type", "coolant_capacity_l",
    "thermostat_temp_c", "timing_belt_oem", "serpentine_belt_oem",
    "spark_plug_oem", "spark_plug_gap_mm",
]

_CROSSREF_FIELDS = [
    "oem_part_number", "category", "brand", "brand_part_number", "notes",
]


class PartsDatabase:
    """Thin wrapper around a SQLite database of vehicle specs and parts.

    Supports ``db_path=":memory:"`` for fast in-memory testing.
    Schema is created idempotently on construction (IF NOT EXISTS).
    """

    def __init__(
        self,
        db_path: str,
        *,
        check_same_thread: bool = True,
    ) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> PartsDatabase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # --- Specs ---------------------------------------------------------

    def insert_specs(self, specs: VehicleSpecs) -> None:
        """Insert or replace a vehicle specs record.

        Uses INSERT OR REPLACE keyed on the UNIQUE constraint
        (make, model, year_from, year_to, engine_code).

        .. note:: INSERT OR REPLACE deletes the conflicting row and inserts
           a new one, so the ``id`` column value changes on every upsert.
           Do not use ``id`` as a stable foreign key.
        """
        cols = ", ".join(_SPECS_FIELDS)
        placeholders = ", ".join("?" for _ in _SPECS_FIELDS)
        values = tuple(getattr(specs, f) for f in _SPECS_FIELDS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO vehicle_specs ({cols})"
            f" VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def find_specs(
        self,
        make: str,
        model: str,
        year: int,
        engine_code: str,
    ) -> Optional[VehicleSpecs]:
        """Find specs matching a vehicle identity.

        The query year must fall within the record's [year_from, year_to]
        range (inclusive).  Returns the first match or ``None``.
        """
        row = self._conn.execute(
            "SELECT * FROM vehicle_specs"
            " WHERE make = ? AND model = ? AND engine_code = ?"
            " AND year_from <= ? AND year_to >= ?"
            " LIMIT 1",
            (make, model, engine_code, year, year),
        ).fetchone()
        if row is None:
            return None
        return VehicleSpecs(**{f: row[f] for f in _SPECS_FIELDS})

    def find_specs_by_model_year_for_oil(
        self,
        make: str,
        model: str,
        year: int,
        fuel_type: str | None = None,
    ) -> Optional[VehicleSpecs]:
        """Find specs matching make + model + year, preferring oil data.

        Like :meth:`find_specs_by_model_year` but ORDER BY prefers records
        with ``oil_viscosity != ''``.  When *fuel_type* is provided and
        multiple records match (different engine variants for the same
        model-year), the record matching the fuel type is preferred.

        This tier bridges the gap when the government API returns engine
        codes (e.g. ``"G4FD"``) that don't match the DB's descriptive
        engine labels (e.g. ``"1.6T Petrol"``).
        """
        if fuel_type:
            # Try fuel-type-specific match first
            row = self._conn.execute(
                "SELECT * FROM vehicle_specs"
                " WHERE make = ? AND model = ? AND fuel_type = ?"
                " AND year_from <= ? AND year_to >= ?"
                " AND oil_viscosity != ''"
                " ORDER BY (CASE WHEN oil_viscosity != '' THEN 0 ELSE 1 END)"
                " LIMIT 1",
                (make, model, fuel_type, year, year),
            ).fetchone()
            if row is not None:
                return VehicleSpecs(**{f: row[f] for f in _SPECS_FIELDS})

        # Fall back to any engine variant with oil data
        row = self._conn.execute(
            "SELECT * FROM vehicle_specs"
            " WHERE make = ? AND model = ?"
            " AND year_from <= ? AND year_to >= ?"
            " ORDER BY (CASE WHEN oil_viscosity != '' THEN 0 ELSE 1 END)"
            " LIMIT 1",
            (make, model, year, year),
        ).fetchone()
        if row is None:
            return None
        return VehicleSpecs(**{f: row[f] for f in _SPECS_FIELDS})

    def find_specs_by_engine_family(
        self,
        make: str,
        engine_prefix: str,
        year: int,
    ) -> Optional[VehicleSpecs]:
        """Find specs matching a make and engine family prefix.

        Uses ``LIKE 'prefix%'`` to match engine codes sharing the same
        family (e.g. "2ZR" matches "2ZR-FE", "2ZR-FAE").  Returns the
        first match within the year range, or ``None``.
        """
        row = self._conn.execute(
            "SELECT * FROM vehicle_specs"
            " WHERE make = ? AND engine_code LIKE ?"
            " AND year_from <= ? AND year_to >= ?"
            " LIMIT 1",
            (make, f"{engine_prefix}%", year, year),
        ).fetchone()
        if row is None:
            return None
        return VehicleSpecs(**{f: row[f] for f in _SPECS_FIELDS})

    def find_specs_by_model_year(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[VehicleSpecs]:
        """Find specs matching make + model + year, ignoring engine_code.

        Used for bulb lookups where different engine variants on the same
        chassis typically share identical light housings.  Returns the
        first match within the year range, or ``None``.
        """
        row = self._conn.execute(
            "SELECT * FROM vehicle_specs"
            " WHERE make = ? AND model = ?"
            " AND year_from <= ? AND year_to >= ?"
            " ORDER BY (CASE WHEN low_beam_bulb != '' THEN 0 ELSE 1 END)"
            " LIMIT 1",
            (make, model, year, year),
        ).fetchone()
        if row is None:
            return None
        return VehicleSpecs(**{f: row[f] for f in _SPECS_FIELDS})

    def find_specs_by_model_year_for_coolant(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[VehicleSpecs]:
        """Find specs matching make + model + year, preferring coolant data.

        Like :meth:`find_specs_by_model_year` but ORDER BY prefers records
        with ``coolant_type != ''`` instead of ``low_beam_bulb != ''``.

        Purpose-specific queries are the established pattern (bulb query
        prefers bulb-populated rows; coolant query prefers coolant-populated
        rows).  When the same model has two records — one with bulb data but
        no coolant, another with coolant but no bulb — each query returns
        the appropriate record.
        """
        row = self._conn.execute(
            "SELECT * FROM vehicle_specs"
            " WHERE make = ? AND model = ?"
            " AND year_from <= ? AND year_to >= ?"
            " ORDER BY (CASE WHEN coolant_type != '' THEN 0 ELSE 1 END)"
            " LIMIT 1",
            (make, model, year, year),
        ).fetchone()
        if row is None:
            return None
        return VehicleSpecs(**{f: row[f] for f in _SPECS_FIELDS})

    def find_specs_by_model_year_for_brakes(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[VehicleSpecs]:
        """Find specs matching make + model + year, preferring brake data.

        Like :meth:`find_specs_by_model_year` but ORDER BY prefers records
        with ``front_brake_pad_oem != ''`` instead of ``low_beam_bulb != ''``.

        Purpose-specific queries are the established pattern — each lookup
        module gets a query that prefers rows populated with *its* data.
        """
        row = self._conn.execute(
            "SELECT * FROM vehicle_specs"
            " WHERE make = ? AND model = ?"
            " AND year_from <= ? AND year_to >= ?"
            " ORDER BY (CASE WHEN front_brake_pad_oem != '' THEN 0 ELSE 1 END)"
            " LIMIT 1",
            (make, model, year, year),
        ).fetchone()
        if row is None:
            return None
        return VehicleSpecs(**{f: row[f] for f in _SPECS_FIELDS})

    def find_specs_by_model_year_for_filters(
        self,
        make: str,
        model: str,
        year: int,
    ) -> Optional[VehicleSpecs]:
        """Find specs matching make + model + year, preferring filter data.

        Like :meth:`find_specs_by_model_year` but ORDER BY prefers records
        with ``air_filter_oem != ''`` instead of ``low_beam_bulb != ''``.

        Purpose-specific queries are the established pattern — each lookup
        module gets a query that prefers rows populated with *its* data.
        """
        row = self._conn.execute(
            "SELECT * FROM vehicle_specs"
            " WHERE make = ? AND model = ?"
            " AND year_from <= ? AND year_to >= ?"
            " ORDER BY (CASE WHEN air_filter_oem != '' THEN 0 ELSE 1 END)"
            " LIMIT 1",
            (make, model, year, year),
        ).fetchone()
        if row is None:
            return None
        return VehicleSpecs(**{f: row[f] for f in _SPECS_FIELDS})

    def find_brand_default_oil(
        self,
        make: str,
    ) -> Optional[tuple[str, str, int]]:
        """Find the most common oil spec for a make.

        Returns ``(viscosity, spec, change_interval_km)`` aggregated
        across all models, or ``None`` if the make has no records.

        Groups by all three columns intentionally: records with the same
        viscosity but different spec codes (e.g. "API SP" vs "ACEA C3")
        are distinct recommendations and must be counted separately.
        """
        row = self._conn.execute(
            "SELECT oil_viscosity, oil_spec, oil_change_interval_km,"
            " COUNT(*) AS cnt"
            " FROM vehicle_specs"
            " WHERE make = ? AND oil_viscosity != ''"
            " GROUP BY oil_viscosity, oil_spec, oil_change_interval_km"
            " ORDER BY cnt DESC"
            " LIMIT 1",
            (make,),
        ).fetchone()
        if row is None:
            return None
        return (row["oil_viscosity"], row["oil_spec"], row["oil_change_interval_km"])

    # --- Cross-references ----------------------------------------------

    def insert_crossref(self, crossref: ProductCrossRef) -> None:
        """Insert or replace a product cross-reference record.

        Uses INSERT OR REPLACE keyed on the UNIQUE constraint
        (oem_part_number, brand, brand_part_number).

        .. note:: INSERT OR REPLACE deletes the conflicting row and inserts
           a new one, so the ``id`` column value changes on every upsert.
           Do not use ``id`` as a stable foreign key.
        """
        cols = ", ".join(_CROSSREF_FIELDS)
        placeholders = ", ".join("?" for _ in _CROSSREF_FIELDS)
        values = tuple(getattr(crossref, f) for f in _CROSSREF_FIELDS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO product_crossref ({cols})"
            f" VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def find_crossrefs(self, oem_part_number: str) -> list[ProductCrossRef]:
        """Find all aftermarket equivalents for an OEM part number.

        Results are ordered by brand and part number for deterministic output.
        """
        rows = self._conn.execute(
            "SELECT * FROM product_crossref WHERE oem_part_number = ?"
            " ORDER BY brand, brand_part_number",
            (oem_part_number,),
        ).fetchall()
        return [
            ProductCrossRef(**{f: row[f] for f in _CROSSREF_FIELDS})
            for row in rows
        ]
