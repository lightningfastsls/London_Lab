"""Diagnostic script to trace the full plate-lookup pipeline.

Calls the government API directly, prints ALL raw fields, runs NameMapper,
attempts DB lookup at each tier, and reports where the pipeline breaks.

Usage:
    python scripts/diagnose_lookup.py --plate 4552333
    python scripts/diagnose_lookup.py --plate 4552333 --db parts_finder.db
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

# Fix Windows console encoding for Hebrew output
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Allow running from repo root: parts-finder/scripts/diagnose_lookup.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import httpx

from parts_finder.config import AppConfig
from parts_finder.db import PartsDatabase
from parts_finder.models import VehicleRecord
from parts_finder.name_mapper import NameMapper
from parts_finder.oil_lookup import OilLookup, extract_engine_family


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose plate lookup pipeline")
    parser.add_argument("--plate", required=True, help="License plate number")
    parser.add_argument("--db", default="parts_finder.db", help="SQLite DB path")
    parser.add_argument(
        "--mapping", default="data/hebrew_names.json",
        help="Hebrew-to-English name mapping JSON",
    )
    return parser.parse_args(argv)


async def fetch_raw_record(plate: str) -> dict:
    """Call gov API and return the raw record dict."""
    digits = "".join(ch for ch in plate if ch.isdigit()).lstrip("0")
    config = AppConfig()
    params = {
        "resource_id": config.gov_api_resource_id,
        "filters": json.dumps({"mispar_rechev": digits}),
        "limit": 1,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(config.gov_api_base_url, params=params)
        resp.raise_for_status()
        data = resp.json()

    records = data.get("result", {}).get("records", [])
    if not records:
        print(f"ERROR: Plate {digits} not found in registry")
        sys.exit(1)
    return records[0]


def diagnose(record: dict, plate: str, db_path: str, mapping_path: str) -> None:
    """Trace the pipeline step by step."""
    sep = "=" * 60

    # ── Step 1: Raw API fields ────────────────────────────────────
    print(f"\n{sep}")
    print("STEP 1: Raw API Record")
    print(sep)
    key_fields = [
        "tozeret_nm", "kinuy_mishari", "shnat_yitzur", "degem_manoa",
        "sug_delek_nm", "tozeret_eretz_nm", "degem_nm", "tozeret_cd",
        "ramat_gimur", "misgeret",
    ]
    for field in key_fields:
        val = record.get(field, "<MISSING>")
        print(f"  {field:20s} = {val!r}")

    # ── Step 2: VehicleRecord construction ────────────────────────
    print(f"\n{sep}")
    print("STEP 2: VehicleRecord (from_api_record)")
    print(sep)
    vehicle = VehicleRecord.from_api_record(plate, record)
    print(f"  make_hebrew   = {vehicle.make_hebrew!r}")
    print(f"  model_hebrew  = {vehicle.model_hebrew!r}")
    print(f"  year          = {vehicle.year}")
    print(f"  engine_code   = {vehicle.engine_code!r}")
    print(f"  fuel_type     = {vehicle.fuel_type!r}")
    print(f"  make_english  = {vehicle.make_english!r}  (from tozeret_eretz_nm)")
    print(f"  model_english = {vehicle.model_english!r}  (from degem_nm)")

    # ── Step 3: NameMapper enrichment ─────────────────────────────
    print(f"\n{sep}")
    print("STEP 3: NameMapper Enrichment")
    print(sep)
    mp = Path(mapping_path)
    if not mp.exists():
        print(f"  WARNING: Mapping file not found at {mp}")
        enriched = vehicle
    else:
        mapper = NameMapper(mp)
        enriched = mapper.enrich_vehicle(vehicle)
        print(f"  make_english  = {enriched.make_english!r}")
        print(f"  model_english = {enriched.model_english!r}")
        print(f"  fuel_type     = {enriched.fuel_type!r}  (translated from {vehicle.fuel_type!r})")

        # Check if API's tozeret_eretz_nm looks like a country name
        raw_eretz = record.get("tozeret_eretz_nm", "")
        if raw_eretz and enriched.make_english != raw_eretz:
            print(f"  NOTE: API tozeret_eretz_nm={raw_eretz!r} != mapped make={enriched.make_english!r}")
            print(f"        tozeret_eretz_nm may be a country name, not an English make")

    # ── Step 4: DB lookup attempt ─────────────────────────────────
    print(f"\n{sep}")
    print("STEP 4: Database Lookup (Oil)")
    print(sep)
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"  ERROR: Database not found at {db_file}")
        return

    make = enriched.make_english or enriched.make_hebrew
    model = enriched.model_english or enriched.model_hebrew
    print(f"  Lookup make  = {make!r}")
    print(f"  Lookup model = {model!r}")

    with PartsDatabase(str(db_file)) as db:
        # Show what's in the DB for this make
        rows = db._conn.execute(
            "SELECT DISTINCT model, engine_code, fuel_type, year_from, year_to, oil_viscosity "
            "FROM vehicle_specs WHERE make = ? ORDER BY model, year_from",
            (make,),
        ).fetchall()
        print(f"\n  DB records for make={make!r}: {len(rows)} rows")
        for row in rows[:20]:
            print(
                f"    {row['model']:20s} engine={row['engine_code']:20s} "
                f"fuel={row['fuel_type']:10s} years={row['year_from']}-{row['year_to']} "
                f"oil={row['oil_viscosity']}"
            )

        # Tier 1: exact
        print(f"\n  Tier 1 (exact): make={make!r} model={model!r} "
              f"year={enriched.year} engine={enriched.engine_code!r}")
        specs = db.find_specs(make, model, enriched.year, enriched.engine_code)
        if specs:
            print(f"    MATCH: {specs.oil_viscosity} / {specs.oil_spec}")
        else:
            print("    NO MATCH")

        # Tier 1.5: model-year (proposed new tier)
        print(f"\n  Tier 1.5 (model-year): make={make!r} model={model!r} "
              f"year={enriched.year}")
        specs_my = db.find_specs_by_model_year(make, model, enriched.year)
        if specs_my:
            print(f"    MATCH: engine={specs_my.engine_code!r} "
                  f"fuel={specs_my.fuel_type!r} oil={specs_my.oil_viscosity}")
        else:
            print("    NO MATCH")

        # Tier 2: engine family
        family = extract_engine_family(enriched.engine_code)
        print(f"\n  Tier 2 (engine family): prefix={family!r}")
        if family:
            specs_ef = db.find_specs_by_engine_family(make, family, enriched.year)
            if specs_ef:
                print(f"    MATCH: {specs_ef.oil_viscosity} / {specs_ef.oil_spec}")
            else:
                print("    NO MATCH")
        else:
            print("    SKIPPED (no dash in engine code)")

        # Tier 3: brand default
        print(f"\n  Tier 3 (brand default): make={make!r}")
        brand = db.find_brand_default_oil(make)
        if brand:
            visc, spec, interval = brand
            print(f"    MATCH: {visc} / {spec} / {interval}km")
        else:
            print("    NO MATCH")

        # Full OilLookup result
        print(f"\n  Full OilLookup.lookup() result:")
        oil = OilLookup(db)
        result = oil.lookup(enriched)
        if result:
            print(f"    viscosity={result.viscosity} spec={result.spec} "
                  f"confidence={result.confidence} source={result.source}")
        else:
            print("    None — no oil data found")


async def main() -> None:
    args = parse_args()
    record = await fetch_raw_record(args.plate)
    diagnose(record, args.plate, args.db, args.mapping)


if __name__ == "__main__":
    asyncio.run(main())
