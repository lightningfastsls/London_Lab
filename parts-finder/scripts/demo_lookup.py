"""End-to-end parts lookup demo.

Takes a license plate, resolves it via data.gov.il, matches against
the local database, and prints a formatted report across all categories.

Usage:
    python scripts/demo_lookup.py --plate 12-345-67 --db parts_finder.db
    python scripts/demo_lookup.py --plate 1234567    # dashes optional
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow running from repo root: parts-finder/scripts/demo_lookup.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from parts_finder.config import AppConfig
from parts_finder.db import PartsDatabase
from parts_finder.exceptions import GovApiError, PlateNotFoundError
from parts_finder.lookup_engine import LookupEngine, LookupResult
from parts_finder.models import (
    BrakeResult,
    BulbResult,
    CoolantResult,
    FilterResult,
    OilResult,
)
from parts_finder.name_mapper import NameMapper


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Look up vehicle parts by Israeli license plate number.",
    )
    parser.add_argument(
        "--plate",
        required=True,
        help="License plate number (dashes optional, e.g. 12-345-67 or 1234567)",
    )
    parser.add_argument(
        "--db",
        default="parts_finder.db",
        help="Path to SQLite database (default: parts_finder.db)",
    )
    parser.add_argument(
        "--mapping",
        default="data/hebrew_names.json",
        help="Path to Hebrew-to-English name mapping JSON (default: data/hebrew_names.json)",
    )
    return parser.parse_args(argv)


# ── Formatting helpers ───────────────────────────────────────────────


def _format_vehicle(result: LookupResult) -> str:
    """Format the vehicle identity header line."""
    v = result.vehicle
    make = v.make_english or v.make_hebrew
    model = v.model_english or v.model_hebrew
    return f"Vehicle: {make} {model} {v.year} ({v.engine_code}, {v.fuel_type})"


def _format_oil(oil: OilResult | None) -> str:
    """Format the oil section."""
    if oil is None:
        return "Oil:       [no match found]"
    lines = ["Oil:"]
    lines.append(
        f"  Viscosity: {oil.viscosity}  |  Capacity: {oil.capacity_l}L"
        f"  |  Spec: {oil.spec}"
    )
    if oil.oem_approval:
        lines.append(f"  OEM approval: {oil.oem_approval}")
    lines.append(f"  Change interval: {oil.change_interval_km:,} km")
    lines.append(f"  Confidence: {oil.confidence} ({oil.source})")
    return "\n".join(lines)


def _format_coolant(coolant: CoolantResult | None) -> str:
    """Format the coolant section."""
    if coolant is None:
        return "Coolant:   [no match found]"
    lines = ["Coolant:"]
    cap = f"  |  Capacity: {coolant.capacity_l}L" if coolant.capacity_l else ""
    lines.append(f"  Type: {coolant.spec} ({coolant.color}){cap}")
    if coolant.aftermarket_match:
        lines.append(f"  Aftermarket: {coolant.aftermarket_match}")
    if coolant.mixing_warning:
        lines.append(f"  Warning: {coolant.mixing_warning}")
    lines.append(f"  Confidence: {coolant.source}")
    return "\n".join(lines)


def _format_brakes(brakes: BrakeResult | None) -> str:
    """Format the brakes section."""
    if brakes is None:
        return "Brakes:    [no match found]"
    if not brakes.has_data:
        return "Brakes:    [no match found]"
    lines = ["Brakes:"]
    if brakes.front_pad_oem:
        xrefs = ", ".join(
            f"{x.brand} {x.brand_part_number}" for x in brakes.front_pad_crossrefs
        )
        suffix = f" -> {xrefs}" if xrefs else ""
        lines.append(f"  Front pads: {brakes.front_pad_oem}{suffix}")
    if brakes.rear_pad_oem:
        xrefs = ", ".join(
            f"{x.brand} {x.brand_part_number}" for x in brakes.rear_pad_crossrefs
        )
        suffix = f" -> {xrefs}" if xrefs else ""
        lines.append(f"  Rear pads: {brakes.rear_pad_oem}{suffix}")
    if brakes.front_disc_oem:
        xrefs = ", ".join(
            f"{x.brand} {x.brand_part_number}" for x in brakes.front_disc_crossrefs
        )
        suffix = f" -> {xrefs}" if xrefs else ""
        lines.append(f"  Front discs: {brakes.front_disc_oem}{suffix}")
    if brakes.rear_disc_oem:
        xrefs = ", ".join(
            f"{x.brand} {x.brand_part_number}" for x in brakes.rear_disc_crossrefs
        )
        suffix = f" -> {xrefs}" if xrefs else ""
        lines.append(f"  Rear discs: {brakes.rear_disc_oem}{suffix}")
    if brakes.brake_fluid_type:
        lines.append(f"  Brake fluid: {brakes.brake_fluid_type}")
    lines.append(f"  Confidence: {brakes.source}")
    return "\n".join(lines)


def _format_bulbs(bulbs: BulbResult | None) -> str:
    """Format the bulbs section."""
    if bulbs is None:
        return "Bulbs:     [no match found]"
    replaceable = bulbs.replaceable_positions
    if not replaceable:
        return "Bulbs:     [no match found]"
    lines = ["Bulbs:"]
    labels = {
        "low_beam": "Low beam",
        "high_beam": "High beam",
        "front_turn": "Front turn",
        "rear_turn": "Rear turn",
        "tail_brake": "Tail/brake",
        "reverse": "Reverse",
        "fog": "Fog",
        "license_plate": "License plate",
    }
    for pos, code in replaceable.items():
        lines.append(f"  {labels.get(pos, pos)}: {code}")
    lines.append(f"  Confidence: {bulbs.source}")
    return "\n".join(lines)


def _format_filters(filters: FilterResult | None) -> str:
    """Format the filters section."""
    if filters is None:
        return "Filters:   [no match found]"
    if not filters.has_data:
        return "Filters:   [no match found]"
    lines = ["Filters:"]
    label_map = {
        "oil_filter_oem": ("Oil filter", "oil_filter_crossrefs"),
        "air_filter_oem": ("Air filter", "air_filter_crossrefs"),
        "cabin_filter_oem": ("Cabin filter", "cabin_filter_crossrefs"),
        "fuel_filter_oem": ("Fuel filter", "fuel_filter_crossrefs"),
    }
    for oem_field, (label, xref_field) in label_map.items():
        oem = getattr(filters, oem_field)
        if oem:
            xrefs = ", ".join(
                f"{x.brand} {x.brand_part_number}"
                for x in getattr(filters, xref_field)
            )
            suffix = f" -> {xrefs}" if xrefs else ""
            lines.append(f"  {label}: {oem}{suffix}")
    lines.append(f"  Confidence: {filters.source}")
    return "\n".join(lines)


# Categories not yet implemented — shown as placeholders
_FUTURE_CATEGORIES = ("Belts", "Spark Plugs")


def format_report(result: LookupResult) -> str:
    """Format a complete lookup result as a human-readable report.

    Shows all 7 categories: 5 implemented (oil, coolant, brakes, bulbs,
    filters) and 2 future (belts, spark plugs).
    """
    sections = [
        _format_vehicle(result),
        "",
        _format_oil(result.oil),
        "",
        _format_coolant(result.coolant),
        "",
        _format_brakes(result.brakes),
        "",
        _format_bulbs(result.bulbs),
        "",
        _format_filters(result.filters),
    ]

    # Future categories
    for cat in _FUTURE_CATEGORIES:
        sections.append("")
        sections.append(f"{cat}:{'  ' if len(cat) < 8 else ' '}[not yet in database]")

    return "\n".join(sections)


# ── Main entry point ─────────────────────────────────────────────────


async def _run(args: argparse.Namespace) -> None:
    """Async entry point — resolves plate and prints report."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}. Run seed_database.py first.")
        sys.exit(1)

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"Error: Name mapping file not found at {mapping_path}.")
        sys.exit(1)

    config = AppConfig(db_path=str(db_path))
    with PartsDatabase(config.db_path) as db:
        mapper = NameMapper(mapping_path)
        engine = LookupEngine(config, db, mapper)

        try:
            result = await engine.lookup(args.plate)
        except PlateNotFoundError as exc:
            print(f"Error: Plate {exc.plate} not found in the registry.")
            sys.exit(1)
        except GovApiError as exc:
            if "timeout" in str(exc).lower():
                print("Error: Government API timed out. Try again later.")
            else:
                print(f"Error: Government API error — {exc}")
            sys.exit(1)
        except ValueError as exc:
            print(f"Error: Invalid plate number — {exc}")
            sys.exit(1)

    print(format_report(result))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
