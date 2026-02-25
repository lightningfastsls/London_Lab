"""Extract OIL_DB from oil-finder-free.jsx and export as vehicle_specs CSV.

Parses the JS object literal using regex-based JS→JSON conversion (no JS
runtime needed), extracts fuel types from engine descriptions via
priority-ordered keyword matching, and writes a CSV compatible with
``import_data.py``.

Usage::

    python scripts/seed_from_jsx.py --jsx ../../oil-finder-free.jsx
    python scripts/seed_from_jsx.py --jsx ../../oil-finder-free.jsx --year-from 2019 --year-to 2024
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OilEntry:
    """Single row destined for the vehicle_specs table."""

    make: str
    model: str
    engine_desc: str  # raw key from OIL_DB (used as engine_code)
    fuel_type: str  # extracted via keyword matching
    viscosity: str
    capacity_l: float
    spec: str
    oem_approval: str  # JS "None" → ""
    interval_km: int


@dataclass
class SeedReport:
    """Summary of the JSX → CSV extraction."""

    total_entries: int
    skipped_ev: int
    exported: int
    output_file: str


# ---------------------------------------------------------------------------
# JS parsing helpers
# ---------------------------------------------------------------------------


def extract_oil_db_block(jsx_text: str) -> str:
    """Extract the ``OIL_DB = { ... };`` block via brace-counting.

    Returns the object literal (including outer braces).
    Raises ``ValueError`` if the block cannot be found.
    """
    match = re.search(r'(?:const|let|var)\s+OIL_DB\s*=\s*\{', jsx_text)
    if not match:
        raise ValueError("Could not find OIL_DB declaration in JSX source")

    # Start right at the opening brace
    start = match.end() - 1  # -1 to include the '{'
    depth = 0
    for i, ch in enumerate(jsx_text[start:], start=start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return jsx_text[start:i + 1]

    raise ValueError("Unbalanced braces in OIL_DB block")


def js_object_to_json(js_text: str) -> str:
    """Convert a JS object literal to valid JSON.

    Handles:
    - Unquoted keys (``viscosity:`` → ``"viscosity":``)
    - Trailing commas (``"foo",}`` → ``"foo"}``)
    - Single-line ``//`` comments
    - Block ``/* ... */`` comments
    """
    # Strip single-line comments
    text = re.sub(r'//[^\n]*', '', js_text)
    # Strip block comments (/* ... */)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Protect quoted strings from key-quoting regex: extract all "..."
    # values into placeholders, so colons inside strings are never touched
    strings: list[str] = []

    def _save(m: re.Match) -> str:
        strings.append(m.group(0))
        return f'"__S{len(strings) - 1}__"'

    text = re.sub(r'"(?:[^"\\]|\\.)*"', _save, text)

    # Quote unquoted keys (safe now — no string content to corrupt)
    text = re.sub(r'(?<!["\w])([A-Za-z_]\w*)\s*:', r'"\1":', text)

    # Restore original strings
    for i, s in enumerate(strings):
        text = text.replace(f'"__S{i}__"', s)

    # Strip trailing commas before } or ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def parse_oil_db(jsx_text: str) -> dict:
    """Extract and parse OIL_DB from JSX source into a Python dict."""
    block = extract_oil_db_block(jsx_text)
    json_text = js_object_to_json(block)
    return json.loads(json_text)


# ---------------------------------------------------------------------------
# Fuel-type extraction
# ---------------------------------------------------------------------------

# Priority-ordered rules: first match wins
_FUEL_RULES: list[tuple[str, str | None]] = [
    # Exact match for pure EVs — skip entirely (return None)
    ("Electric", None),
    # Mazda e-Skyactiv X is a petrol SPCCI engine, not electric
    ("e-Skyactiv X", "petrol"),
    # eHybrid (VW naming) is a plug-in hybrid
    ("eHybrid", "hybrid"),
    # PHEV before generic Hybrid
    ("PHEV", "phev"),
    # Mild Hybrid before generic Hybrid
    ("Mild Hybrid", "hybrid"),
    # Generic Hybrid
    ("Hybrid", "hybrid"),
    # Diesel keyword
    ("Diesel", "diesel"),
    # Petrol keyword
    ("Petrol", "petrol"),
]

# Regex-based rules for German makes (BMW/MB): "320d 2.0D" or "530i 2.0T"
_DIESEL_SUFFIX_RE = re.compile(r'\d+[Dd]\b')
_PETROL_SUFFIX_RE = re.compile(r'\d+[Ti]\b')

# Performance / sport keywords (always petrol)
_SPORT_KEYWORDS = {"GTI", "AMG", "Sport", "R", "V6", "V8"}


def extract_fuel_type(engine_desc: str) -> str | None:
    """Determine fuel type from an engine description string.

    Returns a fuel type string (``"petrol"``, ``"diesel"``, ``"hybrid"``,
    ``"phev"``) or ``None`` for electric vehicles (should be skipped).
    """
    # Priority keyword rules
    for keyword, fuel in _FUEL_RULES:
        if keyword in engine_desc:
            return fuel

    # Regex suffix rules (BMW: "318d 2.0D" → diesel, "320i 2.0T" → petrol)
    if _DIESEL_SUFFIX_RE.search(engine_desc):
        return "diesel"
    if _PETROL_SUFFIX_RE.search(engine_desc):
        return "petrol"

    # Sport / performance keywords
    for kw in _SPORT_KEYWORDS:
        if kw in engine_desc:
            return "petrol"

    # Fallback: assume petrol (most common for remaining entries)
    return "petrol"


# ---------------------------------------------------------------------------
# Entry iteration
# ---------------------------------------------------------------------------


def iter_oil_entries(oil_db: dict) -> Iterator[OilEntry]:
    """Yield :class:`OilEntry` for every non-EV engine in the database.

    Entries with all N/A values (electric vehicles) are silently skipped.
    The JS string ``"None"`` for OEM approval is mapped to ``""``.
    """
    for make, models in oil_db.items():
        for model, engines in models.items():
            for engine_desc, props in engines.items():
                fuel = extract_fuel_type(engine_desc)
                if fuel is None:
                    continue  # skip EVs

                oem = props.get("oem", "")
                if oem == "None":
                    oem = ""

                yield OilEntry(
                    make=make,
                    model=model,
                    engine_desc=engine_desc,
                    fuel_type=fuel,
                    viscosity=props.get("viscosity", ""),
                    capacity_l=float(props.get("capacity", 0)),
                    spec=props.get("spec", ""),
                    oem_approval=oem,
                    interval_km=int(props.get("interval", 0)),
                )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

# Column order matching vehicle_specs schema (subset with oil data)
CSV_COLUMNS = [
    "make", "model", "year_from", "year_to", "engine_code",
    "fuel_type", "oil_viscosity", "oil_capacity_l", "oil_spec",
    "oil_oem_approval", "oil_change_interval_km",
]


def export_csv(
    entries: list[OilEntry],
    output_path: Path,
    year_from: int = 2018,
    year_to: int = 2025,
) -> int:
    """Write entries to a CSV file compatible with ``import_data.py``.

    Returns the number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for entry in entries:
            writer.writerow({
                "make": entry.make,
                "model": entry.model,
                "year_from": year_from,
                "year_to": year_to,
                "engine_code": entry.engine_desc,
                "fuel_type": entry.fuel_type,
                "oil_viscosity": entry.viscosity,
                "oil_capacity_l": entry.capacity_l,
                "oil_spec": entry.spec,
                "oil_oem_approval": entry.oem_approval,
                "oil_change_interval_km": entry.interval_km,
            })

    return len(entries)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

DEFAULT_JSX = Path(__file__).resolve().parents[2] / "oil-finder-free.jsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "seed" / "oil_db_seed.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OIL_DB from JSX and export as vehicle_specs CSV.",
    )
    parser.add_argument(
        "--jsx", type=Path, default=DEFAULT_JSX,
        help="Path to oil-finder-free.jsx (default: repo root)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help="Output CSV path (default: data/seed/oil_db_seed.csv)",
    )
    parser.add_argument(
        "--year-from", type=int, default=2018,
        help="Default year_from for all entries (default: 2018)",
    )
    parser.add_argument(
        "--year-to", type=int, default=2025,
        help="Default year_to for all entries (default: 2025)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> SeedReport:
    """Parse JSX, extract entries, write CSV. Returns a SeedReport."""
    args = parse_args(argv)

    jsx_text = args.jsx.read_text(encoding="utf-8")
    oil_db = parse_oil_db(jsx_text)

    # Count total entries (including EVs)
    total = sum(
        len(engines)
        for models in oil_db.values()
        for engines in models.values()
    )

    entries = list(iter_oil_entries(oil_db))
    skipped = total - len(entries)

    rows_written = export_csv(entries, args.output, args.year_from, args.year_to)

    report = SeedReport(
        total_entries=total,
        skipped_ev=skipped,
        exported=rows_written,
        output_file=str(args.output),
    )

    print(f"Seed extraction complete:")
    print(f"  Total entries in OIL_DB: {report.total_entries}")
    print(f"  Skipped (EV):           {report.skipped_ev}")
    print(f"  Exported to CSV:        {report.exported}")
    print(f"  Output file:            {report.output_file}")

    return report


if __name__ == "__main__":
    main()
