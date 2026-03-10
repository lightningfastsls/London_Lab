"""Import vehicle specs and part cross-references from CSV files.

Phase 2b data population script.  Maps CSV rows to frozen dataclass
instances (VehicleSpecs / ProductCrossRef) and feeds them through the
existing PartsDatabase insert operations.

Usage::

    python scripts/import_data.py --db parts.db --specs data/oil_specs_sample.csv
    python scripts/import_data.py --db parts.db --crossref data/filter_crossref_sample.csv
    python scripts/import_data.py --db parts.db --all data/
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Union

# --- Bootstrap: add src/ to path so parts_finder is importable -----------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parts_finder.db import (  # noqa: E402
    PartsDatabase,
    _CROSSREF_FIELDS,
    _SPECS_FIELDS,
)
from parts_finder.models import ProductCrossRef, VehicleSpecs  # noqa: E402

# --- Required columns (subset needed for UNIQUE constraints) -------------
_REQUIRED_SPECS = {"make", "model", "year_from", "year_to", "engine_code", "fuel_type"}
_REQUIRED_CROSSREF = {"oem_part_number", "category", "brand", "brand_part_number"}

# --- Type coercion maps (derived from dataclass field annotations) -------
_SPECS_INT_FIELDS = {
    f.name for f in dc_fields(VehicleSpecs) if f.type == "int" or f.type is int
}
_SPECS_FLOAT_FIELDS = {
    f.name for f in dc_fields(VehicleSpecs) if f.type == "float" or f.type is float
}


@dataclass
class ImportReport:
    """Summary of a single CSV import operation."""

    file: str
    table: str
    inserted: int
    updated: int
    errors: int
    unknown_columns: list[str]
    error_details: list[str]


# --- Public API ----------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import vehicle specs and part cross-references from CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/import_data.py --db parts.db --specs data/oil_specs_sample.csv
  python scripts/import_data.py --db parts.db --all data/
""",
    )
    parser.add_argument(
        "--db", required=True,
        help="Path to SQLite database file (use :memory: for testing)",
    )
    parser.add_argument(
        "--specs", type=Path,
        help="CSV file with vehicle specs rows",
    )
    parser.add_argument(
        "--crossref", type=Path,
        help="CSV file with product cross-reference rows",
    )
    parser.add_argument(
        "--all", type=Path, dest="all_dir",
        help="Directory of CSVs to auto-detect and import",
    )
    return parser.parse_args(argv)


def detect_csv_type(csv_path: Path) -> str:
    """Peek at CSV headers to determine if it's specs or crossref.

    Returns ``"specs"`` or ``"crossref"``.
    Raises ``ValueError`` if headers don't match either table.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
    header_set = {h.strip() for h in headers}

    specs_overlap = header_set & set(_SPECS_FIELDS)
    crossref_overlap = header_set & set(_CROSSREF_FIELDS)

    # Heuristic: whichever schema has more matching columns wins.
    # Tie-break by checking required columns.
    if specs_overlap >= _REQUIRED_SPECS:
        return "specs"
    if crossref_overlap >= _REQUIRED_CROSSREF:
        return "crossref"

    raise ValueError(
        f"Cannot auto-detect CSV type for {csv_path.name}: "
        f"headers {sorted(header_set)} don't match specs or crossref schema"
    )


def validate_headers(
    headers: list[str], table_type: str
) -> tuple[list[str], list[str]]:
    """Check CSV headers against the schema for *table_type*.

    Returns ``(valid_columns, unknown_columns)``.
    Raises ``ValueError`` if required columns are missing.
    """
    schema_fields = set(_SPECS_FIELDS if table_type == "specs" else _CROSSREF_FIELDS)
    required = _REQUIRED_SPECS if table_type == "specs" else _REQUIRED_CROSSREF

    valid = [h for h in headers if h in schema_fields]
    unknown = [h for h in headers if h not in schema_fields]

    missing = required - set(headers)
    if missing:
        raise ValueError(
            f"Missing required columns for {table_type}: {sorted(missing)}"
        )

    return valid, unknown


def coerce_row(
    row: dict[str, str], table_type: str
) -> Union[VehicleSpecs, ProductCrossRef]:
    """Convert a CSV row dict to a typed dataclass instance.

    Applies int/float coercion for VehicleSpecs numeric fields.
    All ProductCrossRef fields are strings (no coercion needed).

    Raises ``ValueError`` on type conversion failure.
    """
    if table_type == "specs":
        schema_fields = set(_SPECS_FIELDS)
        kwargs: dict = {}
        for key, value in row.items():
            if key not in schema_fields:
                continue
            if key in _SPECS_INT_FIELDS:
                kwargs[key] = int(value) if value.strip() else 0
            elif key in _SPECS_FLOAT_FIELDS:
                kwargs[key] = float(value) if value.strip() else 0.0
            else:
                kwargs[key] = value
        return VehicleSpecs(**kwargs)
    else:
        schema_fields = set(_CROSSREF_FIELDS)
        kwargs = {k: v for k, v in row.items() if k in schema_fields}
        return ProductCrossRef(**kwargs)


def import_csv(
    db: PartsDatabase, csv_path: Path, table_type: str
) -> ImportReport:
    """Import a single CSV file into the database.

    Returns an :class:`ImportReport` with counts and error details.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        valid_cols, unknown_cols = validate_headers(list(headers), table_type)

        # Snapshot row count before import
        table_name = "vehicle_specs" if table_type == "specs" else "product_crossref"
        before_count = db._conn.execute(
            f"SELECT COUNT(*) AS n FROM {table_name}"  # noqa: S608
        ).fetchone()["n"]

        errors = 0
        error_details: list[str] = []
        processed = 0

        for row_num, row in enumerate(reader, start=2):  # row 1 = header
            try:
                record = coerce_row(row, table_type)
                if table_type == "specs":
                    db.insert_specs(record)  # type: ignore[arg-type]
                else:
                    db.insert_crossref(record)  # type: ignore[arg-type]
                processed += 1
            except (ValueError, TypeError) as exc:
                errors += 1
                error_details.append(f"Row {row_num}: {exc}")

    after_count = db._conn.execute(
        f"SELECT COUNT(*) AS n FROM {table_name}"  # noqa: S608
    ).fetchone()["n"]

    inserted = after_count - before_count
    updated = processed - inserted

    return ImportReport(
        file=csv_path.name,
        table=table_type,
        inserted=inserted,
        updated=updated,
        errors=errors,
        unknown_columns=unknown_cols,
        error_details=error_details,
    )


def _print_report(report: ImportReport) -> None:
    """Pretty-print an import report to stdout."""
    print(f"\n--- {report.file} -> {report.table} ---")
    print(f"  Inserted: {report.inserted}")
    print(f"  Updated:  {report.updated}")
    print(f"  Errors:   {report.errors}")
    if report.unknown_columns:
        print(f"  WARNING: Unknown columns ignored: {report.unknown_columns}")
    for detail in report.error_details:
        print(f"  ERROR: {detail}")


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 on success, 1 on fatal error."""
    args = parse_args(argv)

    try:
        with PartsDatabase(args.db) as db:
            reports: list[ImportReport] = []

            if args.specs:
                reports.append(import_csv(db, args.specs, "specs"))

            if args.crossref:
                reports.append(import_csv(db, args.crossref, "crossref"))

            if args.all_dir:
                csv_dir = Path(args.all_dir)
                if not csv_dir.is_dir():
                    print(f"Error: {csv_dir} is not a directory", file=sys.stderr)
                    return 1
                for csv_file in sorted(csv_dir.glob("*.csv")):
                    try:
                        csv_type = detect_csv_type(csv_file)
                    except ValueError as exc:
                        print(f"Skipping {csv_file.name}: {exc}", file=sys.stderr)
                        continue
                    reports.append(import_csv(db, csv_file, csv_type))

            if not reports:
                print("No files to import. Use --specs, --crossref, or --all.",
                      file=sys.stderr)
                return 1

            for report in reports:
                _print_report(report)

    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
