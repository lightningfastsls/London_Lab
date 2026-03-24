"""Import filter specs from CSV files into the database.

Handles two operations:
1. Updates vehicle_specs rows with OEM filter part numbers
   (oil_filter_oem, air_filter_oem, cabin_filter_oem)
2. Inserts aftermarket cross-references (Mann, Filtron) into product_crossref

Usage::

    python scripts/import_filters.py --db parts_finder.db data/seed/filter_specs.csv
    python scripts/import_filters.py --db parts_finder.db data/seed/*.csv
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def import_filter_csv(db_path: str, csv_path: Path) -> dict:
    """Import a single filter CSV into the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    stats = {
        "file": csv_path.name,
        "rows": 0,
        "specs_updated": 0,
        "specs_inserted": 0,
        "crossrefs_inserted": 0,
        "errors": 0,
        "error_details": [],
    }

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            stats["rows"] += 1
            try:
                make = row["make"].strip()
                model = row["model"].strip()
                year_from = int(row["year_from"])
                year_to = int(row["year_to"])
                engine_code = row["engine_code"].strip()
                fuel_type = row.get("fuel_type", "").strip()

                oil_oem = row.get("oil_filter_oem", "").strip()
                air_oem = row.get("air_filter_oem", "").strip()
                cabin_oem = row.get("cabin_filter_oem", "").strip()

                # Check if row exists in vehicle_specs
                existing = conn.execute(
                    "SELECT id, fuel_type FROM vehicle_specs"
                    " WHERE make = ? AND model = ? AND year_from = ?"
                    " AND year_to = ? AND engine_code = ?",
                    (make, model, year_from, year_to, engine_code),
                ).fetchone()

                if existing:
                    # Update existing row with filter OEM numbers
                    updates = []
                    params = []
                    if oil_oem:
                        updates.append("oil_filter_oem = ?")
                        params.append(oil_oem)
                    if air_oem:
                        updates.append("air_filter_oem = ?")
                        params.append(air_oem)
                    if cabin_oem:
                        updates.append("cabin_filter_oem = ?")
                        params.append(cabin_oem)
                    if updates:
                        params.append(existing["id"])
                        conn.execute(
                            f"UPDATE vehicle_specs SET {', '.join(updates)}"
                            f" WHERE id = ?",
                            params,
                        )
                        stats["specs_updated"] += 1
                else:
                    # Insert new row with filter data only
                    conn.execute(
                        "INSERT INTO vehicle_specs"
                        " (make, model, year_from, year_to, engine_code,"
                        "  fuel_type, oil_filter_oem, air_filter_oem,"
                        "  cabin_filter_oem)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (make, model, year_from, year_to, engine_code,
                         fuel_type, oil_oem, air_oem, cabin_oem),
                    )
                    stats["specs_inserted"] += 1

                # Insert cross-references for aftermarket brands
                _BRANDS = {
                    "oil_filter": [
                        ("oil_filter_mann", "Mann"),
                        ("oil_filter_filtron", "Filtron"),
                    ],
                    "air_filter": [
                        ("air_filter_mann", "Mann"),
                        ("air_filter_filtron", "Filtron"),
                    ],
                    "cabin_filter": [
                        ("cabin_filter_mann", "Mann"),
                        ("cabin_filter_filtron", "Filtron"),
                    ],
                }

                for category, brand_cols in _BRANDS.items():
                    # Get the OEM number for this category
                    oem = {"oil_filter": oil_oem, "air_filter": air_oem,
                           "cabin_filter": cabin_oem}.get(category, "")
                    if not oem:
                        continue
                    for col_name, brand in brand_cols:
                        part_num = row.get(col_name, "").strip()
                        if not part_num:
                            continue
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO product_crossref"
                                " (oem_part_number, category, brand,"
                                "  brand_part_number)"
                                " VALUES (?, ?, ?, ?)",
                                (oem, category, brand, part_num),
                            )
                            stats["crossrefs_inserted"] += 1
                        except sqlite3.IntegrityError:
                            pass  # duplicate, skip

            except (ValueError, KeyError) as exc:
                stats["errors"] += 1
                stats["error_details"].append(f"Row {row_num}: {exc}")

    conn.commit()
    conn.close()
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import filter specs CSVs")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("files", nargs="+", type=Path, help="CSV files to import")
    args = parser.parse_args()

    for csv_path in args.files:
        stats = import_filter_csv(args.db, csv_path)
        print(f"\n--- {stats['file']} ---")
        print(f"  Rows processed: {stats['rows']}")
        print(f"  Specs updated:  {stats['specs_updated']}")
        print(f"  Specs inserted: {stats['specs_inserted']}")
        print(f"  Cross-refs:     {stats['crossrefs_inserted']}")
        print(f"  Errors:         {stats['errors']}")
        for d in stats["error_details"]:
            print(f"  ERROR: {d}")


if __name__ == "__main__":
    main()
