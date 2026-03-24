"""Import brake pad specs from CSV files into the database.

Handles two operations:
1. Updates vehicle_specs rows with OEM brake pad part numbers
   (front_brake_pad_oem, rear_brake_pad_oem) — updates ALL engine variants
   matching the make/model/year range since brake pads are per-platform.
2. Inserts aftermarket cross-references (Brembo, TRW, Textar, Ferodo, Bosch,
   Mintex) into product_crossref.

Usage::

    python scripts/import_brake_pads.py --db parts_finder.db data/seed/brake_pads_specs.csv
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

# Aftermarket brand columns in the CSV → (csv_column, brand_name)
_AFTERMARKET_BRANDS = [
    ("brembo", "Brembo"),
    ("trw", "TRW"),
    ("textar", "Textar"),
    ("ferodo", "Ferodo"),
    ("bosch", "Bosch"),
    ("mintex", "Mintex"),
]


def import_brake_pads_csv(db_path: str, csv_path: Path) -> dict:
    """Import a single brake pads CSV into the database."""
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
                axle = row["axle"].strip().lower()  # "front" or "rear"
                oem_pn = row["oem_part_number"].strip()

                if axle not in ("front", "rear"):
                    stats["errors"] += 1
                    stats["error_details"].append(
                        f"Row {row_num}: unknown axle '{axle}'"
                    )
                    continue

                # Determine which column to update
                col = "front_brake_pad_oem" if axle == "front" else "rear_brake_pad_oem"
                category = "front_brake_pad" if axle == "front" else "rear_brake_pad"

                # Find ALL matching rows (brake pads are per-platform, not
                # per-engine, so update every engine variant in the range)
                matching = conn.execute(
                    "SELECT id FROM vehicle_specs"
                    " WHERE make = ? AND model = ?"
                    " AND year_from <= ? AND year_to >= ?",
                    (make, model, year_to, year_from),
                ).fetchall()

                if matching:
                    for m in matching:
                        conn.execute(
                            f"UPDATE vehicle_specs SET {col} = ? WHERE id = ?",
                            (oem_pn, m["id"]),
                        )
                    stats["specs_updated"] += len(matching)
                else:
                    # No existing row — insert a minimal one with brake data
                    conn.execute(
                        "INSERT INTO vehicle_specs"
                        " (make, model, year_from, year_to, engine_code,"
                        f"  fuel_type, {col})"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (make, model, year_from, year_to, "", "", oem_pn),
                    )
                    stats["specs_inserted"] += 1

                # Insert cross-references for all 6 aftermarket brands
                if oem_pn:
                    for csv_col, brand in _AFTERMARKET_BRANDS:
                        part_num = row.get(csv_col, "").strip()
                        if not part_num:
                            continue
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO product_crossref"
                                " (oem_part_number, category, brand,"
                                "  brand_part_number)"
                                " VALUES (?, ?, ?, ?)",
                                (oem_pn, category, brand, part_num),
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
    parser = argparse.ArgumentParser(description="Import brake pad specs CSVs")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("files", nargs="+", type=Path, help="CSV files to import")
    args = parser.parse_args()

    for csv_path in args.files:
        stats = import_brake_pads_csv(args.db, csv_path)
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
