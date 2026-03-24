"""Import vehicle bulb specs from CSV files into the database.

Updates vehicle_specs rows with bulb type data. Bulbs are per-platform (not
per-engine), so one CSV row updates ALL engine variants matching the
make/model/year range.

CSV columns → DB columns mapping:
    low_beam      → low_beam_bulb
    high_beam     → high_beam_bulb
    front_fog     → fog_bulb
    front_turn    → front_turn_bulb
    rear_turn     → rear_turn_bulb
    tail_brake    → tail_brake_bulb
    reverse       → reverse_bulb
    license_plate → license_plate_bulb

Usage::

    python scripts/import_bulbs.py --db parts_finder.db data/seed/vehicle_bulbs_specs.csv
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

# CSV column → DB column
_BULB_MAP = {
    "low_beam": "low_beam_bulb",
    "high_beam": "high_beam_bulb",
    "front_fog": "fog_bulb",
    "front_turn": "front_turn_bulb",
    "rear_turn": "rear_turn_bulb",
    "tail_brake": "tail_brake_bulb",
    "reverse": "reverse_bulb",
    "license_plate": "license_plate_bulb",
}


def import_bulbs_csv(db_path: str, csv_path: Path) -> dict:
    """Import a single bulbs CSV into the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    stats = {
        "file": csv_path.name,
        "rows": 0,
        "specs_updated": 0,
        "specs_inserted": 0,
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

                # Build SET clause from non-empty bulb columns
                updates = []
                params = []
                for csv_col, db_col in _BULB_MAP.items():
                    val = row.get(csv_col, "").strip()
                    # Treat "—" (em dash) as empty/not-applicable
                    if val and val != "—":
                        updates.append(f"{db_col} = ?")
                        params.append(val)

                if not updates:
                    continue

                # Find ALL matching rows (bulbs are per-platform)
                matching = conn.execute(
                    "SELECT id FROM vehicle_specs"
                    " WHERE make = ? AND model = ?"
                    " AND year_from <= ? AND year_to >= ?",
                    (make, model, year_to, year_from),
                ).fetchall()

                if matching:
                    for m in matching:
                        conn.execute(
                            f"UPDATE vehicle_specs SET {', '.join(updates)}"
                            f" WHERE id = ?",
                            params + [m["id"]],
                        )
                    stats["specs_updated"] += len(matching)
                else:
                    # No existing row — insert a minimal one with bulb data
                    bulb_cols = [db_col for csv_col, db_col in _BULB_MAP.items()
                                 if row.get(csv_col, "").strip()
                                 and row.get(csv_col, "").strip() != "—"]
                    bulb_vals = [row[csv_col].strip() for csv_col, db_col in _BULB_MAP.items()
                                 if row.get(csv_col, "").strip()
                                 and row.get(csv_col, "").strip() != "—"]
                    all_cols = ["make", "model", "year_from", "year_to",
                                "engine_code", "fuel_type"] + bulb_cols
                    all_vals = [make, model, year_from, year_to, "", ""] + bulb_vals
                    placeholders = ", ".join("?" for _ in all_cols)
                    conn.execute(
                        f"INSERT INTO vehicle_specs ({', '.join(all_cols)})"
                        f" VALUES ({placeholders})",
                        all_vals,
                    )
                    stats["specs_inserted"] += 1

            except (ValueError, KeyError) as exc:
                stats["errors"] += 1
                stats["error_details"].append(f"Row {row_num}: {exc}")

    conn.commit()
    conn.close()
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import vehicle bulb specs CSVs")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("files", nargs="+", type=Path, help="CSV files to import")
    args = parser.parse_args()

    for csv_path in args.files:
        stats = import_bulbs_csv(args.db, csv_path)
        print(f"\n--- {stats['file']} ---")
        print(f"  Rows processed: {stats['rows']}")
        print(f"  Specs updated:  {stats['specs_updated']}")
        print(f"  Specs inserted: {stats['specs_inserted']}")
        print(f"  Errors:         {stats['errors']}")
        for d in stats["error_details"]:
            print(f"  ERROR: {d}")


if __name__ == "__main__":
    main()
