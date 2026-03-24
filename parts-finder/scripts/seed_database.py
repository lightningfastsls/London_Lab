"""Seed the Parts Finder database from oil-finder-free.jsx.

End-to-end orchestrator: JSX → CSV → SQLite.

Usage::

    python scripts/seed_database.py --db parts.db
    python scripts/seed_database.py --db parts.db --jsx ../../oil-finder-free.jsx
    python scripts/seed_database.py --db parts.db --year-from 2019 --year-to 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Bootstrap: add src/ and scripts/ to path ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (SRC_ROOT, SCRIPTS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from import_data import import_csv  # noqa: E402
from parts_finder.db import PartsDatabase  # noqa: E402
from seed_from_jsx import (  # noqa: E402
    DEFAULT_JSX,
    DEFAULT_OUTPUT,
    export_csv,
    iter_oil_entries,
    parse_oil_db,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the Parts Finder DB from oil-finder-free.jsx (JSX → CSV → SQLite).",
    )
    parser.add_argument(
        "--db", required=True,
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--jsx", type=Path, default=DEFAULT_JSX,
        help="Path to oil-finder-free.jsx",
    )
    parser.add_argument(
        "--year-from", type=int, default=2018,
        help="Default year_from (default: 2018)",
    )
    parser.add_argument(
        "--year-to", type=int, default=2025,
        help="Default year_to (default: 2025)",
    )
    parser.add_argument(
        "--csv-output", type=Path, default=DEFAULT_OUTPUT,
        help="Intermediate CSV path (default: data/seed/oil_db_seed.csv)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full seed pipeline. Returns 0 on success, 1 on error."""
    args = parse_args(argv)

    # Step 1: Parse JSX and export CSV
    print(f"[1/2] Parsing {args.jsx.name} ...")
    jsx_text = args.jsx.read_text(encoding="utf-8")
    oil_db = parse_oil_db(jsx_text)

    total = sum(
        len(engines)
        for models in oil_db.values()
        for engines in models.values()
    )

    entries = list(iter_oil_entries(oil_db))
    skipped = total - len(entries)
    export_csv(entries, args.csv_output, args.year_from, args.year_to)

    print(f"       Extracted {len(entries)} entries ({skipped} EVs skipped)")
    print(f"       CSV written to {args.csv_output}")

    # Step 2: Import CSV into SQLite
    print(f"\n[2/2] Importing into {args.db} ...")
    with PartsDatabase(args.db) as db:
        report = import_csv(db, args.csv_output, "specs")

    print(f"       Inserted: {report.inserted}")
    print(f"       Updated:  {report.updated}")
    print(f"       Errors:   {report.errors}")

    if report.errors:
        for detail in report.error_details:
            print(f"       ERROR: {detail}")
        return 1

    print(f"\nDone. Database seeded with {report.inserted} vehicle oil specs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
