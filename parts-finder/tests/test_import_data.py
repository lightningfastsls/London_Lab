"""Tests for the CSV import CLI (scripts/import_data.py)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from parts_finder.db import PartsDatabase

# Bootstrap handled by conftest.py; import the script module directly.
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from import_data import (  # noqa: E402
    ImportReport,
    coerce_row,
    detect_csv_type,
    import_csv,
    main,
    validate_headers,
)

DATA_DIR = REPO_ROOT / "data"


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    """Write a CSV file from headers and row lists."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


class TestImportSpecsCsv(unittest.TestCase):
    """Test 1: Import oil_specs_sample.csv and verify records in DB."""

    def test_import_specs_csv(self) -> None:
        specs_csv = DATA_DIR / "oil_specs_sample.csv"
        self.assertTrue(specs_csv.exists(), f"{specs_csv} not found")

        with PartsDatabase(":memory:") as db:
            report = import_csv(db, specs_csv, "specs")

            self.assertEqual(report.inserted, 3)
            self.assertEqual(report.errors, 0)

            # Verify a specific record round-trips
            result = db.find_specs("Toyota", "Corolla", 2020, "2ZR-FE")
            self.assertIsNotNone(result)
            self.assertEqual(result.oil_viscosity, "0W-20")
            self.assertAlmostEqual(result.oil_capacity_l, 4.2)

            result2 = db.find_specs("Hyundai", "i30", 2020, "G4FJ")
            self.assertIsNotNone(result2)
            self.assertEqual(result2.fuel_type, "petrol")


class TestIdempotentReimport(unittest.TestCase):
    """Test 2: Import same CSV twice, verify no duplicates."""

    def test_idempotent_reimport(self) -> None:
        specs_csv = DATA_DIR / "oil_specs_sample.csv"

        with PartsDatabase(":memory:") as db:
            report1 = import_csv(db, specs_csv, "specs")
            self.assertEqual(report1.inserted, 3)

            report2 = import_csv(db, specs_csv, "specs")
            self.assertEqual(report2.inserted, 0)
            self.assertEqual(report2.updated, 3)

            # Total rows should still be 3
            count = db._conn.execute(
                "SELECT COUNT(*) AS n FROM vehicle_specs"
            ).fetchone()["n"]
            self.assertEqual(count, 3)


class TestMissingRequiredColumns(unittest.TestCase):
    """Test 3: CSV without a required column raises clear error."""

    def test_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "bad.csv"
            # Missing 'make' column
            _write_csv(csv_path,
                       ["model", "year_from", "year_to", "engine_code", "fuel_type"],
                       [["Corolla", "2019", "2023", "1ZR-FE", "petrol"]])

            with PartsDatabase(":memory:") as db:
                with self.assertRaises(ValueError) as ctx:
                    import_csv(db, csv_path, "specs")
                self.assertIn("make", str(ctx.exception))


class TestUnknownColumnsWarned(unittest.TestCase):
    """Test 4: Extra columns produce warning but rows still import."""

    def test_unknown_columns_warned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "extra_cols.csv"
            _write_csv(csv_path,
                       ["make", "model", "year_from", "year_to", "engine_code",
                        "fuel_type", "color", "horsepower"],
                       [["Toyota", "Corolla", "2019", "2023", "1ZR-FE",
                         "petrol", "red", "140"]])

            with PartsDatabase(":memory:") as db:
                report = import_csv(db, csv_path, "specs")

                self.assertEqual(report.inserted, 1)
                self.assertEqual(report.errors, 0)
                self.assertIn("color", report.unknown_columns)
                self.assertIn("horsepower", report.unknown_columns)


class TestInvalidTypeSkipped(unittest.TestCase):
    """Test 5: Row with 'abc' in year_from is skipped with error."""

    def test_invalid_type_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "bad_type.csv"
            _write_csv(csv_path,
                       ["make", "model", "year_from", "year_to", "engine_code",
                        "fuel_type"],
                       [["Toyota", "Corolla", "abc", "2023", "1ZR-FE", "petrol"],
                        ["Toyota", "Yaris", "2020", "2024", "1NR-FE", "diesel"]])

            with PartsDatabase(":memory:") as db:
                report = import_csv(db, csv_path, "specs")

                self.assertEqual(report.errors, 1)
                self.assertEqual(report.inserted, 1)
                self.assertIn("Row 2", report.error_details[0])


class TestImportCrossrefCsv(unittest.TestCase):
    """Test 6: Import filter_crossref_sample.csv."""

    def test_import_crossref_csv(self) -> None:
        crossref_csv = DATA_DIR / "filter_crossref_sample.csv"
        self.assertTrue(crossref_csv.exists(), f"{crossref_csv} not found")

        with PartsDatabase(":memory:") as db:
            report = import_csv(db, crossref_csv, "crossref")

            self.assertEqual(report.inserted, 3)
            self.assertEqual(report.errors, 0)

            results = db.find_crossrefs("04152-YZZA1")
            self.assertEqual(len(results), 2)
            brands = {r.brand for r in results}
            self.assertEqual(brands, {"Mann", "Wix"})


class TestDetectCsvTypeSpecs(unittest.TestCase):
    """Test 7: Auto-detection returns 'specs' for specs headers."""

    def test_detect_csv_type_specs(self) -> None:
        specs_csv = DATA_DIR / "oil_specs_sample.csv"
        self.assertEqual(detect_csv_type(specs_csv), "specs")


class TestDetectCsvTypeCrossref(unittest.TestCase):
    """Test 8: Auto-detection returns 'crossref' for crossref headers."""

    def test_detect_csv_type_crossref(self) -> None:
        crossref_csv = DATA_DIR / "filter_crossref_sample.csv"
        self.assertEqual(detect_csv_type(crossref_csv), "crossref")


class TestAllModeImportsDirectory(unittest.TestCase):
    """Test 9: --all mode auto-detects and imports both table types."""

    def test_all_mode_imports_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            db_path = tmp_dir / "test.db"

            # Create a specs CSV
            _write_csv(tmp_dir / "specs.csv",
                       ["make", "model", "year_from", "year_to", "engine_code",
                        "fuel_type", "oil_viscosity"],
                       [["Toyota", "Yaris", "2020", "2024", "1NR-FE",
                         "petrol", "0W-20"]])

            # Create a crossref CSV
            _write_csv(tmp_dir / "crossref.csv",
                       ["oem_part_number", "category", "brand",
                        "brand_part_number", "notes"],
                       [["04152-YZZA1", "oil_filter", "Mann", "W 68/3", ""]])

            rc = main(["--db", str(db_path), "--all", str(tmp_dir)])
            self.assertEqual(rc, 0)

            # Verify both tables were populated
            with PartsDatabase(str(db_path)) as db:
                specs = db.find_specs("Toyota", "Yaris", 2022, "1NR-FE")
                self.assertIsNotNone(specs)

                crossrefs = db.find_crossrefs("04152-YZZA1")
                self.assertEqual(len(crossrefs), 1)


class TestReportCounts(unittest.TestCase):
    """Test 10: Verify inserted/updated/error counts are correct."""

    def test_report_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mixed.csv"
            _write_csv(csv_path,
                       ["make", "model", "year_from", "year_to", "engine_code",
                        "fuel_type"],
                       [["Toyota", "Corolla", "2019", "2023", "1ZR-FE", "petrol"],
                        ["Honda", "Civic", "bad", "2023", "K20C2", "petrol"],
                        ["Hyundai", "i30", "2017", "2023", "G4FJ", "petrol"]])

            with PartsDatabase(":memory:") as db:
                report = import_csv(db, csv_path, "specs")

                self.assertEqual(report.inserted, 2)
                self.assertEqual(report.updated, 0)
                self.assertEqual(report.errors, 1)
                self.assertEqual(len(report.error_details), 1)


if __name__ == "__main__":
    unittest.main()
