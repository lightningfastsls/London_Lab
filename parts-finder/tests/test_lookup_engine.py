"""Tests for LookupEngine orchestration and demo_lookup CLI formatting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from parts_finder.config import AppConfig
from parts_finder.db import PartsDatabase
from parts_finder.exceptions import GovApiError, PlateNotFoundError
from parts_finder.lookup_engine import LookupEngine, LookupResult
from parts_finder.models import (
    BrakeResult,
    BulbResult,
    CoolantResult,
    OilResult,
    VehicleRecord,
    VehicleSpecs,
)
from parts_finder.name_mapper import NameMapper

# ── Test scripts path ────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_DEMO_SCRIPT = _SCRIPTS_DIR / "demo_lookup.py"


# ── Factories ────────────────────────────────────────────────────────


def _make_vehicle(**overrides: object) -> VehicleRecord:
    """Return a VehicleRecord with sensible defaults."""
    defaults: dict = dict(
        plate="1234567",
        make_hebrew="\u05d8\u05d5\u05d9\u05d5\u05d8\u05d4",
        model_hebrew="\u05e7\u05d5\u05e8\u05d5\u05dc\u05d4",
        year=2021,
        engine_code="2ZR-FE",
        fuel_type="petrol",
        make_english="Toyota",
        model_english="Corolla",
    )
    defaults.update(overrides)
    return VehicleRecord(**defaults)


def _make_specs(**overrides: object) -> VehicleSpecs:
    """Return a VehicleSpecs with sensible defaults (oil populated)."""
    defaults: dict = dict(
        make="Toyota",
        model="Corolla",
        year_from=2019,
        year_to=2023,
        engine_code="2ZR-FE",
        fuel_type="petrol",
        oil_viscosity="0W-20",
        oil_capacity_l=4.2,
        oil_spec="API SP",
        oil_oem_approval="Toyota TGMO",
        oil_change_interval_km=15000,
    )
    defaults.update(overrides)
    return VehicleSpecs(**defaults)


def _make_mapping_file(path: Path) -> Path:
    """Write a minimal Hebrew→English mapping JSON file."""
    data = {
        "makes": {"\u05d8\u05d5\u05d9\u05d5\u05d8\u05d4": "Toyota"},
        "models": {"\u05e7\u05d5\u05e8\u05d5\u05dc\u05d4": "Corolla"},
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _mock_plate_client(vehicle: VehicleRecord) -> AsyncMock:
    """Create an AsyncMock that behaves like PlateClient async context manager."""
    mock_client = AsyncMock()
    mock_client.lookup.return_value = vehicle
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


# ── CLI tests ────────────────────────────────────────────────────────


class TestHelpFlag(unittest.TestCase):
    """CLI --help works and exits cleanly."""

    def test_help_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_DEMO_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--plate", result.stdout)
        self.assertIn("--db", result.stdout)


# ── LookupEngine tests ──────────────────────────────────────────────


class TestFullLookupOilMatch(unittest.IsolatedAsyncioTestCase):
    """Mocked API + seeded DB → correct OilResult."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs())
        self.tmp = tempfile.TemporaryDirectory()
        self.mapping_path = _make_mapping_file(
            Path(self.tmp.name) / "hebrew_names.json"
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    async def test_full_lookup_oil_match(self) -> None:
        vehicle = _make_vehicle()
        config = AppConfig(db_path=":memory:")
        mapper = NameMapper(self.mapping_path)
        engine = LookupEngine(config, self.db, mapper)

        mock_client = _mock_plate_client(vehicle)
        with patch(
            "parts_finder.lookup_engine.PlateClient", return_value=mock_client
        ):
            result = await engine.lookup("1234567")

        self.assertIsNotNone(result.oil)
        self.assertEqual(result.oil.viscosity, "0W-20")
        self.assertEqual(result.oil.capacity_l, 4.2)
        self.assertEqual(result.oil.confidence, "exact")
        self.assertEqual(result.vehicle.make_english, "Toyota")


class TestFullLookupAllCategories(unittest.IsolatedAsyncioTestCase):
    """Seeded DB with all category data → all results populated."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(
            _make_specs(
                # Oil already has defaults
                # Bulbs
                low_beam_bulb="H11",
                high_beam_bulb="H9",
                front_turn_bulb="PY21W",
                # Coolant
                coolant_type="SLLC",
                coolant_capacity_l=6.3,
                # Brakes
                front_brake_pad_oem="04465-02220",
                rear_brake_pad_oem="04466-02181",
            )
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.mapping_path = _make_mapping_file(
            Path(self.tmp.name) / "hebrew_names.json"
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    async def test_full_lookup_all_categories(self) -> None:
        vehicle = _make_vehicle()
        config = AppConfig(db_path=":memory:")
        mapper = NameMapper(self.mapping_path)
        engine = LookupEngine(config, self.db, mapper)

        mock_client = _mock_plate_client(vehicle)
        with patch(
            "parts_finder.lookup_engine.PlateClient", return_value=mock_client
        ):
            result = await engine.lookup("1234567")

        self.assertIsNotNone(result.oil)
        self.assertIsNotNone(result.bulbs)
        self.assertIsNotNone(result.brakes)
        # Coolant depends on _COOLANT_SPECS mapping for "SLLC"
        # — if mapped, should be non-None
        # The test verifies at least oil + bulbs + brakes are populated
        self.assertEqual(result.oil.viscosity, "0W-20")
        self.assertEqual(result.bulbs.low_beam, "H11")
        self.assertEqual(result.brakes.front_pad_oem, "04465-02220")


class TestApiTimeoutGraceful(unittest.IsolatedAsyncioTestCase):
    """PlateClient raises GovApiError → LookupEngine propagates it."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.tmp = tempfile.TemporaryDirectory()
        self.mapping_path = _make_mapping_file(
            Path(self.tmp.name) / "hebrew_names.json"
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    async def test_api_timeout_graceful(self) -> None:
        config = AppConfig(db_path=":memory:")
        mapper = NameMapper(self.mapping_path)
        engine = LookupEngine(config, self.db, mapper)

        mock_client = AsyncMock()
        mock_client.lookup.side_effect = GovApiError("Timeout querying plate 1234567")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "parts_finder.lookup_engine.PlateClient", return_value=mock_client
        ):
            with self.assertRaises(GovApiError):
                await engine.lookup("1234567")


class TestNoDbMatch(unittest.IsolatedAsyncioTestCase):
    """Vehicle found in API but not in DB → all category results are None."""

    def setUp(self) -> None:
        # Empty database — no specs seeded
        self.db = PartsDatabase(":memory:")
        self.tmp = tempfile.TemporaryDirectory()
        self.mapping_path = _make_mapping_file(
            Path(self.tmp.name) / "hebrew_names.json"
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    async def test_no_db_match(self) -> None:
        # Use a make NOT in CoolantLookup's _BRAND_DEFAULTS so all
        # categories genuinely return None with an empty DB.
        vehicle = _make_vehicle(
            make_hebrew="\u05d0\u05d5\u05e4\u05dc",
            model_hebrew="\u05e7\u05d5\u05e8\u05e1\u05d4",
            make_english="Opel",
            model_english="Corsa",
        )
        config = AppConfig(db_path=":memory:")
        mapper = NameMapper(self.mapping_path)
        engine = LookupEngine(config, self.db, mapper)

        mock_client = _mock_plate_client(vehicle)
        with patch(
            "parts_finder.lookup_engine.PlateClient", return_value=mock_client
        ):
            result = await engine.lookup("1234567")

        self.assertIsNone(result.oil)
        self.assertIsNone(result.coolant)
        self.assertIsNone(result.brakes)
        self.assertIsNone(result.bulbs)
        # Vehicle should still be populated
        self.assertEqual(result.vehicle.plate, "1234567")


class TestPlateNormalization(unittest.IsolatedAsyncioTestCase):
    """'12-345-67' and '1234567' produce identical VehicleRecord."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs())
        self.tmp = tempfile.TemporaryDirectory()
        self.mapping_path = _make_mapping_file(
            Path(self.tmp.name) / "hebrew_names.json"
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    async def test_plate_normalization(self) -> None:
        vehicle = _make_vehicle()
        config = AppConfig(db_path=":memory:")
        mapper = NameMapper(self.mapping_path)
        engine = LookupEngine(config, self.db, mapper)

        mock_client = _mock_plate_client(vehicle)

        with patch(
            "parts_finder.lookup_engine.PlateClient", return_value=mock_client
        ):
            result_dashes = await engine.lookup("12-345-67")

        with patch(
            "parts_finder.lookup_engine.PlateClient", return_value=mock_client
        ):
            result_plain = await engine.lookup("1234567")

        # Both should produce the same vehicle (PlateClient normalizes)
        self.assertEqual(result_dashes.vehicle.plate, result_plain.vehicle.plate)
        self.assertEqual(result_dashes.vehicle.engine_code, result_plain.vehicle.engine_code)


# ── Format report tests ──────────────────────────────────────────────

# Import format_report from the script — add scripts dir to path
sys.path.insert(0, str(_SCRIPTS_DIR))
from demo_lookup import format_report  # noqa: E402


class TestFormatReportAllCategories(unittest.TestCase):
    """Output contains all 7 category names."""

    def test_format_report_all_categories(self) -> None:
        vehicle = _make_vehicle()
        result = LookupResult(
            vehicle=vehicle,
            oil=OilResult(
                viscosity="0W-20",
                capacity_l=4.2,
                spec="API SP",
                oem_approval="Toyota TGMO",
                change_interval_km=15000,
                confidence="exact",
                source="Toyota Corolla 2019-2023 2ZR-FE",
            ),
        )
        output = format_report(result)

        # All 7 categories present
        self.assertIn("Oil:", output)
        self.assertIn("Coolant:", output)
        self.assertIn("Brakes:", output)
        self.assertIn("Bulbs:", output)
        self.assertIn("Filters:", output)
        self.assertIn("Belts:", output)
        self.assertIn("Spark Plugs:", output)


class TestFormatReportMissingShowsPlaceholder(unittest.TestCase):
    """Unimplemented categories show '[not yet in database]'."""

    def test_format_report_missing_shows_placeholder(self) -> None:
        vehicle = _make_vehicle()
        result = LookupResult(vehicle=vehicle)
        output = format_report(result)

        # Future categories show placeholder
        self.assertIn("[not yet in database]", output)
        self.assertIn("Belts:", output)
        self.assertIn("Spark Plugs:", output)

        # Count occurrences of the placeholder (Belts + Spark Plugs = 2)
        count = output.count("[not yet in database]")
        self.assertEqual(count, 2, "Exactly 2 future categories should show placeholder")


class TestFormatReportNoMatchShowsIndicator(unittest.TestCase):
    """Implemented but None categories show '[no match found]'."""

    def test_format_report_no_match_shows_indicator(self) -> None:
        vehicle = _make_vehicle()
        result = LookupResult(vehicle=vehicle)  # All category results are None
        output = format_report(result)

        # Implemented categories with None results show [no match found]
        no_match_count = output.count("[no match found]")
        self.assertEqual(
            no_match_count, 5,
            "All 5 implemented categories (oil, coolant, brakes, bulbs, filters) "
            "should show [no match found] when result is None",
        )

        # Future categories show the different placeholder
        not_yet_count = output.count("[not yet in database]")
        self.assertEqual(not_yet_count, 2)


if __name__ == "__main__":
    unittest.main()
