"""Tests for the Hebrew-to-English NameMapper."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from parts_finder.models import VehicleRecord
from parts_finder.name_mapper import NameMapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_MAPPING = {
    "makes": {
        "טויוטה": "Toyota",
        "יונדאי": "Hyundai",
    },
    "models": {
        "קורולה": "Corolla",
        "טוסון": "Tucson",
    },
}


def _write_mapping(tmp_dir: Path, mapping: dict | None = None) -> Path:
    """Write a mapping JSON file into *tmp_dir* and return its path."""
    path = tmp_dir / "test_names.json"
    path.write_text(
        json.dumps(mapping or _SAMPLE_MAPPING, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _make_vehicle(**overrides: object) -> VehicleRecord:
    """Return a minimal VehicleRecord with optional field overrides."""
    defaults: dict = dict(
        plate="1234567",
        make_hebrew="טויוטה",
        model_hebrew="קורולה",
        year=2020,
        engine_code="1ZR-FE",
        fuel_type="בנזין",
    )
    defaults.update(overrides)
    return VehicleRecord(**defaults)


# ---------------------------------------------------------------------------
# Tests — translate_make
# ---------------------------------------------------------------------------


class TestTranslateMake(unittest.TestCase):
    """NameMapper.translate_make behaviour."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.mapper = NameMapper(_write_mapping(self.tmp_dir))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_known_make_returns_english(self) -> None:
        self.assertEqual(self.mapper.translate_make("טויוטה"), "Toyota")

    def test_unknown_make_returns_none_and_warns(self) -> None:
        with self.assertLogs("parts_finder.name_mapper", level=logging.WARNING) as cm:
            result = self.mapper.translate_make("ג'נסיס")
        self.assertIsNone(result)
        self.assertTrue(any("Unmapped make" in msg for msg in cm.output))

    def test_whitespace_stripped_before_lookup(self) -> None:
        self.assertEqual(self.mapper.translate_make("  טויוטה  "), "Toyota")

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(self.mapper.translate_make(""))

    def test_whitespace_only_returns_none(self) -> None:
        self.assertIsNone(self.mapper.translate_make("   "))


# ---------------------------------------------------------------------------
# Tests — translate_model
# ---------------------------------------------------------------------------


class TestTranslateModel(unittest.TestCase):
    """NameMapper.translate_model behaviour."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.mapper = NameMapper(_write_mapping(self.tmp_dir))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_known_model_returns_english(self) -> None:
        self.assertEqual(self.mapper.translate_model("קורולה"), "Corolla")

    def test_unknown_model_returns_none_and_warns(self) -> None:
        with self.assertLogs("parts_finder.name_mapper", level=logging.WARNING) as cm:
            result = self.mapper.translate_model("קמרי")
        self.assertIsNone(result)
        self.assertTrue(any("Unmapped model" in msg for msg in cm.output))

    def test_whitespace_stripped_before_lookup(self) -> None:
        self.assertEqual(self.mapper.translate_model("\tקורולה\n"), "Corolla")

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(self.mapper.translate_model(""))


# ---------------------------------------------------------------------------
# Tests — enrich_vehicle
# ---------------------------------------------------------------------------


class TestEnrichVehicle(unittest.TestCase):
    """NameMapper.enrich_vehicle behaviour."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.mapper = NameMapper(_write_mapping(self.tmp_dir))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_full_enrichment(self) -> None:
        vehicle = _make_vehicle()
        enriched = self.mapper.enrich_vehicle(vehicle)
        self.assertEqual(enriched.make_english, "Toyota")
        self.assertEqual(enriched.model_english, "Corolla")
        # Original record is untouched (frozen dataclass)
        self.assertIsNone(vehicle.make_english)

    def test_partial_enrichment_only_make(self) -> None:
        vehicle = _make_vehicle(model_hebrew="לא קיים")
        with self.assertLogs("parts_finder.name_mapper", level=logging.WARNING):
            enriched = self.mapper.enrich_vehicle(vehicle)
        self.assertEqual(enriched.make_english, "Toyota")
        self.assertIsNone(enriched.model_english)

    def test_partial_enrichment_only_model(self) -> None:
        vehicle = _make_vehicle(make_hebrew="לא קיים")
        with self.assertLogs("parts_finder.name_mapper", level=logging.WARNING):
            enriched = self.mapper.enrich_vehicle(vehicle)
        self.assertIsNone(enriched.make_english)
        self.assertEqual(enriched.model_english, "Corolla")

    def test_preserves_api_english_when_mapping_missing(self) -> None:
        vehicle = _make_vehicle(
            make_hebrew="ג'נסיס",
            model_hebrew="G80",
            make_english="GENESIS",
            model_english="G80",
        )
        with self.assertLogs("parts_finder.name_mapper", level=logging.WARNING):
            enriched = self.mapper.enrich_vehicle(vehicle)
        # No mapping for Genesis/G80 — API values preserved
        self.assertEqual(enriched.make_english, "GENESIS")
        self.assertEqual(enriched.model_english, "G80")


class TestNameMapperConstruction(unittest.TestCase):
    """NameMapper construction edge cases."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_key_in_json_raises_value_error(self) -> None:
        bad_mapping = {"makes": {"טויוטה": "Toyota"}}  # missing "models"
        with self.assertRaises(ValueError) as ctx:
            NameMapper(_write_mapping(self.tmp_dir, bad_mapping))
        self.assertIn("models", str(ctx.exception))
        self.assertIn("missing required key", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
