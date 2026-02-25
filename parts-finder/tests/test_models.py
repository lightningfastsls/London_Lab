"""Tests for VehicleRecord model."""

from __future__ import annotations

import unittest

from parts_finder.models import VehicleRecord


def _make_api_record(**overrides: object) -> dict:
    """Return a realistic API record dict, with optional overrides."""
    base = {
        "mispar_rechev": "1234567",
        "tozeret_nm": "טויוטה",
        "kinuy_mishari": "קורולה",
        "shnat_yitzur": 2020,
        "degem_manoa": "1ZR-FE",
        "sug_delek_nm": "בנזין",
        "misgeret": "JTDKN3DU5A0123456",
        "ramat_gimur": "GLI",
        "tozeret_eretz_nm": "TOYOTA",
        "degem_nm": "COROLLA",
    }
    base.update(overrides)
    return base


class TestVehicleRecordConstruction(unittest.TestCase):
    """Direct construction and field access."""

    def test_required_fields(self) -> None:
        v = VehicleRecord(
            plate="1234567",
            make_hebrew="טויוטה",
            model_hebrew="קורולה",
            year=2020,
            engine_code="1ZR-FE",
            fuel_type="בנזין",
        )
        self.assertEqual(v.plate, "1234567")
        self.assertEqual(v.year, 2020)
        self.assertEqual(v.engine_code, "1ZR-FE")

    def test_optional_defaults_are_none(self) -> None:
        v = VehicleRecord(
            plate="1234567",
            make_hebrew="טויוטה",
            model_hebrew="קורולה",
            year=2020,
            engine_code="1ZR-FE",
            fuel_type="בנזין",
        )
        self.assertIsNone(v.vin)
        self.assertIsNone(v.trim)
        self.assertIsNone(v.make_english)
        self.assertIsNone(v.model_english)

    def test_frozen_immutability(self) -> None:
        v = VehicleRecord(
            plate="1234567",
            make_hebrew="טויוטה",
            model_hebrew="קורולה",
            year=2020,
            engine_code="1ZR-FE",
            fuel_type="בנזין",
        )
        with self.assertRaises(Exception):
            v.year = 2021  # type: ignore


class TestLookupKey(unittest.TestCase):
    """The lookup_key property for downstream catalog queries."""

    def test_lookup_key_format(self) -> None:
        v = VehicleRecord(
            plate="1234567",
            make_hebrew="טויוטה",
            model_hebrew="קורולה",
            year=2020,
            engine_code="1ZR-FE",
            fuel_type="בנזין",
        )
        self.assertEqual(v.lookup_key, "טויוטה|קורולה|2020|1ZR-FE")

    def test_lookup_key_with_empty_engine_code(self) -> None:
        v = VehicleRecord(
            plate="1234567",
            make_hebrew="טויוטה",
            model_hebrew="קורולה",
            year=2020,
            engine_code="",
            fuel_type="בנזין",
        )
        self.assertEqual(v.lookup_key, "טויוטה|קורולה|2020|")


class TestFromApiRecord(unittest.TestCase):
    """The from_api_record classmethod factory."""

    def test_happy_path(self) -> None:
        rec = _make_api_record()
        v = VehicleRecord.from_api_record("1234567", rec)
        self.assertEqual(v.plate, "1234567")
        self.assertEqual(v.make_hebrew, "טויוטה")
        self.assertEqual(v.model_hebrew, "קורולה")
        self.assertEqual(v.year, 2020)
        self.assertEqual(v.engine_code, "1ZR-FE")
        self.assertEqual(v.fuel_type, "בנזין")
        self.assertEqual(v.vin, "JTDKN3DU5A0123456")
        self.assertEqual(v.trim, "GLI")
        self.assertEqual(v.make_english, "TOYOTA")
        self.assertEqual(v.model_english, "COROLLA")

    def test_missing_optional_fields(self) -> None:
        rec = _make_api_record()
        del rec["misgeret"]
        del rec["ramat_gimur"]
        del rec["tozeret_eretz_nm"]
        del rec["degem_nm"]
        v = VehicleRecord.from_api_record("1234567", rec)
        self.assertIsNone(v.vin)
        self.assertIsNone(v.trim)
        self.assertIsNone(v.make_english)
        self.assertIsNone(v.model_english)

    def test_empty_string_collapses_to_none(self) -> None:
        rec = _make_api_record(misgeret="", ramat_gimur="  ")
        v = VehicleRecord.from_api_record("1234567", rec)
        self.assertIsNone(v.vin)
        self.assertIsNone(v.trim)

    def test_missing_engine_code_defaults_empty(self) -> None:
        rec = _make_api_record()
        del rec["degem_manoa"]
        v = VehicleRecord.from_api_record("1234567", rec)
        self.assertEqual(v.engine_code, "")

    def test_whitespace_stripping(self) -> None:
        rec = _make_api_record(
            tozeret_nm="  טויוטה  ",
            kinuy_mishari=" קורולה\t",
            degem_manoa="  1ZR-FE ",
        )
        v = VehicleRecord.from_api_record(" 1234567 ", rec)
        self.assertEqual(v.plate, "1234567")
        self.assertEqual(v.make_hebrew, "טויוטה")
        self.assertEqual(v.model_hebrew, "קורולה")
        self.assertEqual(v.engine_code, "1ZR-FE")

    def test_missing_required_field_raises(self) -> None:
        rec = _make_api_record()
        del rec["tozeret_nm"]
        with self.assertRaises(KeyError):
            VehicleRecord.from_api_record("1234567", rec)


if __name__ == "__main__":
    unittest.main()
