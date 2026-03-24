"""Tests for coolant specification lookup (three-tier cascade)."""

from __future__ import annotations

import unittest

from parts_finder.db import PartsDatabase
from parts_finder.models import CoolantResult, VehicleRecord, VehicleSpecs
from parts_finder.lookup.coolant import (
    CoolantLookup,
    _build_mixing_warning,
)


def _make_specs(**overrides: object) -> VehicleSpecs:
    """Return a VehicleSpecs with sensible defaults, with optional overrides."""
    defaults: dict = dict(
        make="Volkswagen",
        model="Golf",
        year_from=2019,
        year_to=2023,
        engine_code="DPCA",
        fuel_type="petrol",
        coolant_type="G13",
        coolant_capacity_l=5.6,
    )
    defaults.update(overrides)
    return VehicleSpecs(**defaults)


def _make_vehicle(**overrides: object) -> VehicleRecord:
    """Return a VehicleRecord with sensible defaults."""
    defaults: dict = dict(
        plate="1234567",
        make_hebrew="פולקסווגן",
        model_hebrew="גולף",
        year=2021,
        engine_code="DPCA",
        fuel_type="petrol",
        make_english="Volkswagen",
        model_english="Golf",
    )
    defaults.update(overrides)
    return VehicleRecord(**defaults)


# ── CoolantResult construction ──────────────────────────────────────


class TestCoolantResultConstruction(unittest.TestCase):
    """CoolantResult dataclass and factory methods."""

    def test_from_vehicle_specs(self) -> None:
        specs = _make_specs(coolant_capacity_l=5.6)
        spec_info = ("TL 774 J (G13)", "Si-OAT", "purple", "GLYSANTIN G40/G30")
        result = CoolantResult.from_vehicle_specs(
            specs, spec_info=spec_info,
            mixing_warning="Do NOT mix with IAT or P-OAT coolant",
            source="exact",
        )

        self.assertEqual(result.spec, "TL 774 J (G13)")
        self.assertEqual(result.technology, "Si-OAT")
        self.assertEqual(result.color, "purple")
        self.assertAlmostEqual(result.capacity_l, 5.6)
        self.assertEqual(result.aftermarket_match, "GLYSANTIN G40/G30")
        self.assertEqual(result.mixing_warning, "Do NOT mix with IAT or P-OAT coolant")
        self.assertEqual(result.source, "exact")

    def test_from_brand_default_has_zero_capacity(self) -> None:
        spec_info = ("TL 774 J (G13)", "Si-OAT", "purple", "GLYSANTIN G40/G30")
        result = CoolantResult.from_brand_default(
            make="Volkswagen", spec_info=spec_info,
            mixing_warning="Do NOT mix with IAT or P-OAT coolant",
        )

        self.assertEqual(result.spec, "TL 774 J (G13)")
        self.assertAlmostEqual(result.capacity_l, 0.0)
        self.assertEqual(result.source, "brand_default")

    def test_frozen(self) -> None:
        spec_info = ("TL 774 J (G13)", "Si-OAT", "purple", "GLYSANTIN G40/G30")
        result = CoolantResult.from_brand_default(
            make="Volkswagen", spec_info=spec_info,
            mixing_warning="test",
        )
        with self.assertRaises(AttributeError):
            result.spec = "changed"  # type: ignore[misc]


# ── Mixing warnings ─────────────────────────────────────────────────


class TestMixingWarning(unittest.TestCase):
    """Compatibility matrix and mixing warning generation."""

    def test_oat_warning(self) -> None:
        warning = _build_mixing_warning("OAT")
        self.assertIn("Do NOT mix", warning)
        self.assertIn("HOAT", warning)
        self.assertIn("IAT", warning)
        self.assertIn("P-OAT", warning)
        # Compatibility is asymmetric: Si-OAT can mix with OAT (it's
        # OAT + silicate), but OAT warns against Si-OAT because adding
        # silicate additives to a pure OAT system may not be desired.
        self.assertIn("Si-OAT", warning)

    def test_iat_warning(self) -> None:
        warning = _build_mixing_warning("IAT")
        self.assertIn("Do NOT mix", warning)
        # IAT is only compatible with IAT (and HOAT contains IAT inhibitors,
        # but that doesn't make IAT compatible with HOAT from IAT's perspective)
        self.assertIn("OAT", warning)
        self.assertIn("P-OAT", warning)
        self.assertIn("Si-OAT", warning)

    def test_hoat_partially_compatible(self) -> None:
        warning = _build_mixing_warning("HOAT")
        # HOAT is compatible with HOAT and IAT
        self.assertIn("Do NOT mix", warning)
        self.assertIn("OAT", warning)
        self.assertNotIn("IAT", warning)

    def test_unknown_technology(self) -> None:
        warning = _build_mixing_warning("UNKNOWN")
        self.assertIn("Unknown coolant technology", warning)
        self.assertIn("verify compatibility", warning)


# ── Tier 1: exact match ─────────────────────────────────────────────


class TestCoolantLookupExactMatch(unittest.TestCase):
    """Tier 1 — exact match on make + model + year + engine_code."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            coolant_type="G13",
            coolant_capacity_l=5.6,
        ))
        self.coolant = CoolantLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_exact_match_returns_result(self) -> None:
        vehicle = _make_vehicle()
        result = self.coolant.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.spec, "TL 774 J (G13)")
        self.assertEqual(result.technology, "Si-OAT")
        self.assertEqual(result.color, "purple")

    def test_exact_match_source(self) -> None:
        vehicle = _make_vehicle()
        result = self.coolant.lookup(vehicle)
        self.assertEqual(result.source, "exact")

    def test_no_match_returns_none(self) -> None:
        vehicle = _make_vehicle(
            make_english="Subaru", model_english="Outback",
            make_hebrew="סובארו", model_hebrew="אאוטבק",
            engine_code="FB25",
        )
        result = self.coolant.lookup(vehicle)
        # Subaru not in brand defaults either
        self.assertIsNone(result)


# ── Tier 2: model-year fallback ──────────────────────────────────────


class TestCoolantLookupModelYearFallback(unittest.TestCase):
    """Tier 2 — model-year match (different engine, same chassis)."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        # DB has Golf with DPCA engine — lookup for CZCA should fallback
        self.db.insert_specs(_make_specs(
            engine_code="DPCA",
            coolant_type="G13",
            coolant_capacity_l=5.6,
        ))
        self.coolant = CoolantLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_different_engine_same_model_matches(self) -> None:
        """Golf CZCA not in DB, but DPCA Golf has coolant data."""
        vehicle = _make_vehicle(engine_code="CZCA")
        result = self.coolant.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.spec, "TL 774 J (G13)")

    def test_model_year_source(self) -> None:
        vehicle = _make_vehicle(engine_code="CZCA")
        result = self.coolant.lookup(vehicle)
        self.assertEqual(result.source, "model_year")


# ── Tier 3: brand default ───────────────────────────────────────────


class TestCoolantLookupBrandDefault(unittest.TestCase):
    """Tier 3 — brand-level default from hardcoded knowledge base."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        # Empty DB — will force brand default tier
        self.coolant = CoolantLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_toyota_brand_default(self) -> None:
        vehicle = _make_vehicle(
            make_english="Toyota", model_english="Corolla",
            make_hebrew="טויוטה", model_hebrew="קורולה",
            engine_code="2ZR-FE",
        )
        result = self.coolant.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.spec, "Super Long Life Coolant")
        self.assertEqual(result.technology, "OAT")
        self.assertEqual(result.color, "pink")
        self.assertAlmostEqual(result.capacity_l, 0.0)
        self.assertEqual(result.source, "brand_default")

    def test_unknown_brand_returns_none(self) -> None:
        vehicle = _make_vehicle(
            make_english="Subaru", model_english="Outback",
            make_hebrew="סובארו", model_hebrew="אאוטבק",
            engine_code="FB25",
        )
        result = self.coolant.lookup(vehicle)
        self.assertIsNone(result)


# ── Cascade integration ──────────────────────────────────────────────


class TestCoolantLookupCascade(unittest.TestCase):
    """Integration: cascade short-circuits at the right tier."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            engine_code="DPCA",
            coolant_type="G13",
            coolant_capacity_l=5.6,
        ))
        self.coolant = CoolantLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_exact_match_short_circuits(self) -> None:
        """When exact match hits, cascade stops at tier 1."""
        vehicle = _make_vehicle()
        result = self.coolant.lookup(vehicle)
        self.assertEqual(result.source, "exact")

    def test_full_cascade_to_brand_default(self) -> None:
        """Unknown model + engine → brand default."""
        vehicle = _make_vehicle(
            model_english="Arteon", model_hebrew="ארטיאון",
            engine_code="CZPB",
        )
        result = self.coolant.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "brand_default")

    def test_no_make_returns_none(self) -> None:
        """Vehicle with no usable make name returns None."""
        vehicle = _make_vehicle(
            make_english=None, make_hebrew="",
        )
        result = self.coolant.lookup(vehicle)
        self.assertIsNone(result)

    def test_empty_coolant_type_skips_to_next_tier(self) -> None:
        """VehicleSpecs with empty coolant_type is treated as no match."""
        self.db.insert_specs(_make_specs(
            model="Polo", engine_code="CZEA",
            year_from=2019, year_to=2023,
            coolant_type="",  # no coolant data
        ))
        vehicle = _make_vehicle(
            model_english="Polo", model_hebrew="פולו",
            engine_code="CZEA",
        )
        result = self.coolant.lookup(vehicle)
        # Tier 1: exact match has empty coolant_type -> skipped
        # Tier 2: model-year also finds Polo with empty coolant_type -> skipped
        # Tier 3: Volkswagen brand default -> G13
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "brand_default")
        self.assertEqual(result.spec, "TL 774 J (G13)")

    def test_empty_engine_skips_tier1(self) -> None:
        """Empty engine_code skips tier 1, tries tier 2."""
        vehicle = _make_vehicle(engine_code="")
        result = self.coolant.lookup(vehicle)
        self.assertIsNotNone(result)
        # Should hit tier 2 (model_year) since Golf with coolant data exists
        self.assertEqual(result.source, "model_year")

    def test_mixing_warning_present(self) -> None:
        """Every result includes a mixing warning."""
        vehicle = _make_vehicle()
        result = self.coolant.lookup(vehicle)
        self.assertIn("Do NOT mix", result.mixing_warning)


if __name__ == "__main__":
    unittest.main()
