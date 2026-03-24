"""Tests for build_response() with AI fallback merge logic.

Covers the ai_result parameter path, tri-state data_source computation,
and the guarantee that AI data never overwrites existing DB matches.
"""

from __future__ import annotations

import unittest

from parts_finder.api.fallback import AIFallbackResult
from parts_finder.api.response_builder import build_response
from parts_finder.api.schemas import (
    BrakesResponse,
    CoolantResponse,
    FilterCategoryResponse,
    OilResponse,
)
from parts_finder.lookup_engine import LookupResult
from parts_finder.models import (
    BrakeResult,
    CoolantResult,
    FilterResult,
    OilResult,
    VehicleRecord,
)


def _make_vehicle() -> VehicleRecord:
    return VehicleRecord(
        plate="1234567",
        make_hebrew="טויוטה",
        model_hebrew="קורולה",
        year=2021,
        engine_code="2ZR-FE",
        fuel_type="petrol",
        make_english="Toyota",
        model_english="Corolla",
    )


def _make_oil_result() -> OilResult:
    return OilResult(
        viscosity="0W-20",
        capacity_l=4.2,
        spec="API SP",
        oem_approval="Toyota TGMO",
        change_interval_km=15000,
        confidence="exact",
        source="Toyota Corolla 2019-2023 2ZR-FE",
    )


def _make_brake_result() -> BrakeResult:
    return BrakeResult(
        front_pad_oem="04465-02220",
        rear_pad_oem="04466-02181",
        front_disc_oem="43512-02330",
        rear_disc_oem="42431-02250",
        brake_fluid_type="DOT 4",
    )


def _make_ai_oil() -> OilResponse:
    return OilResponse(
        viscosity="0W-20",
        capacity_l=4.0,
        spec="API SP",
        oem_approval="",
        change_interval_km=15000,
        confidence="ai_fallback",
        source="ai_fallback",
    )


def _make_ai_brakes() -> BrakesResponse:
    return BrakesResponse(
        front_pad_oem="04465-AI",
        rear_pad_oem="04466-AI",
        brake_fluid_type="DOT 4",
    )


def _make_ai_coolant() -> CoolantResponse:
    return CoolantResponse(
        spec="Toyota SLLC",
        technology="OAT",
        color="pink",
        capacity_l=6.4,
        aftermarket_match="GLYSANTIN G30",
        mixing_warning="Do NOT mix with IAT coolant",
    )


class TestBuildResponseWithAI(unittest.TestCase):
    """Tests for build_response() when ai_result is provided."""

    def test_db_only_data_source_database(self) -> None:
        """No AI result → data_source is 'database'."""
        result = LookupResult(
            vehicle=_make_vehicle(),
            oil=_make_oil_result(),
        )
        response = build_response(result)

        self.assertEqual(response.data_source, "database")

    def test_ai_only_data_source_ai_fallback(self) -> None:
        """DB matched nothing, AI fills some → data_source is 'ai_fallback'."""
        result = LookupResult(vehicle=_make_vehicle())
        ai_result = AIFallbackResult(
            oil=_make_ai_oil(),
            coolant=_make_ai_coolant(),
        )
        response = build_response(result, ai_result=ai_result)

        self.assertEqual(response.data_source, "ai_fallback")
        self.assertIsNotNone(response.categories.oil)
        self.assertIsNotNone(response.categories.coolant)

    def test_mixed_data_source_hybrid(self) -> None:
        """DB matched some, AI filled remaining → data_source is 'hybrid'."""
        result = LookupResult(
            vehicle=_make_vehicle(),
            oil=_make_oil_result(),  # DB has oil
        )
        ai_result = AIFallbackResult(
            coolant=_make_ai_coolant(),  # AI fills coolant
        )
        response = build_response(result, ai_result=ai_result)

        self.assertEqual(response.data_source, "hybrid")
        # Oil is from DB
        self.assertIsNotNone(response.categories.oil)
        self.assertEqual(response.categories.oil.confidence, "exact")
        # Coolant is from AI
        self.assertIsNotNone(response.categories.coolant)

    def test_ai_does_not_overwrite_db_match(self) -> None:
        """DB has oil, AI also returns oil → merged oil is the DB version."""
        db_oil = _make_oil_result()
        result = LookupResult(
            vehicle=_make_vehicle(),
            oil=db_oil,
        )
        ai_result = AIFallbackResult(
            oil=_make_ai_oil(),  # AI also has oil — should be ignored
            coolant=_make_ai_coolant(),
        )
        response = build_response(result, ai_result=ai_result)

        # Oil should be the DB version (confidence="exact"), not AI
        self.assertEqual(response.categories.oil.confidence, "exact")
        self.assertEqual(
            response.categories.oil.source,
            "Toyota Corolla 2019-2023 2ZR-FE",
        )
        self.assertEqual(response.data_source, "hybrid")

    def test_unmatched_categories_shrinks_after_ai_fill(self) -> None:
        """AI filling gaps should reduce the unmatched_categories list."""
        result = LookupResult(vehicle=_make_vehicle())
        # DB has nothing — all 7 unmatched
        response_no_ai = build_response(result)
        self.assertEqual(len(response_no_ai.unmatched_categories), 7)

        # AI fills oil and coolant — 5 remain unmatched
        ai_result = AIFallbackResult(
            oil=_make_ai_oil(),
            coolant=_make_ai_coolant(),
        )
        response_with_ai = build_response(result, ai_result=ai_result)
        self.assertEqual(len(response_with_ai.unmatched_categories), 5)
        self.assertNotIn("oil", response_with_ai.unmatched_categories)
        self.assertNotIn("coolant", response_with_ai.unmatched_categories)

    def test_coverage_string_reflects_ai_contribution(self) -> None:
        """Coverage counts AI-filled categories as matched."""
        result = LookupResult(
            vehicle=_make_vehicle(),
            oil=_make_oil_result(),  # 1 from DB
        )
        ai_result = AIFallbackResult(
            brakes=_make_ai_brakes(),
            coolant=_make_ai_coolant(),
        )
        response = build_response(result, ai_result=ai_result)

        # 1 DB + 2 AI = 3 matched
        self.assertEqual(response.coverage, "3/7 categories matched")

    def test_ai_result_with_no_useful_data(self) -> None:
        """AI result exists but all fields are None → data_source stays 'database'."""
        result = LookupResult(
            vehicle=_make_vehicle(),
            oil=_make_oil_result(),
        )
        ai_result = AIFallbackResult()  # empty — nothing to merge
        response = build_response(result, ai_result=ai_result)

        self.assertEqual(response.data_source, "database")


if __name__ == "__main__":
    unittest.main()
