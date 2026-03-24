"""Tests for the FastAPI plate-lookup endpoint.

Uses httpx's TestClient (via fastapi.testclient) with a real in-memory
database and mocked PlateClient to test the full API flow without
hitting the government API.

Test plan from ROADMAP:
1. Full flow — all 7 categories returned
2. Partial coverage — unmatched_categories in response
3. Invalid plate — 400 error
4. Plate not in registry — 404 error
5. data.gov.il unavailable — 503 error
6. Cached plate — PlateClient NOT called again
7. Response JSON matches schema
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from parts_finder.api.routes import router
from parts_finder.config import AppConfig
from parts_finder.db import PartsDatabase
from parts_finder.exceptions import GovApiError, PlateNotFoundError
from parts_finder.lookup_engine import LookupEngine
from parts_finder.models import ProductCrossRef, VehicleRecord, VehicleSpecs
from parts_finder.name_mapper import NameMapper


def _make_specs(**overrides: object) -> VehicleSpecs:
    """Return a VehicleSpecs with sensible defaults."""
    defaults: dict = dict(
        make="Toyota",
        model="Corolla",
        year_from=2019,
        year_to=2023,
        engine_code="2ZR-FE",
        fuel_type="petrol",
    )
    defaults.update(overrides)
    return VehicleSpecs(**defaults)


def _make_vehicle(**overrides: object) -> VehicleRecord:
    """Return a VehicleRecord with sensible defaults."""
    defaults: dict = dict(
        plate="1234567",
        make_hebrew="טויוטה",
        model_hebrew="קורולה",
        year=2021,
        engine_code="2ZR-FE",
        fuel_type="petrol",
        make_english="Toyota",
        model_english="Corolla",
    )
    defaults.update(overrides)
    return VehicleRecord(**defaults)


# Name mapping JSON for NameMapper
_NAME_MAPPING = {
    "makes": {"טויוטה": "Toyota"},
    "models": {"קורולה": "Corolla"},
}


class _APITestBase(unittest.TestCase):
    """Base class that sets up an in-memory DB, name mapping, and test client.

    Rather than using lifespan (which requires TestClient as context
    manager), we set ``app.state`` directly — simpler and sufficient
    for unit tests.
    """

    def setUp(self) -> None:
        # Write name mapping to temp file
        self._mapping_file = NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        json.dump(_NAME_MAPPING, self._mapping_file)
        self._mapping_file.close()
        self._mapping_path = Path(self._mapping_file.name)

        # Create in-memory database.
        # check_same_thread=False needed because TestClient runs the
        # ASGI app in a separate thread from where the DB was created.
        self._db = PartsDatabase(":memory:", check_same_thread=False)

        # Config
        self._config = AppConfig(
            db_path=":memory:",
            cache_enabled=True,
            cache_ttl_minutes=60,
        )

        # Default vehicle for PlateClient mock
        self._vehicle = _make_vehicle()

    def tearDown(self) -> None:
        self._db.close()
        if hasattr(self, "_patcher"):
            self._patcher.stop()
        try:
            self._mapping_path.unlink()
        except OSError:
            pass

    def _get_client(
        self,
        plate_result: VehicleRecord | None = None,
        plate_side_effect: Exception | None = None,
    ) -> TestClient:
        """Create a TestClient with mocked PlateClient.

        Builds a minimal FastAPI app, injects test objects onto
        ``app.state``, and patches PlateClient so we never hit
        the real government API.
        """
        # Patch PlateClient BEFORE building the engine
        mock_lookup = AsyncMock()
        if plate_side_effect:
            mock_lookup.side_effect = plate_side_effect
        else:
            mock_lookup.return_value = plate_result or self._vehicle

        patcher = patch("parts_finder.lookup_engine.PlateClient")
        mock_client_cls = patcher.start()

        # PlateClient is used as async context manager
        mock_instance = AsyncMock()
        mock_instance.lookup = mock_lookup
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance

        self._patcher = patcher
        self._mock_lookup = mock_lookup

        # Build app and set state directly (no lifespan needed)
        mapper = NameMapper(self._mapping_path)
        engine = LookupEngine(self._config, self._db, mapper)

        app = FastAPI()
        app.include_router(router)
        app.state.config = self._config
        app.state.db = self._db
        app.state.mapper = mapper
        app.state.engine = engine
        app.state.cache = {}

        return TestClient(app)


# ── Test 1: Full flow — all 7 categories ─────────────────────────


class TestFullFlowAllCategories(_APITestBase):
    """Full flow with mocked API + populated DB — all 7 categories returned."""

    def setUp(self) -> None:
        super().setUp()
        self._db.insert_specs(_make_specs(
            # Oil
            oil_viscosity="0W-20",
            oil_capacity_l=4.2,
            oil_spec="API SP",
            oil_oem_approval="Toyota TGMO",
            oil_change_interval_km=15000,
            # Filters
            oil_filter_oem="04152-YZZA1",
            air_filter_oem="17801-21050",
            cabin_filter_oem="87139-02020",
            # Brakes
            front_brake_pad_oem="04465-02220",
            rear_brake_pad_oem="04466-02181",
            front_rotor_oem="43512-02330",
            rear_rotor_oem="42431-02250",
            brake_fluid_type="DOT 4",
            # Bulbs
            low_beam_bulb="H11",
            high_beam_bulb="HB3",
            front_turn_bulb="WY21W",
            rear_turn_bulb="WY21W",
            tail_brake_bulb="W21/5W",
            reverse_bulb="W16W",
            fog_bulb="H16",
            license_plate_bulb="W5W",
            # Coolant — key must match _COOLANT_SPECS (e.g. "SLLC", not "Toyota SLLC")
            coolant_type="SLLC",
            coolant_capacity_l=6.4,
        ))

    def test_all_categories_returned(self) -> None:
        client = self._get_client()
        resp = client.post("/api/plate-lookup", json={"plate": "1234567"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Vehicle
        self.assertEqual(data["vehicle"]["plate"], "1234567")
        self.assertEqual(data["vehicle"]["make"], "Toyota")
        self.assertEqual(data["vehicle"]["model"], "Corolla")
        self.assertEqual(data["vehicle"]["year"], 2021)

        # Categories
        cats = data["categories"]
        self.assertIsNotNone(cats["oil"])
        self.assertEqual(cats["oil"]["viscosity"], "0W-20")
        self.assertIsNotNone(cats["oil_filter"])
        self.assertEqual(cats["oil_filter"]["oem_part_number"], "04152-YZZA1")
        self.assertIsNotNone(cats["air_filter"])
        self.assertEqual(cats["air_filter"]["oem_part_number"], "17801-21050")
        self.assertIsNotNone(cats["cabin_filter"])
        self.assertIsNotNone(cats["brakes"])
        self.assertEqual(cats["brakes"]["front_pad_oem"], "04465-02220")
        self.assertIsNotNone(cats["bulbs"])
        self.assertEqual(cats["bulbs"]["low_beam"], "H11")
        self.assertIsNotNone(cats["coolant"])
        # Verify coolant came from DB (capacity=6.4), not brand-default (capacity=0.0)
        self.assertEqual(cats["coolant"]["capacity_l"], 6.4)

        # Coverage
        self.assertEqual(data["coverage"], "7/7 categories matched")
        self.assertEqual(data["unmatched_categories"], [])
        self.assertEqual(data["data_source"], "database")


# ── Test 2: Partial coverage ─────────────────────────────────────


class TestPartialCoverage(_APITestBase):
    """Partial coverage — unmatched_categories listed."""

    def setUp(self) -> None:
        super().setUp()
        # Only insert oil and bulb data
        self._db.insert_specs(_make_specs(
            oil_viscosity="0W-20",
            oil_capacity_l=4.2,
            oil_spec="API SP",
            oil_change_interval_km=15000,
            low_beam_bulb="H11",
            high_beam_bulb="HB3",
        ))

    def test_partial_coverage_shows_unmatched(self) -> None:
        client = self._get_client()
        resp = client.post("/api/plate-lookup", json={"plate": "1234567"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Oil and bulbs have data in DB
        self.assertIsNotNone(data["categories"]["oil"])
        self.assertIsNotNone(data["categories"]["bulbs"])
        # Brakes has no brand-default tier — truly unmatched
        self.assertIsNone(data["categories"]["brakes"])
        # Note: coolant returns a brand-default for Toyota (3-tier cascade)
        # so it may be non-None even without explicit coolant data.
        # Similarly, oil has a brand-default tier.

        # At minimum, filters and brakes should be unmatched
        self.assertIn("brakes", data["unmatched_categories"])
        self.assertIn("oil_filter", data["unmatched_categories"])
        self.assertIn("air_filter", data["unmatched_categories"])


# ── Test 3: Invalid plate format — 400 ───────────────────────────


class TestInvalidPlate(_APITestBase):
    """Invalid plate format returns 400 error."""

    def test_invalid_plate_returns_400(self) -> None:
        client = self._get_client(
            plate_side_effect=ValueError("Invalid plate (no digits): 'abc'"),
        )
        resp = client.post("/api/plate-lookup", json={"plate": "abc"})
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn("detail", data)

    def test_empty_plate_rejected_by_pydantic(self) -> None:
        """Pydantic rejects empty plate before hitting the route."""
        client = self._get_client()
        resp = client.post("/api/plate-lookup", json={"plate": ""})
        self.assertEqual(resp.status_code, 422)  # Pydantic validation error


# ── Test 4: Plate not found — 404 ────────────────────────────────


class TestPlateNotFound(_APITestBase):
    """Plate not in registry returns 404 error."""

    def test_plate_not_found_returns_404(self) -> None:
        client = self._get_client(
            plate_side_effect=PlateNotFoundError("9999999"),
        )
        resp = client.post("/api/plate-lookup", json={"plate": "9999999"})
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertIn("Plate not found", data["detail"])


# ── Test 5: Government API unavailable — 503 ─────────────────────


class TestGovApiUnavailable(_APITestBase):
    """data.gov.il unavailable returns 503 error."""

    def test_gov_api_error_returns_503(self) -> None:
        client = self._get_client(
            plate_side_effect=GovApiError("Connection refused"),
        )
        resp = client.post("/api/plate-lookup", json={"plate": "1234567"})
        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertIn("Government API unavailable", data["detail"])


# ── Test 6: Cached plate ─────────────────────────────────────────


class TestCaching(_APITestBase):
    """Cached plate — second request does not call PlateClient again."""

    def setUp(self) -> None:
        super().setUp()
        self._db.insert_specs(_make_specs(
            oil_viscosity="0W-20",
            oil_capacity_l=4.2,
            oil_spec="API SP",
            oil_change_interval_km=15000,
        ))

    def test_second_request_uses_cache(self) -> None:
        client = self._get_client()
        # First request — calls PlateClient
        resp1 = client.post("/api/plate-lookup", json={"plate": "1234567"})
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(self._mock_lookup.call_count, 1)

        # Second request — should use cache
        resp2 = client.post("/api/plate-lookup", json={"plate": "1234567"})
        self.assertEqual(resp2.status_code, 200)
        # PlateClient should NOT have been called again
        self.assertEqual(self._mock_lookup.call_count, 1)

        # Responses should be identical
        self.assertEqual(resp1.json(), resp2.json())

    def test_expired_cache_triggers_refetch(self) -> None:
        """After TTL expires, PlateClient is called again."""
        client = self._get_client()

        # First request — populates cache
        resp1 = client.post("/api/plate-lookup", json={"plate": "1234567"})
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(self._mock_lookup.call_count, 1)

        # Simulate TTL expiry by backdating the cache timestamp.
        # The route stores (response, monotonic_time) tuples.
        cache = client.app.state.cache
        plate = "1234567"
        self.assertIn(plate, cache)
        old_response, _old_ts = cache[plate]
        # Set timestamp to 2 hours ago (well past the 60-minute TTL)
        import time
        cache[plate] = (old_response, time.monotonic() - 7200)

        # Second request — cache expired, should re-fetch
        resp2 = client.post("/api/plate-lookup", json={"plate": "1234567"})
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(self._mock_lookup.call_count, 2)


# ── Test 7: Response schema ──────────────────────────────────────


class TestResponseSchema(_APITestBase):
    """Response JSON matches expected Pydantic schema structure."""

    def setUp(self) -> None:
        super().setUp()
        self._db.insert_specs(_make_specs(
            oil_viscosity="5W-30",
            oil_capacity_l=4.5,
            oil_spec="ACEA C3",
            oil_change_interval_km=10000,
        ))

    def test_response_has_required_top_level_fields(self) -> None:
        client = self._get_client()
        resp = client.post("/api/plate-lookup", json={"plate": "1234567"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Top-level fields
        self.assertIn("vehicle", data)
        self.assertIn("categories", data)
        self.assertIn("coverage", data)
        self.assertIn("unmatched_categories", data)
        self.assertIn("data_source", data)

        # Vehicle fields
        vehicle = data["vehicle"]
        for field in ("plate", "make", "model", "year", "engine_code", "fuel_type"):
            self.assertIn(field, vehicle)

        # Categories structure
        cats = data["categories"]
        for cat in ("oil", "oil_filter", "air_filter", "cabin_filter",
                    "brakes", "bulbs", "coolant"):
            self.assertIn(cat, cats)


if __name__ == "__main__":
    unittest.main()
