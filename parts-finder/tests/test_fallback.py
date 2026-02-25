"""Tests for the AI fallback module.

All tests mock the Anthropic API — no real API calls are made.
Tests follow the project's unittest pattern (see test_api.py).

Test plan:
1. Happy path — valid AI response populates AIFallbackResult
2. API error — anthropic.APIError → empty result (no crash)
3. Malformed JSON — unparseable response → empty result
4. Selective categories — only unmatched categories populated
5. Miss logging — _log_miss writes correct JSONL structure
6. Config wiring — reads model name and max_retries from AppConfig
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic

from parts_finder.api.fallback import AIFallback, AIFallbackResult
from parts_finder.api.schemas import (
    BrakesResponse,
    CoolantResponse,
    FilterCategoryResponse,
    OilResponse,
)
from parts_finder.config import AppConfig
from parts_finder.models import VehicleRecord


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


def _make_ai_json(*categories: str) -> dict:
    """Build a mock AI JSON response for the given categories."""
    data: dict = {}
    if "oil" in categories:
        data["oil"] = {
            "viscosity": "0W-20",
            "capacity_l": 4.2,
            "spec": "API SP",
            "oem_approval": "Toyota TGMO",
            "change_interval_km": 15000,
        }
    if "oil_filter" in categories:
        data["oil_filter"] = {"oem_part_number": "04152-YZZA1"}
    if "air_filter" in categories:
        data["air_filter"] = {"oem_part_number": "17801-21050"}
    if "cabin_filter" in categories:
        data["cabin_filter"] = {"oem_part_number": "87139-02020"}
    if "brakes" in categories:
        data["brakes"] = {
            "front_pad_oem": "04465-02220",
            "rear_pad_oem": "04466-02181",
            "front_disc_oem": "43512-02330",
            "rear_disc_oem": "42431-02250",
            "brake_fluid_type": "DOT 4",
        }
    if "bulbs" in categories:
        data["bulbs"] = {
            "low_beam": "H11",
            "high_beam": "HB3",
        }
    if "coolant" in categories:
        data["coolant"] = {
            "spec": "Toyota SLLC",
            "technology": "OAT",
            "color": "pink",
            "capacity_l": 6.4,
            "aftermarket_match": "GLYSANTIN G30",
            "mixing_warning": "Do NOT mix with IAT coolant",
        }
    return data


def _mock_anthropic_response(json_data: dict) -> MagicMock:
    """Build a mock Message object matching anthropic's response shape."""
    content_block = MagicMock()
    content_block.text = json.dumps(json_data)
    message = MagicMock()
    message.content = [content_block]
    return message


class _FallbackTestBase(unittest.TestCase):
    """Base class that sets up config with temp directory for miss log."""

    def setUp(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self._misses_path = str(Path(self._tmpdir.name) / "misses.jsonl")
        self._config = AppConfig(misses_log_path=self._misses_path)
        self._vehicle = _make_vehicle()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _get_fallback(self) -> AIFallback:
        """Create AIFallback with mocked Anthropic client."""
        with patch("parts_finder.api.fallback.anthropic.AsyncAnthropic"):
            return AIFallback(self._config)


# ── Test 1: Happy path ──────────────────────────────────────────


class TestHappyPath(_FallbackTestBase):
    """Mocked API returns valid JSON — generate_specs returns populated result."""

    def test_oil_and_brakes_populated(self) -> None:
        fallback = self._get_fallback()
        ai_json = _make_ai_json("oil", "brakes")
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(ai_json),
        )

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, ["oil", "brakes"]),
        )

        self.assertIsNotNone(result.oil)
        self.assertEqual(result.oil.viscosity, "0W-20")
        self.assertEqual(result.oil.capacity_l, 4.2)
        self.assertEqual(result.oil.confidence, "ai_fallback")
        self.assertEqual(result.oil.source, "ai_fallback")

        self.assertIsNotNone(result.brakes)
        self.assertEqual(result.brakes.front_pad_oem, "04465-02220")
        self.assertEqual(result.brakes.brake_fluid_type, "DOT 4")

        # Unrelated categories should be None
        self.assertIsNone(result.bulbs)
        self.assertIsNone(result.coolant)
        self.assertTrue(result.has_data)

    def test_all_seven_categories(self) -> None:
        fallback = self._get_fallback()
        all_cats = [
            "oil", "oil_filter", "air_filter", "cabin_filter",
            "brakes", "bulbs", "coolant",
        ]
        ai_json = _make_ai_json(*all_cats)
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(ai_json),
        )

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, all_cats),
        )

        for cat in all_cats:
            self.assertIsNotNone(
                getattr(result, cat),
                f"Expected {cat} to be populated",
            )


# ── Test 2: API error — graceful degradation ────────────────────


class TestAPIError(_FallbackTestBase):
    """anthropic.APIError → empty result, no crash."""

    def test_api_error_returns_empty(self) -> None:
        fallback = self._get_fallback()
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            side_effect=anthropic.APIError(
                message="Internal server error",
                request=MagicMock(),
                body=None,
            ),
        )

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, ["oil"]),
        )

        self.assertFalse(result.has_data)
        self.assertIsNone(result.oil)

    def test_retries_on_api_error(self) -> None:
        """Should retry up to max_retries times."""
        config = AppConfig(
            misses_log_path=self._misses_path,
            ai_max_retries=2,
        )
        with patch("parts_finder.api.fallback.anthropic.AsyncAnthropic"):
            fallback = AIFallback(config)

        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            side_effect=anthropic.APIError(
                message="Rate limited",
                request=MagicMock(),
                body=None,
            ),
        )

        asyncio.run(fallback.generate_specs(self._vehicle, ["oil"]))

        self.assertEqual(fallback._client.messages.create.call_count, 2)


# ── Test 3: Malformed JSON ──────────────────────────────────────


class TestMalformedJSON(_FallbackTestBase):
    """API returns unparseable response → empty result."""

    def test_invalid_json_returns_empty(self) -> None:
        fallback = self._get_fallback()
        fallback._client = AsyncMock()

        bad_response = MagicMock()
        bad_content = MagicMock()
        bad_content.text = "This is not JSON at all"
        bad_response.content = [bad_content]
        fallback._client.messages.create = AsyncMock(return_value=bad_response)

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, ["oil"]),
        )

        self.assertFalse(result.has_data)

    def test_markdown_fenced_json_is_parsed(self) -> None:
        """Model wraps JSON in ```json fences — should still parse."""
        fallback = self._get_fallback()
        fallback._client = AsyncMock()

        ai_json = _make_ai_json("oil")
        fenced_text = "```json\n" + json.dumps(ai_json) + "\n```"
        fenced_response = MagicMock()
        fenced_content = MagicMock()
        fenced_content.text = fenced_text
        fenced_response.content = [fenced_content]
        fallback._client.messages.create = AsyncMock(return_value=fenced_response)

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, ["oil"]),
        )

        self.assertTrue(result.has_data)
        self.assertIsNotNone(result.oil)
        self.assertEqual(result.oil.viscosity, "0W-20")

    def test_empty_content_list_retries(self) -> None:
        """Empty message.content triggers retry (not immediate break)."""
        config = AppConfig(
            misses_log_path=self._misses_path,
            ai_max_retries=2,
        )
        with patch("parts_finder.api.fallback.anthropic.AsyncAnthropic"):
            fallback = AIFallback(config)

        empty_response = MagicMock()
        empty_response.content = []
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(return_value=empty_response)

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, ["oil"]),
        )

        self.assertFalse(result.has_data)
        # Should have retried (not broken on first attempt)
        self.assertEqual(fallback._client.messages.create.call_count, 2)

    def test_json_with_wrong_types_skips_category(self) -> None:
        """AI returns a string instead of dict for a category — skip it."""
        fallback = self._get_fallback()
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response({"oil": "not a dict"}),
        )

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, ["oil"]),
        )

        self.assertIsNone(result.oil)


# ── Test 4: Selective categories ────────────────────────────────


class TestSelectiveCategories(_FallbackTestBase):
    """Only unmatched categories populated, matched ones stay None."""

    def test_only_requested_categories_populated(self) -> None:
        fallback = self._get_fallback()
        # AI returns data for all, but we only request oil_filter
        ai_json = _make_ai_json("oil", "oil_filter", "brakes")
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(ai_json),
        )

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, ["oil_filter"]),
        )

        # oil_filter was requested and AI returned it → populated
        self.assertIsNotNone(result.oil_filter)
        self.assertEqual(result.oil_filter.oem_part_number, "04152-YZZA1")

        # oil and brakes were NOT requested → None even though AI had data
        self.assertIsNone(result.oil)
        self.assertIsNone(result.brakes)

    def test_empty_unmatched_returns_empty(self) -> None:
        """No unmatched categories → no API call needed."""
        fallback = self._get_fallback()
        fallback._client = AsyncMock()

        result = asyncio.run(
            fallback.generate_specs(self._vehicle, []),
        )

        self.assertFalse(result.has_data)
        # Client should not have been called
        fallback._client.messages.create.assert_not_called()


# ── Test 5: Miss logging ────────────────────────────────────────


class TestMissLogging(_FallbackTestBase):
    """_log_miss writes correct JSONL structure."""

    def test_successful_response_logged(self) -> None:
        fallback = self._get_fallback()
        ai_json = _make_ai_json("oil")
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(ai_json),
        )

        asyncio.run(
            fallback.generate_specs(self._vehicle, ["oil"]),
        )

        # Read the JSONL file
        misses_path = Path(self._misses_path)
        self.assertTrue(misses_path.exists(), "misses.jsonl should exist")
        lines = misses_path.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 1)

        record = json.loads(lines[0])
        self.assertIn("timestamp", record)
        self.assertEqual(record["plate"], "1234567")
        self.assertEqual(record["make"], "Toyota")
        self.assertEqual(record["model"], "Corolla")
        self.assertEqual(record["year"], 2021)
        self.assertEqual(record["engine_code"], "2ZR-FE")
        self.assertEqual(record["unmatched_categories"], ["oil"])
        self.assertIsNotNone(record["ai_response"])
        self.assertIn("oil", record["ai_response"])

    def test_failed_response_logged_with_null_ai(self) -> None:
        """On API error, miss logged with ai_response=None."""
        fallback = self._get_fallback()
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            side_effect=anthropic.APIError(
                message="error",
                request=MagicMock(),
                body=None,
            ),
        )

        asyncio.run(
            fallback.generate_specs(self._vehicle, ["brakes"]),
        )

        misses_path = Path(self._misses_path)
        self.assertTrue(misses_path.exists())
        record = json.loads(
            misses_path.read_text(encoding="utf-8").strip(),
        )
        self.assertIsNone(record["ai_response"])
        self.assertEqual(record["unmatched_categories"], ["brakes"])

    def test_multiple_misses_appended(self) -> None:
        """Multiple calls append to the same file."""
        fallback = self._get_fallback()
        ai_json = _make_ai_json("oil")
        fallback._client = AsyncMock()
        fallback._client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(ai_json),
        )

        asyncio.run(fallback.generate_specs(self._vehicle, ["oil"]))
        asyncio.run(fallback.generate_specs(self._vehicle, ["oil"]))

        lines = (
            Path(self._misses_path)
            .read_text(encoding="utf-8")
            .strip()
            .split("\n")
        )
        self.assertEqual(len(lines), 2)


# ── Test 6: Config wiring ───────────────────────────────────────


class TestConfigWiring(_FallbackTestBase):
    """AIFallback reads model name and max_retries from AppConfig."""

    def test_model_from_config(self) -> None:
        config = AppConfig(
            ai_model="claude-3-5-haiku-20241022",
            misses_log_path=self._misses_path,
        )
        with patch("parts_finder.api.fallback.anthropic.AsyncAnthropic"):
            fallback = AIFallback(config)

        self.assertEqual(fallback._model, "claude-3-5-haiku-20241022")

    def test_custom_model(self) -> None:
        config = AppConfig(
            ai_model="claude-3-haiku-20240307",
            misses_log_path=self._misses_path,
        )
        with patch("parts_finder.api.fallback.anthropic.AsyncAnthropic"):
            fallback = AIFallback(config)

        self.assertEqual(fallback._model, "claude-3-haiku-20240307")

    def test_max_retries_from_config(self) -> None:
        config = AppConfig(
            ai_max_retries=5,
            misses_log_path=self._misses_path,
        )
        with patch("parts_finder.api.fallback.anthropic.AsyncAnthropic"):
            fallback = AIFallback(config)

        self.assertEqual(fallback._max_retries, 5)

    def test_misses_path_from_config(self) -> None:
        custom_path = str(Path(self._tmpdir.name) / "custom" / "log.jsonl")
        config = AppConfig(misses_log_path=custom_path)
        with patch("parts_finder.api.fallback.anthropic.AsyncAnthropic"):
            fallback = AIFallback(config)

        self.assertEqual(str(fallback._misses_path), custom_path)


# ── Test: AIFallbackResult dataclass ─────────────────────────────


class TestAIFallbackResult(unittest.TestCase):
    """AIFallbackResult is a frozen dataclass with correct defaults."""

    def test_empty_result_has_no_data(self) -> None:
        self.assertFalse(AIFallbackResult().has_data)

    def test_result_with_oil_has_data(self) -> None:
        result = AIFallbackResult(
            oil=OilResponse(
                viscosity="0W-20",
                capacity_l=4.2,
                spec="API SP",
                oem_approval="",
                change_interval_km=15000,
                confidence="ai_fallback",
                source="ai_fallback",
            ),
        )
        self.assertTrue(result.has_data)

    def test_frozen(self) -> None:
        """AIFallbackResult is immutable."""
        with self.assertRaises(AttributeError):
            AIFallbackResult().oil = "should fail"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
