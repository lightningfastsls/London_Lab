"""Tests for normalize_plate and PlateClient."""

from __future__ import annotations

import json
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from parts_finder.config import AppConfig
from parts_finder.exceptions import GovApiError, PlateNotFoundError
from parts_finder.plate_client import PlateClient, normalize_plate


# ---------------------------------------------------------------------------
# normalize_plate
# ---------------------------------------------------------------------------

class TestNormalizePlate(unittest.TestCase):
    """Pure-function tests for plate string normalization."""

    def test_digits_passthrough(self) -> None:
        self.assertEqual(normalize_plate("1234567"), "1234567")

    def test_strip_hyphens(self) -> None:
        self.assertEqual(normalize_plate("12-345-67"), "1234567")

    def test_strip_spaces(self) -> None:
        self.assertEqual(normalize_plate("12 345 67"), "1234567")

    def test_strip_leading_zeros(self) -> None:
        self.assertEqual(normalize_plate("0012345"), "12345")

    def test_mixed_separators(self) -> None:
        self.assertEqual(normalize_plate(" 12-345 67 "), "1234567")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_plate("")

    def test_all_zeros_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_plate("000-00-00")

    def test_no_digits_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_plate("abc-def")

    def test_eight_digit_plate(self) -> None:
        self.assertEqual(normalize_plate("12-345-678"), "12345678")


# ---------------------------------------------------------------------------
# PlateClient
# ---------------------------------------------------------------------------

def _success_response(records: list[dict]) -> httpx.Response:
    """Build a fake httpx.Response with a successful CKAN payload."""
    body = {"success": True, "result": {"records": records}}
    resp = httpx.Response(
        status_code=200,
        json=body,
        request=httpx.Request("GET", "https://fake"),
    )
    return resp


def _make_record(**overrides: object) -> dict:
    """Realistic API record dict."""
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


class TestPlateClientLookup(unittest.IsolatedAsyncioTestCase):
    """Async tests for PlateClient.lookup with mocked HTTP."""

    async def _make_client(self, mock_response: httpx.Response | Exception) -> PlateClient:
        """Create a PlateClient with a mocked internal HTTP client."""
        config = AppConfig()
        client = PlateClient(config)
        # Manually set up the internal HTTP client with a mock
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        if isinstance(mock_response, Exception):
            mock_http.get.side_effect = mock_response
        else:
            mock_http.get.return_value = mock_response
        mock_http.aclose = AsyncMock()
        client._http = mock_http
        return client

    # --- Happy path ---

    async def test_lookup_happy_path(self) -> None:
        client = await self._make_client(_success_response([_make_record()]))
        vehicle = await client.lookup("12-345-67")
        self.assertEqual(vehicle.plate, "1234567")
        self.assertEqual(vehicle.engine_code, "1ZR-FE")
        self.assertEqual(vehicle.make_hebrew, "טויוטה")
        self.assertEqual(vehicle.year, 2020)

    async def test_lookup_correct_query_params(self) -> None:
        response = _success_response([_make_record()])
        config = AppConfig()
        client = await self._make_client(response)
        await client.lookup("1234567")
        call_kwargs = client._http.get.call_args  # type: ignore[union-attr]
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        self.assertEqual(params["resource_id"], config.gov_api_resource_id)
        self.assertEqual(params["limit"], 1)
        filters = json.loads(params["filters"])
        self.assertEqual(filters["mispar_rechev"], "1234567")

    # --- Error paths ---

    async def test_plate_not_found(self) -> None:
        response = _success_response([])  # empty records
        client = await self._make_client(response)
        with self.assertRaises(PlateNotFoundError) as ctx:
            await client.lookup("9999999")
        self.assertEqual(ctx.exception.plate, "9999999")

    async def test_api_timeout(self) -> None:
        client = await self._make_client(httpx.ReadTimeout("timed out"))
        with self.assertRaises(GovApiError) as ctx:
            await client.lookup("1234567")
        self.assertIn("Timeout", str(ctx.exception))

    async def test_http_500(self) -> None:
        request = httpx.Request("GET", "https://fake")
        response = httpx.Response(status_code=500, request=request)
        exc = httpx.HTTPStatusError(
            "Server Error", request=request, response=response
        )
        client = await self._make_client(exc)
        with self.assertRaises(GovApiError) as ctx:
            await client.lookup("1234567")
        self.assertEqual(ctx.exception.status_code, 500)

    async def test_api_success_false(self) -> None:
        body = {"success": False, "error": {"message": "bad query"}}
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://fake"),
        )
        client = await self._make_client(resp)
        with self.assertRaises(GovApiError) as ctx:
            await client.lookup("1234567")
        self.assertIn("success=false", str(ctx.exception))

    async def test_invalid_json(self) -> None:
        resp = httpx.Response(
            status_code=200,
            content=b"not json at all",
            request=httpx.Request("GET", "https://fake"),
        )
        client = await self._make_client(resp)
        with self.assertRaises(GovApiError) as ctx:
            await client.lookup("1234567")
        self.assertIn("Invalid JSON", str(ctx.exception))

    async def test_network_connect_error(self) -> None:
        client = await self._make_client(
            httpx.ConnectError("connection refused")
        )
        with self.assertRaises(GovApiError) as ctx:
            await client.lookup("1234567")
        self.assertIn("Network error", str(ctx.exception))

    async def test_malformed_record_raises_gov_api_error(self) -> None:
        bad_record = {
            "kinuy_mishari": "Corolla",
            "shnat_yitzur": 2020,
            "degem_manoa": "1ZR-FE",
            "sug_delek_nm": "petrol",
        }  # Missing required "tozeret_nm"
        client = await self._make_client(_success_response([bad_record]))
        with self.assertRaises(GovApiError) as ctx:
            await client.lookup("1234567")
        self.assertIn("Malformed", str(ctx.exception))

    async def test_null_required_field_raises_gov_api_error(self) -> None:
        record = _make_record(tozeret_nm=None)
        client = await self._make_client(_success_response([record]))
        with self.assertRaises(GovApiError) as ctx:
            await client.lookup("1234567")
        self.assertIn("Malformed", str(ctx.exception))

    async def test_missing_engine_code_warning(self) -> None:
        record = _make_record()
        del record["degem_manoa"]
        client = await self._make_client(_success_response([record]))
        with self.assertLogs("parts_finder.plate_client", level="WARNING") as cm:
            vehicle = await client.lookup("1234567")
        self.assertEqual(vehicle.engine_code, "")
        self.assertTrue(any("engine code" in msg.lower() for msg in cm.output))

    async def test_multiple_records_warning(self) -> None:
        records = [_make_record(), _make_record(degem_manoa="2ZR-FE")]
        client = await self._make_client(_success_response(records))
        with self.assertLogs("parts_finder.plate_client", level="WARNING") as cm:
            vehicle = await client.lookup("1234567")
        self.assertEqual(vehicle.engine_code, "1ZR-FE")  # first record used
        self.assertTrue(any("multiple" in msg.lower() for msg in cm.output))

    # --- Context manager ---

    async def test_context_manager_required(self) -> None:
        config = AppConfig()
        client = PlateClient(config)
        with self.assertRaises(RuntimeError):
            await client.lookup("1234567")

    async def test_context_manager_lifecycle(self) -> None:
        config = AppConfig()
        async with PlateClient(config) as client:
            # Client should be usable — we just verify _http is set
            self.assertIsNotNone(client._http)
        # After exit, _http is None
        self.assertIsNone(client._http)


if __name__ == "__main__":
    unittest.main()
