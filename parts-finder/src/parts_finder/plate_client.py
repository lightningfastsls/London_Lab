"""Async client for the Israeli government vehicle registry API.

Usage::

    async with PlateClient(config) as client:
        vehicle = await client.lookup("12-345-67")
        print(vehicle.engine_code)  # "1ZR-FE"

The data.gov.il CKAN ``datastore_search`` endpoint is free and keyless.
It returns vehicle records keyed by ``mispar_rechev`` (license plate number).
"""

from __future__ import annotations

import json
import logging
from types import TracebackType
from typing import Optional

import httpx

from parts_finder.config import AppConfig
from parts_finder.exceptions import GovApiError, PlateNotFoundError
from parts_finder.models import VehicleRecord

logger = logging.getLogger(__name__)


def normalize_plate(raw: str) -> str:
    """Strip hyphens, spaces, and non-digit characters; drop leading zeros.

    Raises ``ValueError`` if the result is empty (no digits at all).
    """
    digits = "".join(ch for ch in raw if ch.isdigit())
    digits = digits.lstrip("0")
    if not digits:
        raise ValueError(f"Invalid plate (no digits): {raw!r}")
    return digits


class PlateClient:
    """Async HTTP client for the data.gov.il vehicle registry.

    Must be used as an async context manager to ensure the underlying
    ``httpx.AsyncClient`` is properly opened and closed.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._http: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> PlateClient:
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.gov_api_timeout_s),
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def lookup(self, plate: str) -> VehicleRecord:
        """Look up a vehicle by license plate number.

        Parameters
        ----------
        plate:
            Raw plate string (hyphens/spaces are stripped automatically).

        Returns
        -------
        VehicleRecord
            Parsed vehicle information including the critical ``engine_code``.

        Raises
        ------
        RuntimeError
            If the client is used outside an ``async with`` block.
        PlateNotFoundError
            If the plate is not in the registry.
        GovApiError
            For HTTP errors, timeouts, or malformed API responses.
        """
        if self._http is None:
            raise RuntimeError(
                "PlateClient must be used as an async context manager: "
                "async with PlateClient(config) as client: ..."
            )

        normalized = normalize_plate(plate)

        params = {
            "resource_id": self._config.gov_api_resource_id,
            "filters": json.dumps({"mispar_rechev": normalized}),
            "limit": 1,
        }

        try:
            response = await self._http.get(
                self._config.gov_api_base_url,
                params=params,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise GovApiError(
                f"Timeout querying plate {normalized}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GovApiError(
                f"HTTP {exc.response.status_code} for plate {normalized}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise GovApiError(
                f"Network error querying plate {normalized}: {exc}"
            ) from exc

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise GovApiError("Invalid JSON in API response") from exc

        if not data.get("success"):
            raise GovApiError(
                f"API returned success=false: {data.get('error', 'unknown')}"
            )

        records = data.get("result", {}).get("records", [])
        if not records:
            raise PlateNotFoundError(normalized)

        if len(records) > 1:
            logger.warning(
                "Multiple records returned for plate %s, using first",
                normalized,
            )

        record = records[0]
        try:
            vehicle = VehicleRecord.from_api_record(normalized, record)
        except (KeyError, ValueError, TypeError) as exc:
            raise GovApiError(
                f"Malformed API record for plate {normalized}: {exc}"
            ) from exc

        if not vehicle.engine_code:
            logger.warning(
                "No engine code (degem_manoa) for plate %s", normalized
            )

        return vehicle
