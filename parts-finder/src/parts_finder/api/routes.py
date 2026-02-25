"""API route handlers for the Parts Finder service.

Currently exposes a single endpoint:

    POST /api/plate-lookup
        Request:  ``{"plate": "1234567"}``
        Response: Vehicle info + 7 product categories + coverage summary

Error mapping:
    ValueError         → 400 (invalid plate format)
    PlateNotFoundError → 404 (plate not in registry)
    GovApiError        → 503 (government API unavailable)
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from parts_finder.api.response_builder import build_response
from parts_finder.api.schemas import (
    ErrorResponse,
    PlateLookupRequest,
    PlateLookupResponse,
)
from parts_finder.exceptions import GovApiError, PlateNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.post(
    "/plate-lookup",
    response_model=PlateLookupResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid plate format"},
        404: {"model": ErrorResponse, "description": "Plate not found in registry"},
        503: {"model": ErrorResponse, "description": "Government API unavailable"},
    },
)
async def plate_lookup(
    body: PlateLookupRequest,
    request: Request,
) -> PlateLookupResponse | JSONResponse:
    """Look up vehicle parts recommendations by license plate.

    Flow:
    1. Validate plate format
    2. Check in-memory cache
    3. Call government API + run category lookups
    4. Cache and return response
    """
    config = request.app.state.config
    engine = request.app.state.engine
    cache: dict = request.app.state.cache

    plate = body.plate.strip()

    # ── Cache check ──────────────────────────────────────────────
    if config.cache_enabled and plate in cache:
        cached_response, cached_at = cache[plate]
        age_minutes = (time.monotonic() - cached_at) / 60.0
        if age_minutes < config.cache_ttl_minutes:
            logger.debug("Cache hit for plate %s (age %.1f min)", plate, age_minutes)
            return cached_response
        else:
            # Expired — remove stale entry
            del cache[plate]

    # ── Lookup ───────────────────────────────────────────────────
    try:
        result = await engine.lookup(plate)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )
    except PlateNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc)},
        )
    except GovApiError as exc:
        logger.error("Government API error for plate %s: %s", plate, exc)
        return JSONResponse(
            status_code=503,
            content={"detail": f"Government API unavailable: {exc}"},
        )

    # ── Build response ───────────────────────────────────────────
    response = build_response(result)

    # ── AI fallback for unmatched categories ──────────────────────
    fallback = getattr(request.app.state, "fallback", None)
    if fallback is not None and response.unmatched_categories:
        try:
            ai_result = await fallback.generate_specs(
                result.vehicle, response.unmatched_categories,
            )
            if ai_result.has_data:
                response = build_response(result, ai_result=ai_result)
        except Exception:
            logger.exception("AI fallback failed for plate %s", plate)

    # ── Cache store ──────────────────────────────────────────────
    if config.cache_enabled:
        cache[plate] = (response, time.monotonic())

    return response
