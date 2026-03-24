"""AI fallback for unmatched vehicle parts categories.

When the local database has no specs for certain categories, this module
calls Claude Haiku to generate best-effort specifications.  Results are
flagged with ``source="ai_fallback"`` so the UI can display a confidence
badge, and every miss is logged to a JSONL file for later review / DB
enrichment.

Requires ``ANTHROPIC_API_KEY`` environment variable to be set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from parts_finder.api.schemas import (
    BrakesResponse,
    BulbsResponse,
    CoolantResponse,
    FilterCategoryResponse,
    OilResponse,
)
from parts_finder.config import AppConfig
from parts_finder.models import VehicleRecord

logger = logging.getLogger(__name__)

# The 7 product categories the API supports.
_ALL_CATEGORIES = (
    "oil", "oil_filter", "air_filter", "cabin_filter",
    "brakes", "bulbs", "coolant",
)


@dataclass(frozen=True)
class AIFallbackResult:
    """Container for AI-generated specs with provenance tracking.

    Fields mirror ``CategoriesResponse`` — each is ``None`` when the
    category was already matched by the database (or the AI didn't
    return data for it).
    """

    oil: OilResponse | None = None
    oil_filter: FilterCategoryResponse | None = None
    air_filter: FilterCategoryResponse | None = None
    cabin_filter: FilterCategoryResponse | None = None
    brakes: BrakesResponse | None = None
    bulbs: BulbsResponse | None = None
    coolant: CoolantResponse | None = None

    @property
    def has_data(self) -> bool:
        """True if at least one category was populated by the AI."""
        return any(
            getattr(self, cat) is not None for cat in _ALL_CATEGORIES
        )


# Empty singleton for graceful degradation.
_EMPTY_RESULT = AIFallbackResult()


class AIFallback:
    """Generate vehicle parts specs via Claude Haiku for DB misses.

    Usage::

        fallback = AIFallback(config)
        result = await fallback.generate_specs(vehicle, ["oil", "brakes"])
    """

    SYSTEM_PROMPT = (
        "You are a vehicle parts specification expert. Given a vehicle's "
        "make, model, year, engine code, and fuel type, provide accurate "
        "maintenance specifications for the requested categories.\n\n"
        "Return ONLY valid JSON matching the exact schema described in the "
        "user prompt. Do not include explanations or markdown formatting."
    )

    # Per-category JSON schemas that tell the AI what fields to return.
    _CATEGORY_SCHEMAS: dict[str, dict[str, str]] = {
        "oil": (
            '{"viscosity": "e.g. 0W-20", "capacity_l": 0.0, '
            '"spec": "e.g. API SP", "oem_approval": "e.g. Toyota TGMO", '
            '"change_interval_km": 15000}'
        ),
        "oil_filter": '{"oem_part_number": "e.g. 04152-YZZA1"}',
        "air_filter": '{"oem_part_number": "e.g. 17801-21050"}',
        "cabin_filter": '{"oem_part_number": "e.g. 87139-02020"}',
        "brakes": (
            '{"front_pad_oem": "", "rear_pad_oem": "", '
            '"front_disc_oem": "", "rear_disc_oem": "", '
            '"brake_fluid_type": "e.g. DOT 4"}'
        ),
        "bulbs": (
            '{"low_beam": "e.g. H11", "high_beam": "e.g. HB3", '
            '"front_turn": "", "rear_turn": "", "tail_brake": "", '
            '"reverse": "", "fog": "", "license_plate": ""}'
        ),
        "coolant": (
            '{"spec": "e.g. Toyota SLLC", "technology": "e.g. OAT", '
            '"color": "e.g. pink", "capacity_l": 0.0, '
            '"aftermarket_match": "e.g. GLYSANTIN G30", '
            '"mixing_warning": ""}'
        ),
    }

    def __init__(self, config: AppConfig) -> None:
        self._client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        self._model = config.ai_model
        self._max_retries = config.ai_max_retries
        self._misses_path = Path(config.misses_log_path)

    async def generate_specs(
        self,
        vehicle: VehicleRecord,
        unmatched: list[str],
    ) -> AIFallbackResult:
        """Call Claude Haiku to fill in unmatched categories.

        Parameters
        ----------
        vehicle:
            The vehicle record from the government API.
        unmatched:
            Category names (e.g. ``["oil", "brakes"]``) that the
            database had no data for.

        Returns
        -------
        AIFallbackResult
            Populated for the requested categories.  Returns an empty
            result on any error (never crashes the pipeline).
        """
        if not unmatched:
            return _EMPTY_RESULT

        prompt = self._build_prompt(vehicle, unmatched)

        for attempt in range(1, self._max_retries + 1):
            try:
                message = await self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                if not message.content:
                    raise json.JSONDecodeError(
                        "Empty content list from API", "", 0,
                    )
                raw_text = message.content[0].text.strip()
                # Strip markdown fences that models sometimes wrap JSON in
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                raw = json.loads(raw_text)
                result = self._parse_response(raw, unmatched)
                self._log_miss(vehicle, unmatched, raw)
                return result

            except json.JSONDecodeError:
                logger.warning(
                    "AI fallback returned unparseable JSON (attempt %d/%d)",
                    attempt, self._max_retries,
                )
            except anthropic.APIError as exc:
                logger.warning(
                    "AI fallback API error (attempt %d/%d): %s",
                    attempt, self._max_retries, exc,
                )
            except Exception:
                logger.exception(
                    "AI fallback unexpected error (attempt %d/%d)",
                    attempt, self._max_retries,
                )
                break  # Don't retry unknown errors

        # All retries exhausted or unexpected error — log miss without AI data
        self._log_miss(vehicle, unmatched, None)
        return _EMPTY_RESULT

    def _build_prompt(
        self,
        vehicle: VehicleRecord,
        unmatched: list[str],
    ) -> str:
        """Build a prompt asking for only the unmatched categories."""
        make = vehicle.make_english or vehicle.make_hebrew
        model = vehicle.model_english or vehicle.model_hebrew

        lines = [
            f"Vehicle: {make} {model} {vehicle.year}",
            f"Engine code: {vehicle.engine_code}",
            f"Fuel type: {vehicle.fuel_type}",
            "",
            "Provide specifications for these categories ONLY:",
        ]

        for cat in unmatched:
            schema = self._CATEGORY_SCHEMAS.get(cat, "{}")
            lines.append(f"  {cat}: {schema}")

        lines.extend([
            "",
            'Return a single JSON object with category names as keys.',
            'Example: {"oil": {...}, "brakes": {...}}',
            "Only include the categories listed above.",
        ])

        return "\n".join(lines)

    def _parse_response(
        self,
        raw: dict[str, Any],
        unmatched: list[str],
    ) -> AIFallbackResult:
        """Map the AI JSON response to Pydantic schema objects.

        Only populates fields that were in the ``unmatched`` list.
        Unknown categories or malformed sub-objects are silently skipped.
        """
        kwargs: dict[str, Any] = {}

        for cat in unmatched:
            if cat not in raw:
                continue
            data = raw[cat]
            if not isinstance(data, dict):
                continue

            try:
                if cat == "oil":
                    kwargs["oil"] = OilResponse(
                        viscosity=data.get("viscosity", ""),
                        capacity_l=float(data.get("capacity_l", 0)),
                        spec=data.get("spec", ""),
                        oem_approval=data.get("oem_approval", ""),
                        change_interval_km=int(data.get("change_interval_km", 0)),
                        confidence="ai_fallback",
                        source="ai_fallback",
                    )
                elif cat in ("oil_filter", "air_filter", "cabin_filter"):
                    oem = data.get("oem_part_number", "")
                    if oem:
                        kwargs[cat] = FilterCategoryResponse(
                            oem_part_number=oem,
                            aftermarket=[],
                        )
                elif cat == "brakes":
                    kwargs["brakes"] = BrakesResponse(
                        front_pad_oem=data.get("front_pad_oem", ""),
                        rear_pad_oem=data.get("rear_pad_oem", ""),
                        front_disc_oem=data.get("front_disc_oem", ""),
                        rear_disc_oem=data.get("rear_disc_oem", ""),
                        brake_fluid_type=data.get("brake_fluid_type", ""),
                    )
                elif cat == "bulbs":
                    kwargs["bulbs"] = BulbsResponse(
                        low_beam=data.get("low_beam", ""),
                        high_beam=data.get("high_beam", ""),
                        front_turn=data.get("front_turn", ""),
                        rear_turn=data.get("rear_turn", ""),
                        tail_brake=data.get("tail_brake", ""),
                        reverse=data.get("reverse", ""),
                        fog=data.get("fog", ""),
                        license_plate=data.get("license_plate", ""),
                    )
                elif cat == "coolant":
                    kwargs["coolant"] = CoolantResponse(
                        spec=data.get("spec", ""),
                        technology=data.get("technology", ""),
                        color=data.get("color", ""),
                        capacity_l=float(data.get("capacity_l", 0)),
                        aftermarket_match=data.get("aftermarket_match", ""),
                        mixing_warning=data.get("mixing_warning", ""),
                    )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed AI data for %s: %s", cat, exc)
                continue

        return AIFallbackResult(**kwargs)

    def _log_miss(
        self,
        vehicle: VehicleRecord,
        unmatched: list[str],
        ai_response: dict[str, Any] | None,
    ) -> None:
        """Append a miss record to the JSONL log file.

        Each line is a self-contained JSON object for easy grep/review.
        """
        make = vehicle.make_english or vehicle.make_hebrew
        model = vehicle.model_english or vehicle.model_hebrew

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "plate": vehicle.plate,
            "make": make,
            "model": model,
            "year": vehicle.year,
            "engine_code": vehicle.engine_code,
            "fuel_type": vehicle.fuel_type,
            "unmatched_categories": unmatched,
            "ai_response": ai_response,
        }

        try:
            self._misses_path.parent.mkdir(parents=True, exist_ok=True)
            with self._misses_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Failed to write miss log to %s", self._misses_path)
