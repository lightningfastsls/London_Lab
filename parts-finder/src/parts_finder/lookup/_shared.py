"""Shared utilities for lookup modules.

Extracted from the four lookup classes (oil, bulbs, coolant, brakes)
that each had an identical ``_resolve_names`` method.  Centralised
here to eliminate the duplication flagged across multiple reviews.
"""

from __future__ import annotations

import logging
from typing import Optional

from parts_finder.models import VehicleRecord


def resolve_vehicle_names(
    vehicle: VehicleRecord,
    logger: logging.Logger,
) -> tuple[str | None, str | None]:
    """Resolve make/model names, preferring English over Hebrew.

    Returns ``(make, model)`` where either may be ``None``.
    If only Hebrew names are available, logs a warning via the
    caller-supplied *logger* so log messages retain the calling
    module's identity.
    """
    make = vehicle.make_english or vehicle.make_hebrew
    model = vehicle.model_english or vehicle.model_hebrew

    if not make:
        return None, None

    if not vehicle.make_english:
        logger.warning(
            "Using Hebrew make name '%s' for plate %s — "
            "DB match unlikely without NameMapper enrichment",
            vehicle.make_hebrew, vehicle.plate,
        )

    return make, model
