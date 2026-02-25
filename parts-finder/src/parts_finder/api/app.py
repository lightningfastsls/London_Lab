"""FastAPI application factory for the Parts Finder service.

Uses the modern ``lifespan`` async context manager (not the deprecated
``on_event("startup")`` / ``on_event("shutdown")``).  Shared resources
(database, name mapper, lookup engine) are stored on ``app.state`` and
accessed by routes via ``request.app.state``.

Usage::

    from parts_finder.api.app import create_app
    app = create_app()  # uses default AppConfig

    # Or with custom config:
    from parts_finder.config import AppConfig
    app = create_app(AppConfig(db_path="custom.db"))
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from parts_finder.api.routes import router
from parts_finder.config import AppConfig
from parts_finder.db import PartsDatabase
from parts_finder.lookup_engine import LookupEngine
from parts_finder.name_mapper import NameMapper

logger = logging.getLogger(__name__)

# Default path to the Hebrew→English name mapping file.
_DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[2] / "data" / "hebrew_names.json"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage shared resources across the application lifetime.

    Opens the database and name mapper at startup; closes the database
    at shutdown.  The LookupEngine is constructed once and reused for
    all requests.
    """
    config: AppConfig = app.state.config

    # Resolve mapping path — use configured path or default
    mapping_path = getattr(config, "name_mapping_path", None)
    if mapping_path is None:
        mapping_path = _DEFAULT_MAPPING_PATH

    db = PartsDatabase(config.db_path)
    try:
        mapper = NameMapper(Path(mapping_path))
        engine = LookupEngine(config, db, mapper)

        app.state.db = db
        app.state.mapper = mapper
        app.state.engine = engine
        app.state.cache = {}  # plate -> (response, monotonic_timestamp)

        # AI fallback — opt-in via ANTHROPIC_API_KEY env var + config flag
        app.state.fallback = None
        if config.ai_fallback_enabled and os.environ.get("ANTHROPIC_API_KEY"):
            from parts_finder.api.fallback import AIFallback
            app.state.fallback = AIFallback(config)
            logger.info("AI fallback enabled (model=%s)", config.ai_model)
        else:
            logger.info("AI fallback disabled")

        logger.info("Parts Finder API started (db=%s)", config.db_path)
        yield
    finally:
        db.close()
        logger.info("Parts Finder API shutdown — database closed")


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    Parameters
    ----------
    config:
        Application configuration.  If ``None``, uses default
        :class:`AppConfig` values.
    """
    config = config or AppConfig()

    app = FastAPI(
        title="Parts Finder API",
        version="0.1.0",
        description="License plate → vehicle parts recommendations",
        lifespan=_lifespan,
    )

    # Store config on state BEFORE lifespan runs
    app.state.config = config

    app.include_router(router)

    return app
