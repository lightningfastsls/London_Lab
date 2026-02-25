"""Application configuration for the Parts Finder service.

Frozen dataclass pattern adapted from usv_spectrogram/detection/config.py.
All fields have sensible defaults; override via constructor kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Configuration for the Parts Finder application.

    Groups settings for: government API access, local database,
    response caching, AI fallback, and HTTP server.
    """

    # --- Government API (data.gov.il) ---
    gov_api_base_url: str = "https://data.gov.il/api/3/action/datastore_search"
    gov_api_resource_id: str = "053cea08-09bc-40ec-8f7a-156f0677aff3"
    gov_api_timeout_s: float = 10.0  # HTTP timeout in seconds

    # --- Database ---
    db_path: str = "parts_finder.db"

    # --- Cache ---
    cache_enabled: bool = True
    cache_ttl_minutes: int = 60  # Time-to-live for cached responses

    # --- AI fallback (Anthropic) ---
    ai_model: str = "claude-3-5-haiku-20241022"
    ai_max_retries: int = 3
    ai_fallback_enabled: bool = True  # set False to disable even with API key
    misses_log_path: str = "data/misses.jsonl"

    # --- Server ---
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    server_debug: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters at construction time."""
        if not self.gov_api_base_url.strip():
            raise ValueError("gov_api_base_url must not be empty")
        if not self.gov_api_resource_id.strip():
            raise ValueError("gov_api_resource_id must not be empty")
        if self.gov_api_timeout_s <= 0:
            raise ValueError("gov_api_timeout_s must be positive")
        if self.cache_ttl_minutes <= 0:
            raise ValueError("cache_ttl_minutes must be positive")
        if self.ai_max_retries < 0:
            raise ValueError("ai_max_retries must be >= 0")
        if not self.misses_log_path.strip():
            raise ValueError("misses_log_path must not be empty")
        if not (1 <= self.server_port <= 65535):
            raise ValueError("server_port must be between 1 and 65535")
