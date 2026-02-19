"""Configuration loading for Notion Atomic Notes.

Loads settings from environment variables, optionally reading a .env file first.
Uses manual .env parsing to avoid adding python-dotenv as a dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NotionNotesConfig:
    """Immutable configuration for the Notion Notes automation."""

    notion_token: str
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    notion_rate_limit_rps: float = 3.0
    kb_database_id: str = ""
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not self.notion_token:
            raise ValueError("notion_token must not be empty")
        if not self.anthropic_api_key:
            raise ValueError("anthropic_api_key must not be empty")
        if self.notion_rate_limit_rps <= 0:
            raise ValueError(
                f"notion_rate_limit_rps must be positive, got {self.notion_rate_limit_rps}"
            )


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict. Skips comments and blank lines.

    Handles optional quoting (single or double) around values.
    Does NOT overwrite variables already set in the environment.
    """
    result: dict[str, str] = {}
    if not path.is_file():
        return result

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip matching quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def load_config(
    env_path: Path | None = None,
    dry_run: bool = False,
) -> NotionNotesConfig:
    """Load configuration from environment variables, optionally reading a .env file.

    Parameters
    ----------
    env_path : Path or None
        Path to a .env file. If provided, variables from the file are loaded
        into ``os.environ`` (without overwriting existing values).
    dry_run : bool
        If True, write operations will be skipped.

    Raises
    ------
    ValueError
        If required environment variables (NOTION_TOKEN, ANTHROPIC_API_KEY)
        are missing or empty.
    """
    if env_path is not None:
        for key, value in _parse_env_file(env_path).items():
            os.environ[key] = value  # .env file wins over system env vars

    notion_token = os.environ.get("NOTION_TOKEN", "")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    missing: list[str] = []
    if not notion_token:
        missing.append("NOTION_TOKEN")
    if not anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Set them in your environment or in a .env file."
        )

    return NotionNotesConfig(
        notion_token=notion_token,
        anthropic_api_key=anthropic_api_key,
        claude_model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        notion_rate_limit_rps=float(
            os.environ.get("NOTION_RATE_LIMIT_RPS", "3.0")
        ),
        kb_database_id=os.environ.get("NOTION_KB_DATABASE_ID", os.environ.get("KB_DATABASE_ID", "")),
        dry_run=dry_run,
    )
