"""Find and tag 4 specific pages by title."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notion_notes.config import load_config
from notion_notes.notion_client import NotionClientWrapper

KB_DB_ID = "30b2bc599f6e8032b337fbda2c975dda"

TARGETS = [
    "more proteins more technologies",
    "ta13",
    "lecture 12",
    "lecture 3",
]


def main() -> None:
    cfg = load_config(env_path=Path(".env"))
    notion = NotionClientWrapper(
        token=cfg.notion_token,
        rate_limit_rps=cfg.notion_rate_limit_rps,
    )

    pages = notion.query_database(KB_DB_ID)

    found = []
    for page in pages:
        title_lower = page.title.strip().lower()
        for target in TARGETS:
            if target in title_lower:
                found.append((page.id, page.title))
                break

    for pid, title in found:
        try:
            print(f"{pid}  {title}")
        except UnicodeEncodeError:
            print(f"{pid}  {title.encode('ascii', errors='replace').decode('ascii')}")


if __name__ == "__main__":
    main()
