"""Find course 55008 in UNI Courses DB."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notion_notes.config import load_config
from notion_notes.notion_client import NotionClientWrapper

UNI_COURSES_DB_ID = "92fc943e-a4a3-48e5-97b0-196fc3e6d721"


def main() -> None:
    cfg = load_config(env_path=Path(".env"))
    notion = NotionClientWrapper(
        token=cfg.notion_token,
        rate_limit_rps=cfg.notion_rate_limit_rps,
    )

    courses = notion.query_database(UNI_COURSES_DB_ID)
    for c in courses:
        if "55008" in c.title:
            print(f"  {c.id}  {c.title}")


if __name__ == "__main__":
    main()
