"""Find the exact Lecture 3 (course 55008) and Lecture 12 (plain title)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notion_notes.config import load_config
from notion_notes.notion_client import NotionClientWrapper

KB_DB_ID = "30b2bc599f6e8032b337fbda2c975dda"


def main() -> None:
    cfg = load_config(env_path=Path(".env"))
    notion = NotionClientWrapper(
        token=cfg.notion_token,
        rate_limit_rps=cfg.notion_rate_limit_rps,
    )

    pages = notion.query_database(KB_DB_ID)

    # Find all Lecture 3 pages and show their UNI Course relation
    print("=== Lecture 3 candidates ===")
    for page in pages:
        title = page.title.strip()
        if title.lower() == "lecture 3":
            course = NotionClientWrapper.extract_property(page.properties, "UNI Course  ")
            try:
                print(f"  {page.id}  title={title!r}  course_ids={course}")
            except UnicodeEncodeError:
                print(f"  {page.id}  title=Lecture 3  course_ids={course}")

    # Lecture 12 - exact match only
    print("\n=== Lecture 12 (exact) ===")
    for page in pages:
        title = page.title.strip()
        if title.lower() == "lecture 12":
            try:
                print(f"  {page.id}  title={title!r}")
            except UnicodeEncodeError:
                print(f"  {page.id}  title=Lecture 12")


if __name__ == "__main__":
    main()
