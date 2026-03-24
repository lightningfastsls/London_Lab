"""Check KB property names to find the exact Projects property name."""

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

    # Query just 1 page to see property names
    pages = notion.query_database(KB_DB_ID, page_size=1)
    if not pages:
        print("No pages found!")
        return

    page = pages[0]
    print(f"Page: {page.title}")
    print(f"\nAll properties:")
    for key in sorted(page.properties.keys()):
        prop = page.properties[key]
        prop_type = prop.get("type", "?")
        # Show repr to catch trailing spaces, emojis, etc.
        print(f"  {key!r:40s} type={prop_type}")

        # If it's a relation, show the relation details
        if prop_type == "relation":
            rel_ids = [r["id"] for r in prop.get("relation", [])]
            has_more = prop.get("has_more", False)
            print(f"    -> relation IDs: {rel_ids}, has_more={has_more}")


if __name__ == "__main__":
    main()
