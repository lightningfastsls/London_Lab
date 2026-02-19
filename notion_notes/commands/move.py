"""Move command — transfer a page from Notes inbox to Knowledge Base.

Reads a page from the Notes database, creates a copy in the Knowledge Base
with mapped properties and content blocks, and marks the source page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from notion_notes.migration import NoteMigrator
from notion_notes.notion_client import NotionClientWrapper, NotionPage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class MoveResult:
    """Result of moving a single page."""

    source_page_id: str
    source_title: str
    new_page_id: str | None = None
    error: str | None = None


# ------------------------------------------------------------------
# Mover
# ------------------------------------------------------------------


class Mover:
    """Transfer a page from Notes inbox to Knowledge Base."""

    def __init__(
        self,
        notion: NotionClientWrapper,
        target_db_id: str,
        dry_run: bool = False,
    ) -> None:
        self._notion = notion
        self._target_db_id = target_db_id
        self._dry_run = dry_run

    def move_page(self, page_id: str) -> MoveResult:
        """Move a single page from Notes DB to Knowledge Base.

        Steps:
        1. Read the source page (properties + content blocks).
        2. Build target properties (Title, Status=Raw, Is Atomic=False).
        3. Clean content blocks (strip read-only fields, convert images).
        4. Create new page in KB.
        5. Append a "Moved to KB" callout to the source page.
        6. Return the new page ID.
        """
        try:
            page = self._notion.get_page(page_id)
        except Exception as e:
            return MoveResult(
                source_page_id=page_id,
                source_title="(unknown)",
                error=f"Failed to read source page: {e}",
            )

        title = page.title or "(untitled)"
        logger.info("Moving page: %s", title)

        # Build target properties — minimal mapping for inbox pages
        target_props = self._build_target_properties(page)

        # Read and clean content blocks (recursive for tables, toggles, etc.)
        try:
            blocks = self._fetch_blocks_recursive(page_id)
        except Exception as e:
            return MoveResult(
                source_page_id=page_id,
                source_title=title,
                error=f"Failed to read blocks: {e}",
            )

        cleaned_blocks = [
            NoteMigrator._strip_block_metadata(b) for b in blocks
        ]

        if self._dry_run:
            logger.info("[DRY RUN] Would create page '%s' in KB (%d blocks)", title, len(cleaned_blocks))
            return MoveResult(
                source_page_id=page_id,
                source_title=title,
                new_page_id="dry-run",
            )

        # Create page in KB (first 100 blocks, then append rest in chunks)
        try:
            initial_blocks = cleaned_blocks[:100] if cleaned_blocks else None
            new_id = self._notion.create_page(
                database_id=self._target_db_id,
                properties=target_props,
                children=initial_blocks,
            )
        except Exception as e:
            return MoveResult(
                source_page_id=page_id,
                source_title=title,
                error=f"Failed to create KB page: {e}",
            )

        # Append remaining blocks in chunks of 100
        if len(cleaned_blocks) > 100:
            try:
                for i in range(100, len(cleaned_blocks), 100):
                    chunk = cleaned_blocks[i : i + 100]
                    self._notion.append_blocks(new_id, chunk)
            except Exception as e:
                logger.warning(
                    "Page created but failed to append all blocks: %s", e
                )

        # Mark source page with callout
        try:
            self._notion.append_blocks(page_id, [self._moved_callout(new_id)])
        except Exception as e:
            # Non-fatal: the page was moved, just the marker failed
            logger.warning(
                "Page moved but failed to mark source: %s", e
            )

        logger.info("Moved '%s' -> %s", title, new_id)
        return MoveResult(
            source_page_id=page_id,
            source_title=title,
            new_page_id=new_id,
        )

    def _fetch_blocks_recursive(self, page_id: str) -> list[dict]:
        """Fetch all blocks for a page, recursively fetching children.

        Notion's blocks.children.list only returns top-level blocks.
        Blocks with has_children=True (tables, toggles, columns) need
        their children fetched and nested under the parent block's type key.
        """
        blocks = self._notion.get_page_blocks(page_id)
        result: list[dict] = []
        for block in blocks:
            result.append(block)
            if block.get("has_children"):
                children = self._fetch_blocks_recursive(block["id"])
                block_type = block.get("type", "")
                if block_type in block and isinstance(block[block_type], dict):
                    block[block_type]["children"] = [
                        NoteMigrator._strip_block_metadata(c) for c in children
                    ]
        return result

    def _build_target_properties(self, page: NotionPage) -> dict[str, Any]:
        """Build KB properties for a moved page.

        Copies title and sets defaults. Tags/domain are left empty
        for the tag command to fill in later.
        """
        props: dict[str, Any] = {}

        # Title (KB uses "Title" as the title property name)
        props["Title"] = {
            "title": [{"text": {"content": page.title or "(untitled)"}}],
        }

        # Status = Raw (ready for processing)
        props["Status"] = {"select": {"name": "Raw"}}

        # Is Atomic = False (this is a raw class note, not an extracted concept)
        props["Is Atomic"] = {"checkbox": False}

        # Copy Tags if present on source
        tags = NotionClientWrapper.extract_property(
            page.properties, "Tags"
        )
        if tags and isinstance(tags, list):
            props["Tags"] = {
                "multi_select": [{"name": t} for t in tags],
            }

        # Copy Evergreen if present
        evergreen = NotionClientWrapper.extract_property(
            page.properties, "Evergreen"
        )
        if evergreen is not None:
            props["Evergreen"] = {"checkbox": bool(evergreen)}

        # Copy Projects relation (Notes DB: "🔬 Projects" -> KB: "Projects")
        project_ids = NotionClientWrapper.extract_property(
            page.properties, "\U0001f52c Projects"
        )
        if project_ids and isinstance(project_ids, list):
            props["Projects"] = {
                "relation": [{"id": pid} for pid in project_ids],
            }

        return props

    @staticmethod
    def _moved_callout(new_page_id: str) -> dict:
        """Build a callout block marking a page as moved to KB."""
        return {
            "type": "callout",
            "callout": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "Moved to Knowledge Base. ",
                        },
                    },
                    {
                        "type": "mention",
                        "mention": {
                            "type": "page",
                            "page": {"id": new_page_id},
                        },
                    },
                ],
                "icon": {"type": "emoji", "emoji": "\u27a1\ufe0f"},
                "color": "blue_background",
            },
        }
