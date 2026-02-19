"""Notion API wrapper with rate limiting, dry-run support, and helpers.

Wraps the official ``notion-client`` SDK to provide:
- Rate limiting (default 3 req/s) with exponential backoff on 429
- Dry-run mode that skips all write operations
- Helper methods for converting blocks to markdown and extracting properties
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from notion_client import APIResponseError, Client

logger = logging.getLogger(__name__)


@dataclass
class NotionPage:
    """Lightweight representation of a Notion page."""

    id: str
    title: str
    properties: dict[str, Any]
    content_blocks: list[dict] = field(default_factory=list)
    url: str = ""
    created_time: str = ""
    last_edited_time: str = ""


class NotionClientWrapper:
    """Wrapper around the official Notion SDK with rate limiting and helpers.

    Parameters
    ----------
    token : str
        Notion integration token.
    dry_run : bool
        When True, write operations log what *would* happen but do nothing.
    rate_limit_rps : float
        Maximum requests per second (Notion's limit is 3).
    """

    def __init__(
        self,
        token: str,
        dry_run: bool = False,
        rate_limit_rps: float = 3.0,
    ) -> None:
        self.client = Client(auth=token, notion_version="2022-06-28")
        self.dry_run = dry_run
        self._min_interval = 1.0 / rate_limit_rps
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_page(self, page_id: str) -> NotionPage:
        """Retrieve a single page by ID (metadata only, no blocks)."""
        raw = self._rate_limited_call(self.client.pages.retrieve, page_id=page_id)
        return self._raw_page_to_notion_page(raw)

    def get_page_blocks(self, page_id: str) -> list[dict]:
        """Retrieve all content blocks for a page (handles pagination)."""
        blocks: list[dict] = []
        cursor = None
        while True:
            kwargs: dict[str, Any] = {"block_id": page_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor
            resp = self._rate_limited_call(
                self.client.blocks.children.list, **kwargs
            )
            blocks.extend(resp["results"])
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
        return blocks

    def query_database(
        self,
        database_id: str,
        filter: dict | None = None,
        sorts: list | None = None,
        page_size: int = 100,
    ) -> list[NotionPage]:
        """Query a database with optional filter and sort (handles pagination)."""
        pages: list[NotionPage] = []
        cursor = None
        while True:
            body: dict[str, Any] = {"page_size": page_size}
            if filter:
                body["filter"] = filter
            if sorts:
                body["sorts"] = sorts
            if cursor:
                body["start_cursor"] = cursor
            resp = self._rate_limited_call(
                self.client.request,
                path=f"databases/{database_id}/query",
                method="POST",
                body=body,
            )
            for raw in resp["results"]:
                pages.append(self._raw_page_to_notion_page(raw))
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
        return pages

    def search_database(self, database_id: str, query: str) -> list[NotionPage]:
        """Search within a database by text query."""
        resp = self._rate_limited_call(
            self.client.search,
            query=query,
            filter={"value": "page", "property": "object"},
        )
        pages: list[NotionPage] = []
        for raw in resp["results"]:
            # Only include pages from the target database
            if raw.get("parent", {}).get("database_id", "").replace("-", "") == database_id.replace("-", ""):
                pages.append(self._raw_page_to_notion_page(raw))
        return pages

    # ------------------------------------------------------------------
    # Write operations (respect dry_run)
    # ------------------------------------------------------------------

    def create_page(
        self,
        database_id: str,
        properties: dict,
        children: list[dict] | None = None,
    ) -> str | None:
        """Create a new page in a database. Returns the new page ID, or None in dry-run."""
        if self.dry_run:
            logger.info("[DRY RUN] Would create page in database %s", database_id)
            return None
        kwargs: dict[str, Any] = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }
        if children:
            kwargs["children"] = children
        resp = self._rate_limited_call(self.client.pages.create, **kwargs)
        return resp["id"]

    def update_page_properties(self, page_id: str, properties: dict) -> None:
        """Update properties on an existing page."""
        if self.dry_run:
            logger.info("[DRY RUN] Would update properties on page %s", page_id)
            return
        self._rate_limited_call(
            self.client.pages.update, page_id=page_id, properties=properties
        )

    def append_blocks(self, page_id: str, children: list[dict]) -> None:
        """Append content blocks to a page."""
        if self.dry_run:
            logger.info("[DRY RUN] Would append %d blocks to page %s", len(children), page_id)
            return
        self._rate_limited_call(
            self.client.blocks.children.append,
            block_id=page_id,
            children=children,
        )

    def add_relation(
        self,
        page_id: str,
        property_name: str,
        related_page_ids: list[str],
    ) -> None:
        """Add relation entries to a page's relation property."""
        if self.dry_run:
            logger.info(
                "[DRY RUN] Would add %d relations to %s.%s",
                len(related_page_ids), page_id, property_name,
            )
            return
        self.update_page_properties(
            page_id,
            {property_name: {"relation": [{"id": rid} for rid in related_page_ids]}},
        )

    # ------------------------------------------------------------------
    # Helpers (pure / static)
    # ------------------------------------------------------------------

    @staticmethod
    def blocks_to_markdown(blocks: list[dict]) -> str:
        """Convert Notion blocks to a markdown string.

        Supports: paragraph, heading_1/2/3, bulleted_list_item,
        numbered_list_item, code, divider. Unknown types produce
        ``[unsupported: <type>]``.
        """
        lines: list[str] = []
        numbered_counter = 0

        for block in blocks:
            block_type = block.get("type", "")

            if block_type == "paragraph":
                text = _rich_text_to_plain(block.get("paragraph", {}).get("rich_text", []))
                lines.append(text)
                numbered_counter = 0

            elif block_type == "heading_1":
                text = _rich_text_to_plain(block.get("heading_1", {}).get("rich_text", []))
                lines.append(f"# {text}")
                numbered_counter = 0

            elif block_type == "heading_2":
                text = _rich_text_to_plain(block.get("heading_2", {}).get("rich_text", []))
                lines.append(f"## {text}")
                numbered_counter = 0

            elif block_type == "heading_3":
                text = _rich_text_to_plain(block.get("heading_3", {}).get("rich_text", []))
                lines.append(f"### {text}")
                numbered_counter = 0

            elif block_type == "bulleted_list_item":
                text = _rich_text_to_plain(
                    block.get("bulleted_list_item", {}).get("rich_text", [])
                )
                lines.append(f"- {text}")
                numbered_counter = 0

            elif block_type == "numbered_list_item":
                numbered_counter += 1
                text = _rich_text_to_plain(
                    block.get("numbered_list_item", {}).get("rich_text", [])
                )
                lines.append(f"{numbered_counter}. {text}")

            elif block_type == "code":
                code_data = block.get("code", {})
                text = _rich_text_to_plain(code_data.get("rich_text", []))
                language = code_data.get("language", "")
                lines.append(f"```{language}")
                lines.append(text)
                lines.append("```")
                numbered_counter = 0

            elif block_type == "divider":
                lines.append("---")
                numbered_counter = 0

            else:
                lines.append(f"[unsupported: {block_type}]")
                numbered_counter = 0

        return "\n".join(lines)

    @staticmethod
    def markdown_to_blocks(markdown: str) -> list[dict]:
        """Convert a markdown string to Notion block dicts.

        Inverse of :meth:`blocks_to_markdown`. Supports: headings (1-3),
        bulleted and numbered lists, fenced code blocks, dividers, and
        paragraphs. Empty lines are skipped.
        """
        blocks: list[dict] = []
        lines = markdown.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]

            # Fenced code block
            if line.startswith("```"):
                language = line[3:].strip()
                code_lines: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                # Skip closing fence
                if i < len(lines):
                    i += 1
                code_text = "\n".join(code_lines)
                block: dict = {
                    "type": "code",
                    "code": {
                        "rich_text": [
                            {"type": "text", "text": {"content": code_text}, "plain_text": code_text}
                        ],
                        "language": language or "plain text",
                    },
                }
                blocks.append(block)
                continue

            # Divider
            if line.strip() == "---":
                blocks.append({"type": "divider", "divider": {}})
                i += 1
                continue

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Headings
            if line.startswith("### "):
                blocks.append(_text_block("heading_3", line[4:]))
            elif line.startswith("## "):
                blocks.append(_text_block("heading_2", line[3:]))
            elif line.startswith("# "):
                blocks.append(_text_block("heading_1", line[2:]))
            # Bulleted list
            elif line.startswith("- "):
                blocks.append(_text_block("bulleted_list_item", line[2:]))
            # Numbered list (e.g. "1. ", "12. ")
            elif _is_numbered_list(line):
                text = line.split(". ", 1)[1] if ". " in line else line
                blocks.append(_text_block("numbered_list_item", text))
            # Paragraph (everything else)
            else:
                blocks.append(_text_block("paragraph", line))

            i += 1

        return blocks

    @staticmethod
    def extract_title(page_properties: dict) -> str:
        """Extract the title string from a page's properties dict.

        Scans properties for the one with ``type: "title"`` rather than
        assuming the key is "Name".
        """
        for prop_value in page_properties.values():
            if prop_value.get("type") == "title":
                return _rich_text_to_plain(prop_value.get("title", []))
        return ""

    @staticmethod
    def extract_property(page_properties: dict, property_name: str) -> Any:
        """Extract a typed value from a page property.

        Handles: multi_select, select, checkbox, date, relation, rich_text,
        number, url. Returns None for unrecognised types.
        """
        prop = page_properties.get(property_name)
        if prop is None:
            return None

        prop_type = prop.get("type", "")

        if prop_type == "multi_select":
            return [opt["name"] for opt in prop.get("multi_select", [])]

        if prop_type == "select":
            sel = prop.get("select")
            return sel["name"] if sel else None

        if prop_type == "checkbox":
            return prop.get("checkbox", False)

        if prop_type == "date":
            date_obj = prop.get("date")
            return date_obj.get("start") if date_obj else None

        if prop_type == "relation":
            return [rel["id"] for rel in prop.get("relation", [])]

        if prop_type == "rich_text":
            return _rich_text_to_plain(prop.get("rich_text", []))

        if prop_type == "number":
            return prop.get("number")

        if prop_type == "url":
            return prop.get("url")

        if prop_type == "title":
            return _rich_text_to_plain(prop.get("title", []))

        return None

    # ------------------------------------------------------------------
    # Rate limiting internals
    # ------------------------------------------------------------------

    def _rate_limited_call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Call *func* while respecting the rate limit.

        Sleeps if needed to maintain ``rate_limit_rps``. On HTTP 429 responses
        uses exponential backoff (1s, 2s, 4s) with up to 3 retries.
        """
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                self._last_request_time = time.monotonic()
                return func(*args, **kwargs)
            except APIResponseError as exc:
                if exc.status == 429 and attempt < max_retries:
                    backoff = 2**attempt  # 1, 2, 4 seconds
                    logger.warning(
                        "Rate limited (429), backing off %ds (attempt %d/%d)",
                        backoff, attempt + 1, max_retries,
                    )
                    time.sleep(backoff)
                else:
                    raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_page_to_notion_page(self, raw: dict) -> NotionPage:
        """Convert a raw Notion API page response to a NotionPage."""
        properties = raw.get("properties", {})
        return NotionPage(
            id=raw["id"],
            title=self.extract_title(properties),
            properties=properties,
            url=raw.get("url", ""),
            created_time=raw.get("created_time", ""),
            last_edited_time=raw.get("last_edited_time", ""),
        )


def _rich_text_to_plain(rich_text: list[dict]) -> str:
    """Concatenate rich_text array items into a plain string."""
    return "".join(item.get("plain_text", "") for item in rich_text)


def _text_block(block_type: str, text: str) -> dict:
    """Build a Notion block dict with a single plain-text rich_text entry."""
    return {
        "type": block_type,
        block_type: {
            "rich_text": [
                {"type": "text", "text": {"content": text}, "plain_text": text}
            ],
        },
    }


def _is_numbered_list(line: str) -> bool:
    """Check if a line matches a numbered list pattern like '1. ' or '12. '."""
    import re
    return bool(re.match(r"^\d+\.\s", line))
