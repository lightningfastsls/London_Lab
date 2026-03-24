"""Tests for the move command (Notes inbox -> Knowledge Base)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call

from notion_notes.commands.move import MoveResult, Mover
from notion_notes.notion_client import NotionPage


def _make_page(
    page_id: str = "src-page-1",
    title: str = "Test Note",
    tags: list[str] | None = None,
    evergreen: bool = False,
) -> NotionPage:
    props: dict = {
        "Name": {
            "type": "title",
            "title": [{"plain_text": title}],
        },
        "Status": {"type": "status", "status": {"name": "Not started"}},
    }
    if tags:
        props["Tags"] = {
            "type": "multi_select",
            "multi_select": [{"name": t} for t in tags],
        }
    if evergreen:
        props["Evergreen"] = {"type": "checkbox", "checkbox": True}
    return NotionPage(id=page_id, title=title, properties=props)


class TestMover(unittest.TestCase):
    def setUp(self) -> None:
        self.notion = MagicMock()
        self.mover = Mover(
            notion=self.notion,
            target_db_id="kb-db-id",
        )

    def test_move_creates_page_in_kb(self) -> None:
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = [
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Hello"}]}},
        ]
        self.notion.create_page.return_value = "new-page-id"

        result = self.mover.move_page("src-page-1")

        assert result.new_page_id == "new-page-id"
        assert result.error is None
        assert result.source_title == "Test Note"

        # Verify create_page was called with correct target DB
        create_call = self.notion.create_page.call_args
        assert create_call[1]["database_id"] == "kb-db-id"

    def test_move_sets_status_raw(self) -> None:
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []
        self.notion.create_page.return_value = "new-page-id"

        self.mover.move_page("src-page-1")

        create_call = self.notion.create_page.call_args
        props = create_call[1]["properties"]
        assert props["Status"] == {"select": {"name": "Raw"}}

    def test_move_sets_is_atomic_false(self) -> None:
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []
        self.notion.create_page.return_value = "new-page-id"

        self.mover.move_page("src-page-1")

        create_call = self.notion.create_page.call_args
        props = create_call[1]["properties"]
        assert props["Is Atomic"] == {"checkbox": False}

    def test_move_sets_title(self) -> None:
        page = _make_page(title="My Note Title")
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []
        self.notion.create_page.return_value = "new-page-id"

        self.mover.move_page("src-page-1")

        create_call = self.notion.create_page.call_args
        props = create_call[1]["properties"]
        assert props["Title"]["title"][0]["text"]["content"] == "My Note Title"

    def test_move_copies_tags(self) -> None:
        page = _make_page(tags=["ML", "USV"])
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []
        self.notion.create_page.return_value = "new-page-id"

        self.mover.move_page("src-page-1")

        create_call = self.notion.create_page.call_args
        props = create_call[1]["properties"]
        tag_names = [t["name"] for t in props["Tags"]["multi_select"]]
        assert tag_names == ["ML", "USV"]

    def test_move_copies_evergreen(self) -> None:
        page = _make_page(evergreen=True)
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []
        self.notion.create_page.return_value = "new-page-id"

        self.mover.move_page("src-page-1")

        create_call = self.notion.create_page.call_args
        props = create_call[1]["properties"]
        assert props["Evergreen"] == {"checkbox": True}

    def test_move_marks_source_page(self) -> None:
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []
        self.notion.create_page.return_value = "new-page-id"

        self.mover.move_page("src-page-1")

        # Verify append_blocks was called on the source page
        append_call = self.notion.append_blocks.call_args
        assert append_call[0][0] == "src-page-1"
        blocks = append_call[0][1]
        assert blocks[0]["type"] == "callout"

    def test_move_dry_run_creates_nothing(self) -> None:
        mover = Mover(self.notion, "kb-db-id", dry_run=True)
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []

        result = mover.move_page("src-page-1")

        assert result.new_page_id == "dry-run"
        assert result.error is None
        self.notion.create_page.assert_not_called()
        self.notion.append_blocks.assert_not_called()

    def test_move_handles_get_page_failure(self) -> None:
        self.notion.get_page.side_effect = Exception("API error")

        result = self.mover.move_page("bad-id")

        assert result.error is not None
        assert "Failed to read source page" in result.error
        self.notion.create_page.assert_not_called()

    def test_move_handles_create_page_failure(self) -> None:
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []
        self.notion.create_page.side_effect = Exception("Validation error")

        result = self.mover.move_page("src-page-1")

        assert result.error is not None
        assert "Failed to create KB page" in result.error

    def test_move_survives_source_marking_failure(self) -> None:
        """If marking the source page fails, the move should still succeed."""
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = []
        self.notion.create_page.return_value = "new-page-id"
        self.notion.append_blocks.side_effect = Exception("Permission denied")

        result = self.mover.move_page("src-page-1")

        assert result.new_page_id == "new-page-id"
        assert result.error is None  # Non-fatal

    def test_move_strips_block_metadata(self) -> None:
        """Content blocks should have read-only fields stripped."""
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = [
            {
                "id": "block-1",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Hello"}]},
                "has_children": False,
                "created_time": "2026-01-01T00:00:00Z",
            },
        ]
        self.notion.create_page.return_value = "new-page-id"

        self.mover.move_page("src-page-1")

        create_call = self.notion.create_page.call_args
        children = create_call[1]["children"]
        # Read-only fields should be stripped
        assert "id" not in children[0]
        assert "has_children" not in children[0]
        assert "created_time" not in children[0]
        # Content should remain
        assert children[0]["type"] == "paragraph"

    def test_move_converts_image_blocks(self) -> None:
        """Notion-hosted images should be converted to external format."""
        page = _make_page()
        self.notion.get_page.return_value = page
        self.notion.get_page_blocks.return_value = [
            {
                "id": "img-1",
                "type": "image",
                "image": {
                    "type": "file",
                    "file": {"url": "https://s3.amazonaws.com/signed-url"},
                    "caption": [],
                },
                "has_children": False,
            },
        ]
        self.notion.create_page.return_value = "new-page-id"

        self.mover.move_page("src-page-1")

        create_call = self.notion.create_page.call_args
        children = create_call[1]["children"]
        img = children[0]["image"]
        assert img["type"] == "external"
        assert img["external"]["url"] == "https://s3.amazonaws.com/signed-url"


class TestMoveCallout(unittest.TestCase):
    def test_callout_contains_page_mention(self) -> None:
        callout = Mover._moved_callout("new-id-123")
        assert callout["type"] == "callout"
        rich_text = callout["callout"]["rich_text"]
        # Should have text + mention
        assert len(rich_text) == 2
        assert rich_text[0]["text"]["content"] == "Moved to Knowledge Base. "
        assert rich_text[1]["mention"]["page"]["id"] == "new-id-123"


if __name__ == "__main__":
    unittest.main()
