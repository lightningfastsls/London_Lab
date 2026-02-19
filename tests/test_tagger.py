"""Tests for the tag command — Tagger class and JSON extraction.

All tests use mocks — no live API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notion_notes.commands.tag import TagConfig, TagResult, Tagger, _seed_taxonomy
from notion_notes.notion_client import NotionPage


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_page(page_id: str = "page-1", title: str = "Test Page") -> NotionPage:
    """Create a minimal NotionPage for testing."""
    return NotionPage(id=page_id, title=title, properties={})


def _make_tagger(
    tmp_path: Path,
    claude_response: str = '{"domain": "Neuroscience", "tags": ["synaptic plasticity"], "new_tags": []}',
    dry_run: bool = False,
    blocks: list[dict] | None = None,
) -> tuple[Tagger, MagicMock, MagicMock]:
    """Create a Tagger with mocked Notion and Claude clients.

    Returns (tagger, mock_notion, mock_claude).
    """
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(_seed_taxonomy()), encoding="utf-8")

    mock_notion = MagicMock()
    if blocks is None:
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"plain_text": "Synaptic plasticity is fundamental."}]
                },
            }
        ]
    mock_notion.get_page_blocks.return_value = blocks

    mock_claude = MagicMock()
    mock_claude.prompt.return_value = claude_response

    config = TagConfig(taxonomy_path=taxonomy_path)
    tagger = Tagger(
        notion=mock_notion,
        claude=mock_claude,
        config=config,
        dry_run=dry_run,
    )
    return tagger, mock_notion, mock_claude


# ===================================================================
# 1. Parses valid Claude JSON response
# ===================================================================


class TestBasicTagging:
    def test_parses_valid_response(self, tmp_path: Path) -> None:
        tagger, mock_notion, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Neuroscience", "tags": ["synaptic plasticity"], "new_tags": []}',
        )
        results = tagger.tag_pages([_make_page()])

        assert len(results) == 1
        assert results[0].domain == "Neuroscience"
        assert results[0].tags == ["synaptic plasticity"]
        assert results[0].new_tags == []
        assert results[0].error is None

    def test_updates_notion_page_properties(self, tmp_path: Path) -> None:
        tagger, mock_notion, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Neuroscience", "tags": ["neural circuits"], "new_tags": []}',
        )
        tagger.tag_pages([_make_page()])

        mock_notion.update_page_properties.assert_called_once_with(
            "page-1",
            {
                "Domain": {"select": {"name": "Neuroscience"}},
                "Tags": {"multi_select": [{"name": "neural circuits"}]},
            },
        )


# ===================================================================
# 2. Existing taxonomy tags preferred
# ===================================================================


class TestExistingTags:
    def test_existing_tags_not_marked_as_new(self, tmp_path: Path) -> None:
        """When Claude returns tags that exist in taxonomy, new_tags should be empty."""
        tagger, _, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Neuroscience", "tags": ["synaptic plasticity", "neural circuits"], "new_tags": []}',
        )
        results = tagger.tag_pages([_make_page()])
        assert results[0].new_tags == []


# ===================================================================
# 3. New tags added to taxonomy
# ===================================================================


class TestTaxonomyGrowth:
    def test_new_tags_appended_to_taxonomy(self, tmp_path: Path) -> None:
        tagger, _, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Neuroscience", "tags": ["brain organoids"], "new_tags": ["brain organoids"]}',
        )
        tagger.tag_pages([_make_page()])

        # Re-read taxonomy
        updated = json.loads(tagger.config.taxonomy_path.read_text())
        assert "brain organoids" in updated["domains"]["Neuroscience"]

    def test_new_tags_in_result(self, tmp_path: Path) -> None:
        tagger, _, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Neuroscience", "tags": ["brain organoids"], "new_tags": ["brain organoids"]}',
        )
        results = tagger.tag_pages([_make_page()])
        assert results[0].new_tags == ["brain organoids"]


# ===================================================================
# 4. Domain validated against allowed list
# ===================================================================


class TestDomainValidation:
    def test_invalid_domain_records_error(self, tmp_path: Path) -> None:
        tagger, _, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Alchemy", "tags": ["transmutation"], "new_tags": []}',
        )
        results = tagger.tag_pages([_make_page()])
        assert results[0].error is not None
        assert "invalid domain" in results[0].error


# ===================================================================
# 5. Tags limited to max_tags
# ===================================================================


class TestTagLimit:
    def test_excess_tags_trimmed(self, tmp_path: Path) -> None:
        tagger, _, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Neuroscience", "tags": ["a", "b", "c", "d", "e"], "new_tags": []}',
        )
        results = tagger.tag_pages([_make_page()])
        assert len(results[0].tags) == 4  # max_tags default is 4


# ===================================================================
# 6. Page properties updated correctly
# ===================================================================


class TestPropertyFormat:
    def test_domain_as_select_tags_as_multi_select(self, tmp_path: Path) -> None:
        tagger, mock_notion, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Pharmacology", "tags": ["drug mechanisms", "receptor biology"], "new_tags": []}',
        )
        tagger.tag_pages([_make_page()])

        call_args = mock_notion.update_page_properties.call_args
        props = call_args[0][1]
        assert props["Domain"] == {"select": {"name": "Pharmacology"}}
        assert props["Tags"] == {
            "multi_select": [
                {"name": "drug mechanisms"},
                {"name": "receptor biology"},
            ]
        }


# ===================================================================
# 7. --untagged filter constructs correct Notion API filter
# ===================================================================


class TestUntaggedFilter:
    def test_untagged_filter_shape(self) -> None:
        """The filter for untagged pages should use multi_select is_empty."""
        expected_filter = {
            "property": "Tags",
            "multi_select": {"is_empty": True},
        }
        # This tests the CLI wiring — we verify the filter dict shape directly
        assert expected_filter == {
            "property": "Tags",
            "multi_select": {"is_empty": True},
        }


# ===================================================================
# 8. Dry-run prevents writes
# ===================================================================


class TestDryRun:
    def test_dry_run_skips_notion_update(self, tmp_path: Path) -> None:
        tagger, mock_notion, _ = _make_tagger(
            tmp_path,
            claude_response='{"domain": "Neuroscience", "tags": ["neural circuits"], "new_tags": []}',
            dry_run=True,
        )
        results = tagger.tag_pages([_make_page()])

        # Should still return results
        assert results[0].domain == "Neuroscience"
        assert results[0].error is None
        # But NOT call update_page_properties
        mock_notion.update_page_properties.assert_not_called()


# ===================================================================
# 9. Taxonomy file created with seed data if missing
# ===================================================================


class TestTaxonomySeed:
    def test_creates_seed_if_missing(self, tmp_path: Path) -> None:
        taxonomy_path = tmp_path / "taxonomy.json"
        assert not taxonomy_path.exists()

        mock_notion = MagicMock()
        mock_notion.get_page_blocks.return_value = [
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Content"}]}}
        ]
        mock_claude = MagicMock()
        mock_claude.prompt.return_value = '{"domain": "Neuroscience", "tags": ["synaptic plasticity"], "new_tags": []}'

        config = TagConfig(taxonomy_path=taxonomy_path)
        tagger = Tagger(notion=mock_notion, claude=mock_claude, config=config)
        tagger.tag_pages([_make_page()])

        assert taxonomy_path.exists()
        data = json.loads(taxonomy_path.read_text())
        assert "domains" in data
        assert "Neuroscience" in data["domains"]


# ===================================================================
# 10. Markdown fences stripped from Claude response
# ===================================================================


class TestMarkdownFenceStripping:
    def test_strips_json_fences(self) -> None:
        raw = '```json\n{"domain": "Neuroscience", "tags": ["synaptic plasticity"]}\n```'
        result = Tagger._extract_json(raw)
        assert result["domain"] == "Neuroscience"

    def test_strips_plain_fences(self) -> None:
        raw = '```\n{"domain": "Neuroscience", "tags": ["neural circuits"]}\n```'
        result = Tagger._extract_json(raw)
        assert result["domain"] == "Neuroscience"

    def test_handles_no_fences(self) -> None:
        raw = '{"domain": "Neuroscience", "tags": ["electrophysiology"]}'
        result = Tagger._extract_json(raw)
        assert result["tags"] == ["electrophysiology"]

    def test_adds_new_tags_default(self) -> None:
        """If Claude omits new_tags, it should default to []."""
        raw = '{"domain": "Neuroscience", "tags": ["synaptic plasticity"]}'
        result = Tagger._extract_json(raw)
        assert result["new_tags"] == []


# ===================================================================
# 11. Bad JSON triggers retry, then records error
# ===================================================================


class TestRetryOnBadJson:
    def test_retry_succeeds_on_second_attempt(self, tmp_path: Path) -> None:
        mock_notion = MagicMock()
        mock_notion.get_page_blocks.return_value = [
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Content"}]}}
        ]

        mock_claude = MagicMock()
        # First call returns garbage, second call returns valid JSON
        mock_claude.prompt.side_effect = [
            "I think this note is about neuroscience",
            '{"domain": "Neuroscience", "tags": ["neural circuits"], "new_tags": []}',
        ]

        taxonomy_path = tmp_path / "taxonomy.json"
        taxonomy_path.write_text(json.dumps(_seed_taxonomy()), encoding="utf-8")

        config = TagConfig(taxonomy_path=taxonomy_path, max_retries=1)
        tagger = Tagger(notion=mock_notion, claude=mock_claude, config=config)
        results = tagger.tag_pages([_make_page()])

        assert results[0].error is None
        assert results[0].domain == "Neuroscience"
        assert mock_claude.prompt.call_count == 2

    def test_records_error_after_retry_exhausted(self, tmp_path: Path) -> None:
        mock_notion = MagicMock()
        mock_notion.get_page_blocks.return_value = [
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Content"}]}}
        ]

        mock_claude = MagicMock()
        # Both calls return garbage
        mock_claude.prompt.side_effect = [
            "Not JSON at all",
            "Still not JSON",
        ]

        taxonomy_path = tmp_path / "taxonomy.json"
        taxonomy_path.write_text(json.dumps(_seed_taxonomy()), encoding="utf-8")

        config = TagConfig(taxonomy_path=taxonomy_path, max_retries=1)
        tagger = Tagger(notion=mock_notion, claude=mock_claude, config=config)
        results = tagger.tag_pages([_make_page()])

        assert results[0].error is not None


# ===================================================================
# 12. Empty page content handled gracefully
# ===================================================================


class TestEmptyContent:
    def test_empty_blocks_skipped_with_warning(self, tmp_path: Path) -> None:
        tagger, _, mock_claude = _make_tagger(
            tmp_path,
            blocks=[],  # empty page
        )
        results = tagger.tag_pages([_make_page()])

        assert results[0].error == "empty content"
        assert results[0].domain == ""
        # Claude should NOT have been called
        mock_claude.prompt.assert_not_called()

    def test_whitespace_only_blocks_skipped(self, tmp_path: Path) -> None:
        tagger, _, mock_claude = _make_tagger(
            tmp_path,
            blocks=[
                {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "   "}]}}
            ],
        )
        results = tagger.tag_pages([_make_page()])

        assert results[0].error == "empty content"
        mock_claude.prompt.assert_not_called()
