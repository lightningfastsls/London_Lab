"""Tests for the atomize command — Atomizer class, markdown_to_blocks, and concept parsing.

All tests use mocks — no live API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notion_notes.commands.atomize import (
    AtomizeConfig,
    AtomizeResult,
    Atomizer,
    ConceptNote,
)
from notion_notes.commands.tag import _seed_taxonomy
from notion_notes.notion_client import NotionClientWrapper, NotionPage


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_page(
    page_id: str = "page-1",
    title: str = "Lecture 5: Synaptic Plasticity",
    status: str | None = None,
) -> NotionPage:
    """Create a minimal NotionPage for testing."""
    props: dict = {}
    if status is not None:
        props["Status"] = {"type": "select", "select": {"name": status}}
    return NotionPage(id=page_id, title=title, properties=props)


def _sample_blocks() -> list[dict]:
    """Sample Notion blocks representing lecture content."""
    return [
        {
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"plain_text": "Synaptic Plasticity"}]
            },
        },
        {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "plain_text": (
                            "Long-term potentiation (LTP) is a persistent "
                            "strengthening of synapses based on recent activity."
                        )
                    }
                ]
            },
        },
        {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"plain_text": "Requires NMDA receptor activation"}]
            },
        },
    ]


def _sample_claude_response() -> str:
    """Valid JSON response from Claude for atomize."""
    return json.dumps({
        "concepts": [
            {
                "title": "Long-term potentiation (LTP)",
                "content": "LTP is a persistent strengthening of synapses...",
                "domain": "Neuroscience",
                "tags": ["synaptic plasticity"],
            },
            {
                "title": "NMDA receptor role in LTP",
                "content": "NMDA receptors act as coincidence detectors...",
                "domain": "Neuroscience",
                "tags": ["synaptic plasticity", "neurotransmitters"],
            },
        ]
    })


def _make_atomizer(
    tmp_path: Path,
    claude_response: str | None = None,
    dry_run: bool = False,
    blocks: list[dict] | None = None,
) -> tuple[Atomizer, MagicMock, MagicMock]:
    """Create an Atomizer with mocked Notion and Claude clients.

    Returns (atomizer, mock_notion, mock_claude).
    """
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(_seed_taxonomy()), encoding="utf-8")

    mock_notion = MagicMock()
    mock_notion.get_page_blocks.return_value = blocks if blocks is not None else _sample_blocks()
    # Mock _rate_limited_call for _get_page_database_id
    mock_notion._rate_limited_call.return_value = {
        "parent": {"database_id": "db-123"}
    }
    mock_notion.create_page.return_value = "new-page-id"

    mock_claude = MagicMock()
    mock_claude.prompt.return_value = claude_response or _sample_claude_response()

    config = AtomizeConfig(taxonomy_path=taxonomy_path)
    atomizer = Atomizer(
        notion=mock_notion,
        claude=mock_claude,
        config=config,
        dry_run=dry_run,
    )
    return atomizer, mock_notion, mock_claude


# ===================================================================
# 1. markdown_to_blocks — basic block types
# ===================================================================


class TestMarkdownToBlocks:
    def test_paragraph(self) -> None:
        blocks = NotionClientWrapper.markdown_to_blocks("Hello world")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"
        rt = blocks[0]["paragraph"]["rich_text"]
        assert rt[0]["text"]["content"] == "Hello world"

    def test_headings(self) -> None:
        md = "# H1\n## H2\n### H3"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        assert blocks[0]["type"] == "heading_1"
        assert blocks[1]["type"] == "heading_2"
        assert blocks[2]["type"] == "heading_3"

    def test_bulleted_list(self) -> None:
        md = "- Item A\n- Item B"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        assert all(b["type"] == "bulleted_list_item" for b in blocks)
        assert blocks[0]["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "Item A"

    def test_numbered_list(self) -> None:
        md = "1. First\n2. Second\n3. Third"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        assert all(b["type"] == "numbered_list_item" for b in blocks)
        assert blocks[0]["numbered_list_item"]["rich_text"][0]["text"]["content"] == "First"

    def test_code_block(self) -> None:
        md = "```python\nprint('hello')\n```"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["code"]["language"] == "python"
        assert blocks[0]["code"]["rich_text"][0]["text"]["content"] == "print('hello')"

    def test_divider(self) -> None:
        blocks = NotionClientWrapper.markdown_to_blocks("---")
        assert blocks[0]["type"] == "divider"

    def test_empty_lines_skipped(self) -> None:
        md = "Line 1\n\n\nLine 2"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        assert len(blocks) == 2

    def test_mixed_content(self) -> None:
        md = "# Title\nBody text\n- Bullet\n---"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        assert len(blocks) == 4
        assert blocks[0]["type"] == "heading_1"
        assert blocks[1]["type"] == "paragraph"
        assert blocks[2]["type"] == "bulleted_list_item"
        assert blocks[3]["type"] == "divider"


# ===================================================================
# 2. markdown_to_blocks round-trip with blocks_to_markdown
# ===================================================================


class TestMarkdownRoundTrip:
    def test_paragraph_round_trip(self) -> None:
        md = "Hello world"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        result = NotionClientWrapper.blocks_to_markdown(blocks)
        assert result == md

    def test_heading_round_trip(self) -> None:
        md = "# Heading One"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        result = NotionClientWrapper.blocks_to_markdown(blocks)
        assert result == md

    def test_bullet_round_trip(self) -> None:
        md = "- Item A\n- Item B"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        result = NotionClientWrapper.blocks_to_markdown(blocks)
        assert result == md

    def test_divider_round_trip(self) -> None:
        md = "---"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        result = NotionClientWrapper.blocks_to_markdown(blocks)
        assert result == md

    def test_code_round_trip(self) -> None:
        md = "```python\nprint('hi')\n```"
        blocks = NotionClientWrapper.markdown_to_blocks(md)
        result = NotionClientWrapper.blocks_to_markdown(blocks)
        assert result == md


# ===================================================================
# 3. _parse_concepts / _extract_concepts
# ===================================================================


class TestParseConcepts:
    def test_valid_json(self) -> None:
        raw = json.dumps({
            "concepts": [
                {
                    "title": "LTP",
                    "content": "Long-term potentiation...",
                    "domain": "Neuroscience",
                    "tags": ["synaptic plasticity"],
                }
            ]
        })
        result = Atomizer._extract_concepts(raw)
        assert len(result) == 1
        assert result[0].title == "LTP"
        assert result[0].content_markdown == "Long-term potentiation..."
        assert result[0].domain == "Neuroscience"

    def test_json_with_fences(self) -> None:
        inner = json.dumps({
            "concepts": [
                {"title": "NMDA", "content": "Receptor...", "domain": "Neuroscience", "tags": []}
            ]
        })
        raw = f"```json\n{inner}\n```"
        result = Atomizer._extract_concepts(raw)
        assert len(result) == 1
        assert result[0].title == "NMDA"

    def test_missing_concepts_key_raises(self) -> None:
        raw = '{"results": []}'
        with pytest.raises(ValueError, match="concepts"):
            Atomizer._extract_concepts(raw)

    def test_empty_concepts_array(self) -> None:
        raw = '{"concepts": []}'
        result = Atomizer._extract_concepts(raw)
        assert result == []

    def test_skips_concepts_without_title(self) -> None:
        raw = json.dumps({
            "concepts": [
                {"title": "", "content": "No title", "domain": "X", "tags": []},
                {"title": "Good", "content": "Has title", "domain": "Y", "tags": []},
            ]
        })
        result = Atomizer._extract_concepts(raw)
        assert len(result) == 1
        assert result[0].title == "Good"

    def test_skips_concepts_without_content(self) -> None:
        raw = json.dumps({
            "concepts": [
                {"title": "No Content", "content": "", "domain": "X", "tags": []},
            ]
        })
        result = Atomizer._extract_concepts(raw)
        assert result == []

    def test_missing_optional_fields_default(self) -> None:
        raw = json.dumps({
            "concepts": [
                {"title": "Minimal", "content": "Just the basics"}
            ]
        })
        result = Atomizer._extract_concepts(raw)
        assert len(result) == 1
        assert result[0].domain == ""
        assert result[0].tags == []


# ===================================================================
# 4. _build_atomize_prompt
# ===================================================================


class TestBuildPrompt:
    def test_includes_title_and_content(self, tmp_path: Path) -> None:
        atomizer, _, _ = _make_atomizer(tmp_path)
        taxonomy = atomizer._load_taxonomy()
        system, user_msg = atomizer._build_atomize_prompt(
            title="Test Note",
            content="Some content here",
            taxonomy=taxonomy,
        )
        assert "Test Note" in user_msg
        assert "Some content here" in user_msg

    def test_includes_taxonomy_domains(self, tmp_path: Path) -> None:
        atomizer, _, _ = _make_atomizer(tmp_path)
        taxonomy = atomizer._load_taxonomy()
        _, user_msg = atomizer._build_atomize_prompt(
            title="X", content="Y", taxonomy=taxonomy
        )
        assert "Neuroscience" in user_msg
        assert "Pharmacology" in user_msg

    def test_includes_taxonomy_tags(self, tmp_path: Path) -> None:
        atomizer, _, _ = _make_atomizer(tmp_path)
        taxonomy = atomizer._load_taxonomy()
        _, user_msg = atomizer._build_atomize_prompt(
            title="X", content="Y", taxonomy=taxonomy
        )
        assert "synaptic plasticity" in user_msg

    def test_truncates_long_content(self, tmp_path: Path) -> None:
        atomizer, _, _ = _make_atomizer(tmp_path)
        atomizer.config = AtomizeConfig(
            taxonomy_path=atomizer.config.taxonomy_path,
            max_content_chars=50,
        )
        taxonomy = atomizer._load_taxonomy()
        _, user_msg = atomizer._build_atomize_prompt(
            title="X", content="A" * 200, taxonomy=taxonomy
        )
        # Content should be truncated
        assert "A" * 200 not in user_msg
        assert "A" * 50 in user_msg


# ===================================================================
# 5. atomize_pages — skips already-atomized pages
# ===================================================================


class TestSkipAtomized:
    def test_skips_atomized_pages(self, tmp_path: Path) -> None:
        atomizer, _, mock_claude = _make_atomizer(tmp_path)
        page = _make_page(status="Atomized")
        results = atomizer.atomize_pages([page])

        assert len(results) == 1
        assert results[0].error == "already atomized"
        # Claude should NOT have been called
        mock_claude.prompt.assert_not_called()


# ===================================================================
# 6. _create_concept_page — builds correct Notion properties
# ===================================================================


class TestCreateConceptPage:
    def test_creates_page_with_correct_properties(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)

        concept = ConceptNote(
            title="LTP Mechanism",
            content_markdown="LTP involves NMDA receptor activation.",
            domain="Neuroscience",
            tags=["synaptic plasticity", "neural circuits"],
        )

        atomizer._create_concept_page(concept, "source-page-1", "db-123")

        mock_notion.create_page.assert_called_once()
        call_kwargs = mock_notion.create_page.call_args[1]

        # Check database
        assert call_kwargs["database_id"] == "db-123"

        # Check properties
        props = call_kwargs["properties"]
        assert props["Name"]["title"][0]["text"]["content"] == "LTP Mechanism"
        assert props["Is Atomic"]["checkbox"] is True
        assert props["Status"]["select"]["name"] == "Atomized"
        assert props["Source Note"]["relation"] == [{"id": "source-page-1"}]
        assert props["Domain"]["select"]["name"] == "Neuroscience"
        assert props["Tags"]["multi_select"] == [
            {"name": "synaptic plasticity"},
            {"name": "neural circuits"},
        ]

        # Check content blocks
        children = call_kwargs["children"]
        assert len(children) > 0

    def test_omits_domain_and_tags_when_empty(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)

        concept = ConceptNote(
            title="Minimal",
            content_markdown="Just text.",
            domain="",
            tags=[],
        )

        atomizer._create_concept_page(concept, "src-1", "db-1")

        props = mock_notion.create_page.call_args[1]["properties"]
        assert "Domain" not in props
        assert "Tags" not in props


# ===================================================================
# 7. _annotate_source — appends callout block
# ===================================================================


class TestAnnotateSource:
    def test_appends_callout_with_concept_titles(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)

        created = [
            {"title": "LTP", "new_page_id": "p1"},
            {"title": "NMDA Receptors", "new_page_id": "p2"},
        ]
        atomizer._annotate_source("source-page-1", created)

        mock_notion.append_blocks.assert_called_once()
        call_args = mock_notion.append_blocks.call_args
        page_id = call_args[0][0]
        blocks = call_args[0][1]

        assert page_id == "source-page-1"
        assert len(blocks) == 1
        assert blocks[0]["type"] == "callout"

        callout_text = blocks[0]["callout"]["rich_text"][0]["text"]["content"]
        assert "LTP" in callout_text
        assert "NMDA Receptors" in callout_text
        assert "2 concept note(s)" in callout_text


# ===================================================================
# 8. Dry-run mode — computes concepts but doesn't create pages
# ===================================================================


class TestDryRun:
    def test_dry_run_skips_page_creation(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path, dry_run=True)
        page = _make_page()
        results = atomizer.atomize_pages([page])

        assert len(results) == 1
        assert results[0].error is None
        assert len(results[0].concepts_created) == 2  # from sample response

        # Should NOT have created pages or annotated source
        mock_notion.create_page.assert_not_called()
        mock_notion.append_blocks.assert_not_called()
        mock_notion.update_page_properties.assert_not_called()


# ===================================================================
# 9. Empty page content handled gracefully
# ===================================================================


class TestEmptyContent:
    def test_empty_blocks_skipped(self, tmp_path: Path) -> None:
        atomizer, _, mock_claude = _make_atomizer(tmp_path, blocks=[])
        results = atomizer.atomize_pages([_make_page()])

        assert results[0].error == "empty content"
        mock_claude.prompt.assert_not_called()


# ===================================================================
# 10. Full atomize flow — end to end with mocks
# ===================================================================


class TestFullFlow:
    def test_atomizes_page_creates_concepts_and_annotates(
        self, tmp_path: Path
    ) -> None:
        atomizer, mock_notion, mock_claude = _make_atomizer(tmp_path)
        page = _make_page()
        results = atomizer.atomize_pages([page])

        assert len(results) == 1
        r = results[0]
        assert r.error is None
        assert len(r.concepts_created) == 2
        assert r.concepts_created[0]["title"] == "Long-term potentiation (LTP)"
        assert r.concepts_created[1]["title"] == "NMDA receptor role in LTP"

        # Should have created 2 pages
        assert mock_notion.create_page.call_count == 2

        # Should have annotated source
        mock_notion.append_blocks.assert_called_once()

        # Should have updated status to Atomized
        mock_notion.update_page_properties.assert_called()

    def test_zero_concepts_marks_atomized_without_children(
        self, tmp_path: Path
    ) -> None:
        response = '{"concepts": []}'
        atomizer, mock_notion, _ = _make_atomizer(
            tmp_path, claude_response=response
        )
        page = _make_page()
        results = atomizer.atomize_pages([page])

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].concepts_created == []

        # Should NOT create any pages
        mock_notion.create_page.assert_not_called()

        # Should still update status
        mock_notion.update_page_properties.assert_called_once()


# ===================================================================
# 11. Retry on bad JSON
# ===================================================================


class TestRetryOnBadJson:
    def test_retry_succeeds_on_second_attempt(self, tmp_path: Path) -> None:
        atomizer, mock_notion, mock_claude = _make_atomizer(tmp_path)

        # First call returns garbage, second returns valid JSON
        mock_claude.prompt.side_effect = [
            "I found some interesting concepts here",
            _sample_claude_response(),
        ]

        results = atomizer.atomize_pages([_make_page()])

        assert results[0].error is None
        assert len(results[0].concepts_created) == 2
        assert mock_claude.prompt.call_count == 2
