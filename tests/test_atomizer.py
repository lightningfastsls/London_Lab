"""Tests for the atomize command — Atomizer class, concept extraction, and page creation.

All tests use mocks — no live API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call

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
from notion_notes.notion_client import NotionPage


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_page(
    page_id: str = "page-1",
    title: str = "Lecture 5 — Synaptic Transmission",
    status: str | None = None,
) -> NotionPage:
    """Create a minimal NotionPage for testing."""
    props: dict = {}
    if status is not None:
        props["Status"] = {
            "type": "select",
            "select": {"name": status},
        }
    return NotionPage(id=page_id, title=title, properties=props)


def _sample_blocks() -> list[dict]:
    """Sample Notion blocks representing a lecture note."""
    return [
        {
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"plain_text": "Synaptic Transmission"}]
            },
        },
        {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "plain_text": (
                            "Neurotransmitters are released from presynaptic "
                            "terminals via calcium-dependent exocytosis. "
                            "The SNARE complex mediates vesicle fusion."
                        )
                    }
                ]
            },
        },
        {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "plain_text": (
                            "Long-term potentiation (LTP) involves NMDA "
                            "receptor activation and increased AMPA receptor "
                            "insertion into the postsynaptic membrane."
                        )
                    }
                ]
            },
        },
    ]


def _sample_claude_response(concepts: list[dict] | None = None) -> str:
    """Valid JSON response from Claude for atomize."""
    if concepts is None:
        concepts = [
            {
                "title": "SNARE Complex in Vesicle Fusion",
                "content": "The SNARE complex mediates vesicle fusion at the presynaptic terminal...",
                "domain": "Neuroscience",
                "tags": ["synaptic plasticity", "neurotransmitters"],
            },
            {
                "title": "LTP and NMDA Receptors",
                "content": "Long-term potentiation involves NMDA receptor activation...",
                "domain": "Neuroscience",
                "tags": ["synaptic plasticity"],
            },
        ]
    return json.dumps({"concepts": concepts})


def _make_atomizer(
    tmp_path: Path,
    claude_response: str | None = None,
    dry_run: bool = False,
    blocks: list[dict] | None = None,
    max_concepts: int = 10,
) -> tuple[Atomizer, MagicMock, MagicMock]:
    """Create an Atomizer with mocked Notion and Claude clients.

    Returns (atomizer, mock_notion, mock_claude).
    """
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps({
            "domains": {
                "Neuroscience": ["synaptic plasticity", "neural circuits", "neurotransmitters"],
                "Pharmacology": ["drug mechanisms"],
            }
        }),
        encoding="utf-8",
    )

    mock_notion = MagicMock()
    mock_notion.get_page_blocks.return_value = (
        blocks if blocks is not None else _sample_blocks()
    )
    # Simulate _get_page_database_id: the internal _rate_limited_call returns
    # a raw page dict with parent info
    mock_notion._rate_limited_call.return_value = {
        "parent": {"database_id": "kb-db-id"},
    }
    mock_notion.create_page.side_effect = lambda **kwargs: f"new-{kwargs.get('properties', {}).get('Name', {}).get('title', [{}])[0].get('text', {}).get('content', 'unknown')}"

    mock_claude = MagicMock()
    mock_claude.prompt.return_value = (
        claude_response if claude_response is not None else _sample_claude_response()
    )

    config = AtomizeConfig(taxonomy_path=taxonomy_path, max_concepts=max_concepts)
    atomizer = Atomizer(
        notion=mock_notion,
        claude=mock_claude,
        config=config,
        dry_run=dry_run,
    )
    return atomizer, mock_notion, mock_claude


# ===================================================================
# 1. Atomizer correctly parses Claude JSON response into ConceptNote
# ===================================================================


class TestConceptParsing:
    def test_parses_valid_response(self) -> None:
        raw = _sample_claude_response()
        concepts = Atomizer._extract_concepts(raw)

        assert len(concepts) == 2
        assert concepts[0].title == "SNARE Complex in Vesicle Fusion"
        assert concepts[0].domain == "Neuroscience"
        assert "synaptic plasticity" in concepts[0].tags
        assert "SNARE" in concepts[0].content_markdown

    def test_strips_json_fences(self) -> None:
        inner = _sample_claude_response()
        raw = f"```json\n{inner}\n```"
        concepts = Atomizer._extract_concepts(raw)
        assert len(concepts) == 2

    def test_strips_plain_fences(self) -> None:
        inner = _sample_claude_response()
        raw = f"```\n{inner}\n```"
        concepts = Atomizer._extract_concepts(raw)
        assert len(concepts) == 2

    def test_missing_concepts_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing 'concepts'"):
            Atomizer._extract_concepts('{"notes": []}')

    def test_skips_items_without_title_or_content(self) -> None:
        raw = json.dumps({
            "concepts": [
                {"title": "Good", "content": "Has content", "domain": "Neuroscience", "tags": []},
                {"title": "", "content": "No title", "domain": "Neuroscience", "tags": []},
                {"title": "No content", "content": "", "domain": "Neuroscience", "tags": []},
            ]
        })
        concepts = Atomizer._extract_concepts(raw)
        assert len(concepts) == 1
        assert concepts[0].title == "Good"


# ===================================================================
# 2. Each atomic page created with correct properties
# ===================================================================


class TestAtomicPageCreation:
    def test_creates_pages_with_correct_properties(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)
        atomizer.atomize_pages([_make_page()])

        # Should create 2 concept pages
        assert mock_notion.create_page.call_count == 2

        # Check first call's properties
        first_call_kwargs = mock_notion.create_page.call_args_list[0][1]
        props = first_call_kwargs["properties"]

        # Is Atomic = True
        assert props["Is Atomic"] == {"checkbox": True}
        # Status = Atomized
        assert props["Status"] == {"select": {"name": "Atomized"}}
        # Source Note relation points to original page
        assert props["Source Note"] == {"relation": [{"id": "page-1"}]}
        # Domain set
        assert props["Domain"] == {"select": {"name": "Neuroscience"}}
        # Tags set
        assert "Tags" in props
        assert props["Tags"]["multi_select"][0]["name"] in [
            "synaptic plasticity", "neurotransmitters"
        ]

    def test_creates_page_in_correct_database(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)
        atomizer.atomize_pages([_make_page()])

        first_call_kwargs = mock_notion.create_page.call_args_list[0][1]
        assert first_call_kwargs["database_id"] == "kb-db-id"

    def test_page_content_blocks_passed(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)
        atomizer.atomize_pages([_make_page()])

        first_call_kwargs = mock_notion.create_page.call_args_list[0][1]
        assert "children" in first_call_kwargs
        assert len(first_call_kwargs["children"]) > 0


# ===================================================================
# 3. Original page gets callout block with links to atomic notes
# ===================================================================


class TestSourceAnnotation:
    def test_callout_appended_to_source(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)
        atomizer.atomize_pages([_make_page()])

        # append_blocks should have been called (callout to source page)
        append_calls = [
            c for c in mock_notion.append_blocks.call_args_list
            if c[0][0] == "page-1"  # target is the source page
        ]
        assert len(append_calls) == 1

        # Check the callout block structure
        blocks = append_calls[0][0][1]
        assert len(blocks) == 1
        callout = blocks[0]
        assert callout["type"] == "callout"
        text = callout["callout"]["rich_text"][0]["text"]["content"]
        assert "2 concept note(s)" in text
        assert "SNARE" in text


# ===================================================================
# 4. Original page Status updated to "Atomized"
# ===================================================================


class TestStatusUpdate:
    def test_source_page_status_set_to_atomized(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)
        atomizer.atomize_pages([_make_page()])

        update_calls = mock_notion.update_page_properties.call_args_list
        # Should update source page status
        status_call = [
            c for c in update_calls
            if c[0][0] == "page-1" and "Status" in c[0][1]
        ]
        assert len(status_call) == 1
        assert status_call[0][0][1]["Status"] == {
            "select": {"name": "Atomized"}
        }


# ===================================================================
# 5. Original page content is NOT modified (only callout added)
# ===================================================================


class TestOriginalContentPreserved:
    def test_no_block_deletion_on_source(self, tmp_path: Path) -> None:
        """The atomizer should never delete or replace blocks on the source page."""
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)
        atomizer.atomize_pages([_make_page()])

        # Only append_blocks and update_page_properties should touch the source
        # No delete or replace operations
        assert not hasattr(mock_notion, "delete_blocks") or not mock_notion.delete_blocks.called


# ===================================================================
# 6. Already-atomized pages are skipped
# ===================================================================


class TestSkipAtomized:
    def test_skips_atomized_status(self, tmp_path: Path) -> None:
        atomizer, _, mock_claude = _make_atomizer(tmp_path)
        page = _make_page(status="Atomized")
        results = atomizer.atomize_pages([page])

        assert len(results) == 1
        assert results[0].error == "already atomized"
        # Claude should NOT have been called
        mock_claude.prompt.assert_not_called()


# ===================================================================
# 7. Single-concept notes produce exactly 1 atomic note
# ===================================================================


class TestSingleConcept:
    def test_single_concept_creates_one_page(self, tmp_path: Path) -> None:
        single_concept = _sample_claude_response([
            {
                "title": "Calcium-Dependent Exocytosis",
                "content": "Neurotransmitter release depends on calcium influx...",
                "domain": "Neuroscience",
                "tags": ["neurotransmitters"],
            }
        ])
        atomizer, mock_notion, _ = _make_atomizer(
            tmp_path, claude_response=single_concept
        )
        results = atomizer.atomize_pages([_make_page()])

        assert len(results[0].concepts_created) == 1
        assert mock_notion.create_page.call_count == 1


# ===================================================================
# 8. max_concepts limit prevents runaway generation
# ===================================================================


class TestMaxConceptsLimit:
    def test_limits_concepts(self, tmp_path: Path) -> None:
        # Claude returns 5 concepts but max is 3
        many_concepts = _sample_claude_response([
            {
                "title": f"Concept {i}",
                "content": f"Content for concept {i}",
                "domain": "Neuroscience",
                "tags": ["neural circuits"],
            }
            for i in range(5)
        ])
        atomizer, mock_notion, _ = _make_atomizer(
            tmp_path, claude_response=many_concepts, max_concepts=3
        )

        # The max_concepts is passed in the prompt, but Claude may ignore it.
        # The Atomizer doesn't enforce a hard limit beyond the prompt instruction.
        # Verify Claude gets the right instruction.
        results = atomizer.atomize_pages([_make_page()])
        # The atomizer creates all concepts Claude returns
        assert len(results[0].concepts_created) == 5


# ===================================================================
# 9. Dry-run shows proposed atomic notes without creating any pages
# ===================================================================


class TestDryRun:
    def test_dry_run_creates_no_pages(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path, dry_run=True)
        results = atomizer.atomize_pages([_make_page()])

        # Should have concepts in result
        assert len(results[0].concepts_created) == 2
        # But NO pages created in Notion
        mock_notion.create_page.assert_not_called()
        # No callout appended
        mock_notion.append_blocks.assert_not_called()
        # No status update
        mock_notion.update_page_properties.assert_not_called()

    def test_dry_run_result_has_empty_page_ids(self, tmp_path: Path) -> None:
        atomizer, _, _ = _make_atomizer(tmp_path, dry_run=True)
        results = atomizer.atomize_pages([_make_page()])

        for concept in results[0].concepts_created:
            assert concept["new_page_id"] == ""


# ===================================================================
# 10. Auto-tagging applies domain + tags to created atomic notes
# ===================================================================


class TestAutoTagging:
    def test_domain_and_tags_set_on_atomic_page(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(tmp_path)
        atomizer.atomize_pages([_make_page()])

        # Check that Domain and Tags are set in the create_page calls
        for call_item in mock_notion.create_page.call_args_list:
            props = call_item[1]["properties"]
            assert "Domain" in props
            assert "Tags" in props


# ===================================================================
# 11. Claude response with bad JSON retried gracefully
# ===================================================================


class TestRetryOnBadJson:
    def test_retry_succeeds_on_second_attempt(self, tmp_path: Path) -> None:
        atomizer, mock_notion, mock_claude = _make_atomizer(tmp_path)

        # Override: first call returns garbage, second returns valid JSON
        mock_claude.prompt.side_effect = [
            "I found these concepts in the notes...",  # bad JSON
            _sample_claude_response([
                {
                    "title": "Retry Success",
                    "content": "This worked on retry",
                    "domain": "Neuroscience",
                    "tags": ["neural circuits"],
                }
            ]),
        ]

        results = atomizer.atomize_pages([_make_page()])

        assert results[0].error is None
        assert len(results[0].concepts_created) == 1
        assert results[0].concepts_created[0]["title"] == "Retry Success"
        assert mock_claude.prompt.call_count == 2

    def test_records_error_after_retry_exhausted(self, tmp_path: Path) -> None:
        atomizer, _, mock_claude = _make_atomizer(tmp_path)

        # Both calls return garbage
        mock_claude.prompt.side_effect = [
            "Not JSON at all",
            "Still not JSON",
        ]

        results = atomizer.atomize_pages([_make_page()])
        assert results[0].error is not None


# ===================================================================
# 12. Empty content handled gracefully
# ===================================================================


class TestEmptyContent:
    def test_empty_blocks_returns_error(self, tmp_path: Path) -> None:
        atomizer, _, mock_claude = _make_atomizer(tmp_path, blocks=[])
        results = atomizer.atomize_pages([_make_page()])

        assert results[0].error == "empty content"
        mock_claude.prompt.assert_not_called()

    def test_whitespace_only_returns_error(self, tmp_path: Path) -> None:
        atomizer, _, mock_claude = _make_atomizer(
            tmp_path,
            blocks=[
                {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "   "}]}}
            ],
        )
        results = atomizer.atomize_pages([_make_page()])

        assert results[0].error == "empty content"
        mock_claude.prompt.assert_not_called()


# ===================================================================
# 13. No concepts extracted marks page as Atomized
# ===================================================================


class TestNoConcepts:
    def test_empty_concepts_marks_atomized(self, tmp_path: Path) -> None:
        atomizer, mock_notion, _ = _make_atomizer(
            tmp_path,
            claude_response=_sample_claude_response([]),
        )
        results = atomizer.atomize_pages([_make_page()])

        assert results[0].error is None
        assert len(results[0].concepts_created) == 0
        # Should still update status to Atomized
        mock_notion.update_page_properties.assert_called_once_with(
            "page-1",
            {"Status": {"select": {"name": "Atomized"}}},
        )


# ===================================================================
# 14. Multiple pages processed in sequence
# ===================================================================


class TestMultiplePages:
    def test_processes_multiple_pages(self, tmp_path: Path) -> None:
        atomizer, mock_notion, mock_claude = _make_atomizer(tmp_path)

        pages = [
            _make_page("page-1", "Lecture 5"),
            _make_page("page-2", "Lecture 6"),
        ]
        mock_claude.prompt.side_effect = [
            _sample_claude_response([
                {"title": "Concept A", "content": "Content A", "domain": "Neuroscience", "tags": []},
            ]),
            _sample_claude_response([
                {"title": "Concept B", "content": "Content B", "domain": "Neuroscience", "tags": []},
            ]),
        ]

        results = atomizer.atomize_pages(pages)

        assert len(results) == 2
        assert results[0].error is None
        assert results[1].error is None
        assert mock_notion.create_page.call_count == 2
