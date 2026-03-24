"""Tests for the link command — Linker class, candidate filtering, and connection parsing.

All tests use mocks — no live API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notion_notes.commands.link import (
    Connection,
    LinkConfig,
    LinkResult,
    Linker,
)
from notion_notes.notion_client import NotionPage


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_page(
    page_id: str = "page-1",
    title: str = "LTP Mechanism",
    tags: list[str] | None = None,
    domain: str | None = None,
    status: str | None = None,
) -> NotionPage:
    """Create a NotionPage with Tags, Domain, and Status properties."""
    props: dict = {}
    if tags is not None:
        props["Tags"] = {
            "type": "multi_select",
            "multi_select": [{"name": t} for t in tags],
        }
    if domain is not None:
        props["Domain"] = {
            "type": "select",
            "select": {"name": domain},
        }
    if status is not None:
        props["Status"] = {
            "type": "select",
            "select": {"name": status},
        }
    return NotionPage(id=page_id, title=title, properties=props)


def _sample_blocks() -> list[dict]:
    """Sample Notion blocks for a target page."""
    return [
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
    ]


def _sample_claude_response(
    connections: list[dict] | None = None,
) -> str:
    """Valid JSON response from Claude for link."""
    if connections is None:
        connections = [
            {
                "candidate_title": "NMDA Receptors",
                "connection_type": "shares-mechanism-principle",
                "explanation": "Both describe NMDA receptor-dependent LTP mechanisms",
            },
        ]
    return json.dumps({"connections": connections})


def _make_linker(
    tmp_path: Path,
    claude_response: str | None = None,
    dry_run: bool = False,
    blocks: list[dict] | None = None,
    cache_pairs: list[list[str]] | None = None,
) -> tuple[Linker, MagicMock, MagicMock]:
    """Create a Linker with mocked Notion and Claude clients.

    Returns (linker, mock_notion, mock_claude).
    """
    cache_path = tmp_path / ".cache" / "link_pairs.json"
    if cache_pairs is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_pairs), encoding="utf-8")

    mock_notion = MagicMock()
    mock_notion.get_page_blocks.return_value = (
        blocks if blocks is not None else _sample_blocks()
    )

    mock_claude = MagicMock()
    mock_claude.prompt.return_value = (
        claude_response or _sample_claude_response()
    )

    config = LinkConfig(cache_path=cache_path)
    linker = Linker(
        notion=mock_notion,
        claude=mock_claude,
        config=config,
        dry_run=dry_run,
    )
    return linker, mock_notion, mock_claude


# ===================================================================
# 1. Candidate filtering finds pages with shared tags
# ===================================================================


class TestCandidateFiltering:
    def test_finds_pages_with_shared_tags(self, tmp_path: Path) -> None:
        linker, _, _ = _make_linker(tmp_path)

        target = _make_page(
            page_id="target",
            title="LTP",
            tags=["synaptic plasticity", "neural circuits"],
            domain="Neuroscience",
        )
        match = _make_page(
            page_id="match-1",
            title="NMDA Receptors",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        no_match = _make_page(
            page_id="no-match",
            title="Python Loops",
            tags=["programming"],
            domain="Computer Science",
        )

        all_pages = [target, match, no_match]
        candidates, skipped = linker._find_candidates(
            target, all_pages, set()
        )

        assert len(candidates) == 1
        assert candidates[0].id == "match-1"
        assert skipped == 0


# ===================================================================
# 2. Candidate filtering excludes self and cached pairs
# ===================================================================


class TestCandidateExclusions:
    def test_excludes_self(self, tmp_path: Path) -> None:
        linker, _, _ = _make_linker(tmp_path)

        page = _make_page(
            page_id="page-1",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        all_pages = [page]
        candidates, _ = linker._find_candidates(page, all_pages, set())
        assert len(candidates) == 0

    def test_excludes_cached_pairs(self, tmp_path: Path) -> None:
        linker, _, _ = _make_linker(tmp_path)

        target = _make_page(
            page_id="target",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        already_checked = _make_page(
            page_id="checked",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )

        cache = {("checked", "target")}  # already evaluated
        candidates, skipped = linker._find_candidates(
            target, [target, already_checked], cache
        )

        assert len(candidates) == 0
        assert skipped == 1


# ===================================================================
# 3. Pair key is canonical (sorted) so A-B == B-A
# ===================================================================


class TestPairKey:
    def test_pair_key_is_sorted(self) -> None:
        assert Linker._pair_key("bbb", "aaa") == ("aaa", "bbb")
        assert Linker._pair_key("aaa", "bbb") == ("aaa", "bbb")

    def test_pair_key_symmetric(self) -> None:
        assert Linker._pair_key("x", "y") == Linker._pair_key("y", "x")


# ===================================================================
# 4. Claude response parsed correctly into Connection objects
# ===================================================================


class TestParseConnections:
    def test_parses_valid_response(self, tmp_path: Path) -> None:
        linker, _, _ = _make_linker(tmp_path)

        raw = _sample_claude_response()
        candidate_map = {"NMDA Receptors": "nmda-page-id"}

        connections = linker._parse_connections(raw, "target-id", candidate_map)

        assert len(connections) == 1
        assert connections[0].page_a_id == "target-id"
        assert connections[0].page_b_id == "nmda-page-id"
        assert connections[0].connection_type == "shares-mechanism-principle"
        assert "NMDA" in connections[0].explanation

    def test_strips_markdown_fences(self, tmp_path: Path) -> None:
        linker, _, _ = _make_linker(tmp_path)

        inner = json.dumps({
            "connections": [
                {
                    "candidate_title": "Page B",
                    "connection_type": "prerequisite-builds-on",
                    "explanation": "Builds on foundational concept",
                }
            ]
        })
        raw = f"```json\n{inner}\n```"
        candidate_map = {"Page B": "b-id"}

        connections = linker._parse_connections(raw, "a-id", candidate_map)
        assert len(connections) == 1
        assert connections[0].connection_type == "prerequisite-builds-on"

    def test_skips_unknown_candidate_titles(self, tmp_path: Path) -> None:
        linker, _, _ = _make_linker(tmp_path)

        raw = _sample_claude_response([
            {
                "candidate_title": "Unknown Page",
                "connection_type": "shares-mechanism-principle",
                "explanation": "Some connection",
            }
        ])
        candidate_map = {"Known Page": "known-id"}

        connections = linker._parse_connections(raw, "target", candidate_map)
        assert len(connections) == 0


# ===================================================================
# 5. Related Notes relations created bidirectionally
# ===================================================================


class TestBidirectionalRelations:
    def test_creates_bidirectional_relations(self, tmp_path: Path) -> None:
        linker, mock_notion, _ = _make_linker(tmp_path)

        target = _make_page(
            page_id="target",
            title="LTP",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        candidate = _make_page(
            page_id="nmda-id",
            title="NMDA Receptors",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )

        all_pages = [target, candidate]
        linker.link_pages([target], all_pages)

        # Should call add_relation twice: A->B and B->A
        relation_calls = mock_notion.add_relation.call_args_list
        assert len(relation_calls) == 2

        # First call: target -> candidate
        assert relation_calls[0][0] == ("target", "Related Notes", ["nmda-id"])
        # Second call: candidate -> target
        assert relation_calls[1][0] == ("nmda-id", "Related Notes", ["target"])


# ===================================================================
# 6. Cache updated after evaluation (pair not re-evaluated)
# ===================================================================


class TestCacheUpdate:
    def test_cache_updated_after_evaluation(self, tmp_path: Path) -> None:
        linker, _, _ = _make_linker(tmp_path)

        target = _make_page(
            page_id="target",
            title="LTP",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        candidate = _make_page(
            page_id="nmda-id",
            title="NMDA Receptors",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )

        all_pages = [target, candidate]
        linker.link_pages([target], all_pages)

        # Verify cache was saved
        cache_data = json.loads(
            linker.config.cache_path.read_text(encoding="utf-8")
        )
        assert ["nmda-id", "target"] in cache_data

    def test_cached_pair_not_re_evaluated(self, tmp_path: Path) -> None:
        """Second run should skip already-cached pairs."""
        linker, _, mock_claude = _make_linker(
            tmp_path,
            cache_pairs=[["nmda-id", "target"]],
        )

        target = _make_page(
            page_id="target",
            title="LTP",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        candidate = _make_page(
            page_id="nmda-id",
            title="NMDA Receptors",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )

        all_pages = [target, candidate]
        results = linker.link_pages([target], all_pages)

        # Claude should NOT have been called (no candidates)
        mock_claude.prompt.assert_not_called()
        assert results[0].skipped_cached == 1
        assert results[0].candidates_evaluated == 0


# ===================================================================
# 7. Trivial connections filtered out (prompt instructs Claude to skip)
# ===================================================================


class TestTrivialConnections:
    def test_empty_connections_from_claude(self, tmp_path: Path) -> None:
        """Claude can return empty connections array for trivial matches."""
        linker, mock_notion, _ = _make_linker(
            tmp_path,
            claude_response=_sample_claude_response([]),
        )

        target = _make_page(
            page_id="target",
            title="LTP",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        candidate = _make_page(
            page_id="cand-1",
            title="Lecture 6 Recap",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )

        results = linker.link_pages([target], [target, candidate])

        assert len(results[0].connections_found) == 0
        # No relations should have been created
        mock_notion.add_relation.assert_not_called()


# ===================================================================
# 8. Max candidates limit respected
# ===================================================================


class TestMaxCandidates:
    def test_limits_candidates_to_max(self, tmp_path: Path) -> None:
        config = LinkConfig(
            max_candidates=2,
            cache_path=tmp_path / ".cache" / "link_pairs.json",
        )
        linker = Linker(
            notion=MagicMock(),
            claude=MagicMock(),
            config=config,
        )

        target = _make_page(
            page_id="target",
            tags=["tag-a", "tag-b"],
            domain="Neuroscience",
        )
        # Create 5 matching candidates
        all_pages = [target] + [
            _make_page(
                page_id=f"cand-{i}",
                title=f"Candidate {i}",
                tags=["tag-a"],
                domain="Neuroscience",
            )
            for i in range(5)
        ]

        candidates, _ = linker._find_candidates(target, all_pages, set())
        assert len(candidates) == 2


# ===================================================================
# 9. Dry-run shows proposed connections without creating relations
# ===================================================================


class TestDryRun:
    def test_dry_run_skips_relation_creation(self, tmp_path: Path) -> None:
        linker, mock_notion, _ = _make_linker(tmp_path, dry_run=True)

        target = _make_page(
            page_id="target",
            title="LTP",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        candidate = _make_page(
            page_id="nmda-id",
            title="NMDA Receptors",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )

        results = linker.link_pages([target], [target, candidate])

        # Should find connections
        assert len(results[0].connections_found) == 1
        # But NOT create relations or update status
        mock_notion.add_relation.assert_not_called()
        mock_notion.update_page_properties.assert_not_called()


# ===================================================================
# 10. Empty candidates list handled gracefully (no Claude call)
# ===================================================================


class TestEmptyCandidates:
    def test_no_claude_call_when_no_candidates(self, tmp_path: Path) -> None:
        linker, _, mock_claude = _make_linker(tmp_path)

        target = _make_page(
            page_id="target",
            title="Unique Topic",
            tags=["quantum-biology"],
            domain="Physics",
        )
        # No other pages share tags or domain
        other = _make_page(
            page_id="other",
            title="Python Loops",
            tags=["programming"],
            domain="Computer Science",
        )

        results = linker.link_pages([target], [target, other])

        mock_claude.prompt.assert_not_called()
        assert results[0].connections_found == []
        assert results[0].candidates_evaluated == 0


# ===================================================================
# 11. Already-linked pages can be re-linked (incremental)
# ===================================================================


class TestIncrementalLinking:
    def test_linked_pages_can_find_new_connections(
        self, tmp_path: Path
    ) -> None:
        """Pages with Status=Linked can still be re-linked to find new connections."""
        linker, _, _ = _make_linker(tmp_path)

        target = _make_page(
            page_id="target",
            title="LTP",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
            status="Linked",
        )
        # New page added since last link
        new_page = _make_page(
            page_id="new-page",
            title="NMDA Receptors",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )

        results = linker.link_pages([target], [target, new_page])

        # Should still find connections (not skipped)
        assert results[0].error is None
        assert results[0].candidates_evaluated == 1
        assert len(results[0].connections_found) == 1


# ===================================================================
# 12. Domain-only match (no tag overlap)
# ===================================================================


class TestDomainOnlyMatch:
    def test_same_domain_includes_candidate(self, tmp_path: Path) -> None:
        """Pages with same domain but no tag overlap are included."""
        linker, _, _ = _make_linker(tmp_path)

        target = _make_page(
            page_id="target",
            title="LTP",
            tags=["synaptic plasticity"],
            domain="Neuroscience",
        )
        same_domain = _make_page(
            page_id="domain-match",
            title="Motor Cortex",
            tags=["motor systems"],  # no tag overlap
            domain="Neuroscience",  # same domain
        )

        candidates, _ = linker._find_candidates(
            target, [target, same_domain], set()
        )

        assert len(candidates) == 1
        assert candidates[0].id == "domain-match"
