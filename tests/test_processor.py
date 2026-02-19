"""Tests for the process command (tag -> atomize -> link orchestrator)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from notion_notes.commands.atomize import AtomizeResult
from notion_notes.commands.link import Connection, LinkResult
from notion_notes.commands.process import ProcessConfig, ProcessReport, Processor
from notion_notes.commands.tag import TagResult
from notion_notes.notion_client import NotionPage


def _make_page(page_id: str, title: str, status: str = "Raw") -> NotionPage:
    return NotionPage(
        id=page_id,
        title=title,
        properties={
            "Status": {"type": "select", "select": {"name": status}},
            "Tags": {"type": "multi_select", "multi_select": []},
        },
    )


class TestProcessReport(unittest.TestCase):
    def test_summary_empty(self) -> None:
        report = ProcessReport()
        text = report.summary()
        assert "0 pages input" in text

    def test_summary_with_results(self) -> None:
        report = ProcessReport(
            pages_input=5,
            tags_applied=4,
            tag_errors=1,
            pages_atomized=3,
            concepts_created=7,
            atomize_errors=0,
            pages_linked=5,
            connections_found=2,
            link_errors=0,
            errors=["tag:Page A: timeout"],
        )
        text = report.summary()
        assert "5 pages input" in text
        assert "4 tagged" in text
        assert "1 errors" in text
        assert "7 concepts" in text
        assert "2 connections" in text

    def test_summary_truncates_errors(self) -> None:
        report = ProcessReport(errors=[f"error-{i}" for i in range(10)])
        text = report.summary()
        assert "... and 5 more" in text


class TestProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.notion = MagicMock()
        self.claude = MagicMock()
        self.config = ProcessConfig(kb_database_id="kb-db-id")
        self.pages = [
            _make_page("p1", "Page 1"),
            _make_page("p2", "Page 2"),
        ]

    def test_empty_pages_returns_empty_report(self) -> None:
        proc = Processor(self.notion, self.claude, self.config)
        report = proc.process([])
        assert report.pages_input == 0
        assert report.tags_applied == 0

    @patch("notion_notes.commands.process.Tagger")
    @patch("notion_notes.commands.process.Atomizer")
    @patch("notion_notes.commands.process.Linker")
    def test_runs_tag_atomize_link_in_sequence(
        self, mock_linker_cls, mock_atomizer_cls, mock_tagger_cls
    ) -> None:
        # Setup mocks
        mock_tagger = mock_tagger_cls.return_value
        mock_tagger.tag_pages.return_value = [
            TagResult(page_id="p1", title="Page 1", domain="CS", tags=["ML"], new_tags=[]),
            TagResult(page_id="p2", title="Page 2", domain="Bio", tags=["USV"], new_tags=["USV"]),
        ]

        mock_atomizer = mock_atomizer_cls.return_value
        mock_atomizer.atomize_pages.return_value = [
            AtomizeResult(
                page_id="p1", title="Page 1",
                concepts_created=[{"title": "Concept A", "id": "c1"}],
            ),
            AtomizeResult(page_id="p2", title="Page 2", concepts_created=[]),
        ]

        self.notion.query_database.return_value = self.pages  # for link phase

        mock_linker = mock_linker_cls.return_value
        mock_linker.link_pages.return_value = [
            LinkResult(
                page_id="p1", page_title="Page 1",
                connections_found=[
                    Connection(page_a_id="p1", page_b_id="p2",
                               connection_type="shares-mechanism-principle",
                               explanation="both about ML"),
                ],
                candidates_evaluated=1, skipped_cached=0,
            ),
            LinkResult(page_id="p2", page_title="Page 2",
                       candidates_evaluated=0, skipped_cached=1),
        ]

        proc = Processor(self.notion, self.claude, self.config)
        report = proc.process(self.pages)

        # Verify all three phases ran
        mock_tagger.tag_pages.assert_called_once_with(self.pages)
        mock_atomizer.atomize_pages.assert_called_once_with(self.pages)
        mock_linker.link_pages.assert_called_once()

        # Verify report counts
        assert report.pages_input == 2
        assert report.tags_applied == 2
        assert report.tag_errors == 0
        assert report.pages_atomized == 2  # Both succeeded (0 concepts ≠ error)
        assert report.concepts_created == 1
        assert report.pages_linked == 2
        assert report.connections_found == 1

    @patch("notion_notes.commands.process.Tagger")
    @patch("notion_notes.commands.process.Atomizer")
    @patch("notion_notes.commands.process.Linker")
    def test_collects_errors_across_phases(
        self, mock_linker_cls, mock_atomizer_cls, mock_tagger_cls
    ) -> None:
        mock_tagger_cls.return_value.tag_pages.return_value = [
            TagResult(page_id="p1", title="Page 1", domain="", tags=[], new_tags=[],
                      error="tag failed"),
        ]
        mock_atomizer_cls.return_value.atomize_pages.return_value = [
            AtomizeResult(page_id="p1", title="Page 1", error="atomize failed"),
        ]
        self.notion.query_database.return_value = self.pages
        mock_linker_cls.return_value.link_pages.return_value = [
            LinkResult(page_id="p1", page_title="Page 1", error="link failed"),
        ]

        proc = Processor(self.notion, self.claude, self.config)
        report = proc.process([self.pages[0]])

        assert report.tag_errors == 1
        assert report.atomize_errors == 1
        assert report.link_errors == 1
        assert len(report.errors) == 3
        assert "tag:Page 1" in report.errors[0]
        assert "atomize:Page 1" in report.errors[1]
        assert "link:Page 1" in report.errors[2]

    @patch("notion_notes.commands.process.Tagger")
    @patch("notion_notes.commands.process.Atomizer")
    @patch("notion_notes.commands.process.Linker")
    def test_refetches_all_pages_for_link_phase(
        self, mock_linker_cls, mock_atomizer_cls, mock_tagger_cls
    ) -> None:
        """After atomize creates new pages, link phase should re-fetch all KB pages."""
        mock_tagger_cls.return_value.tag_pages.return_value = []
        mock_atomizer_cls.return_value.atomize_pages.return_value = []

        expanded_pages = self.pages + [_make_page("c1", "Concept A")]
        self.notion.query_database.return_value = expanded_pages

        mock_linker_cls.return_value.link_pages.return_value = []

        proc = Processor(self.notion, self.claude, self.config)
        proc.process(self.pages)

        # Verify link phase got the re-fetched pages (including new atomic note)
        self.notion.query_database.assert_called_once_with("kb-db-id")
        call_args = mock_linker_cls.return_value.link_pages.call_args
        _, all_pages_arg = call_args[0]  # second positional arg
        assert len(all_pages_arg) == 3

    @patch("notion_notes.commands.process.Tagger")
    @patch("notion_notes.commands.process.Atomizer")
    @patch("notion_notes.commands.process.Linker")
    def test_dry_run_passed_through(
        self, mock_linker_cls, mock_atomizer_cls, mock_tagger_cls
    ) -> None:
        mock_tagger_cls.return_value.tag_pages.return_value = []
        mock_atomizer_cls.return_value.atomize_pages.return_value = []
        self.notion.query_database.return_value = []
        mock_linker_cls.return_value.link_pages.return_value = []

        proc = Processor(self.notion, self.claude, self.config, dry_run=True)
        proc.process(self.pages)

        # Verify dry_run was passed to each sub-command
        _, kwargs = mock_tagger_cls.call_args
        assert kwargs["dry_run"] is True
        _, kwargs = mock_atomizer_cls.call_args
        assert kwargs["dry_run"] is True
        _, kwargs = mock_linker_cls.call_args
        assert kwargs["dry_run"] is True


if __name__ == "__main__":
    unittest.main()
