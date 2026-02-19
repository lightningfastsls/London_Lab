"""Process command — run tag -> atomize -> link in sequence on KB pages.

Orchestrates the full processing pipeline. Each stage feeds into the next:
tagging classifies pages, atomizing splits them into concepts, linking
discovers connections across the entire KB including new atomic notes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from notion_notes.claude_client import ClaudeClient
from notion_notes.commands.atomize import AtomizeConfig, AtomizeResult, Atomizer
from notion_notes.commands.link import LinkConfig, LinkResult, Linker
from notion_notes.commands.tag import TagConfig, TagResult, Tagger
from notion_notes.notion_client import NotionClientWrapper, NotionPage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessConfig:
    """Configuration for the process command."""

    tag_config: TagConfig = field(default_factory=TagConfig)
    atomize_config: AtomizeConfig = field(default_factory=AtomizeConfig)
    link_config: LinkConfig = field(default_factory=LinkConfig)
    kb_database_id: str = ""


@dataclass
class ProcessReport:
    """Aggregated result of running the full processing pipeline."""

    pages_input: int = 0
    tags_applied: int = 0
    tag_errors: int = 0
    pages_atomized: int = 0
    concepts_created: int = 0
    atomize_errors: int = 0
    pages_linked: int = 0
    connections_found: int = 0
    link_errors: int = 0
    tag_results: list[TagResult] = field(default_factory=list)
    atomize_results: list[AtomizeResult] = field(default_factory=list)
    link_results: list[LinkResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of the processing run."""
        lines = [
            f"Process Report ({self.pages_input} pages input)",
            f"  Tag:     {self.tags_applied} tagged, {self.tag_errors} errors",
            f"  Atomize: {self.pages_atomized} atomized, "
            f"{self.concepts_created} concepts created, {self.atomize_errors} errors",
            f"  Link:    {self.pages_linked} linked, "
            f"{self.connections_found} connections found, {self.link_errors} errors",
        ]
        if self.errors:
            lines.append(f"  Errors:  {len(self.errors)} total")
            for err in self.errors[:5]:
                lines.append(f"    - {err}")
            if len(self.errors) > 5:
                lines.append(f"    ... and {len(self.errors) - 5} more")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Processor
# ------------------------------------------------------------------


class Processor:
    """Run the full processing pipeline: tag -> atomize -> link."""

    def __init__(
        self,
        notion: NotionClientWrapper,
        claude: ClaudeClient,
        config: ProcessConfig,
        dry_run: bool = False,
    ) -> None:
        self._notion = notion
        self._claude = claude
        self._config = config
        self._dry_run = dry_run

    def process(self, pages: list[NotionPage]) -> ProcessReport:
        """Run tag -> atomize -> link on the given pages.

        The pipeline:
        1. Tag all pages (fast, cheap — Haiku).
        2. Atomize pages that aren't yet atomized (Sonnet).
        3. Re-fetch all KB pages so newly created atomic notes are included,
           then link the original input pages + new atomic notes.
        """
        report = ProcessReport(pages_input=len(pages))

        if not pages:
            return report

        # --- Phase 1: Tag ---
        logger.info("Phase 1/3: Tagging %d pages...", len(pages))
        tagger = Tagger(
            notion=self._notion,
            claude=self._claude,
            config=self._config.tag_config,
            dry_run=self._dry_run,
        )
        tag_results = tagger.tag_pages(pages)
        report.tag_results = tag_results
        for r in tag_results:
            if r.error:
                report.tag_errors += 1
                report.errors.append(f"tag:{r.title}: {r.error}")
            else:
                report.tags_applied += 1

        # --- Phase 2: Atomize ---
        logger.info("Phase 2/3: Atomizing %d pages...", len(pages))
        atomizer = Atomizer(
            notion=self._notion,
            claude=self._claude,
            config=self._config.atomize_config,
            dry_run=self._dry_run,
        )
        atomize_results = atomizer.atomize_pages(pages)
        report.atomize_results = atomize_results
        for r in atomize_results:
            if r.error:
                report.atomize_errors += 1
                report.errors.append(f"atomize:{r.title}: {r.error}")
            else:
                report.pages_atomized += 1
                report.concepts_created += len(r.concepts_created)

        # --- Phase 3: Link ---
        # Re-fetch all KB pages to include newly created atomic notes
        if self._config.kb_database_id:
            logger.info("Phase 3/3: Fetching all KB pages for linking...")
            all_pages = self._notion.query_database(self._config.kb_database_id)
        else:
            all_pages = pages

        linker = Linker(
            notion=self._notion,
            claude=self._claude,
            config=self._config.link_config,
            dry_run=self._dry_run,
        )
        link_results = linker.link_pages(pages, all_pages)
        report.link_results = link_results
        for r in link_results:
            if r.error:
                report.link_errors += 1
                report.errors.append(f"link:{r.page_title}: {r.error}")
            else:
                report.pages_linked += 1
                report.connections_found += len(r.connections_found)

        return report
