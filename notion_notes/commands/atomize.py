"""Atomize command — split raw class notes into atomic concept pages using Claude.

Reads page content, asks Claude to identify distinct concepts, and creates
individual Notion pages for each concept linked back to the source note.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from notion_notes.claude_client import ClaudeClient
from notion_notes.notion_client import NotionClientWrapper, NotionPage

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AtomizeConfig:
    """Configuration for the atomize command."""

    taxonomy_path: Path = Path("taxonomy.json")
    model: str = "claude-sonnet-4-6"
    max_concepts: int = 10
    max_retries: int = 1
    max_content_chars: int = 8000


@dataclass(frozen=True)
class ConceptNote:
    """A single concept extracted from a source page."""

    title: str
    content_markdown: str
    domain: str
    tags: list[str]


@dataclass
class AtomizeResult:
    """Result of atomizing a single page."""

    page_id: str
    title: str
    concepts_created: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are processing a student's class notes into atomic concept notes. "
    "Each concept note should be self-contained and make sense without the "
    "surrounding lecture context."
)

_USER_PROMPT = """\
Analyze these class notes and identify distinct concepts, claims, mechanisms, \
or ideas. For each one, create an atomic note.

## Taxonomy

Domains: {domains}

Existing tags by domain:
{tags_by_domain}

## Source note

Title: {title}

{content}

## Instructions

For each distinct concept in the source:
1. Give it a concise title (the concept name, not a sentence).
2. Write a clear, self-contained explanation that captures the idea fully. \
Use the student's own phrasing as a foundation but ensure the note makes \
sense on its own. If the notes are brief or shorthand, flesh it out into \
a complete explanation.
3. Pick exactly ONE domain from the taxonomy above.
4. Pick 1-4 tags. Prefer existing tags for the chosen domain.
5. Only create separate notes for genuinely distinct concepts. Don't \
over-split — a single mechanism described in detail is one note, not five.
6. Maximum {max_concepts} concept notes.

Respond with ONLY valid JSON, no markdown fences:
{{"concepts": [{{"title": "...", "content": "...", "domain": "...", "tags": ["..."]}}]}}"""


# ------------------------------------------------------------------
# Atomizer
# ------------------------------------------------------------------


class Atomizer:
    """Splits raw Notion pages into atomic concept notes using Claude.

    Parameters
    ----------
    notion : NotionClientWrapper
        Notion API wrapper.
    claude : ClaudeClient
        Claude API wrapper.
    config : AtomizeConfig
        Atomize configuration.
    dry_run : bool
        When True, compute concepts but don't create pages in Notion.
    """

    def __init__(
        self,
        notion: NotionClientWrapper,
        claude: ClaudeClient,
        config: AtomizeConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        self.notion = notion
        self.claude = claude
        self.config = config or AtomizeConfig()
        self.dry_run = dry_run

    def atomize_pages(self, pages: list[NotionPage]) -> list[AtomizeResult]:
        """Atomize a list of pages. Returns one AtomizeResult per page."""
        results: list[AtomizeResult] = []
        for page in pages:
            # Skip already-atomized pages
            status = NotionClientWrapper.extract_property(
                page.properties, "Status"
            )
            if status == "Atomized":
                logger.warning(
                    "Skipping already-atomized page: %s (%s)",
                    page.title, page.id,
                )
                results.append(AtomizeResult(
                    page_id=page.id,
                    title=page.title,
                    error="already atomized",
                ))
                continue

            result = self._atomize_one_page(page)
            results.append(result)
        return results

    def _atomize_one_page(self, page: NotionPage) -> AtomizeResult:
        """Atomize a single page: fetch content, call Claude, create pages."""
        try:
            # 1. Fetch blocks and convert to markdown
            blocks = self.notion.get_page_blocks(page.id)
            content = NotionClientWrapper.blocks_to_markdown(blocks)

            if not content.strip():
                logger.warning(
                    "Skipping empty page: %s (%s)", page.title, page.id
                )
                return AtomizeResult(
                    page_id=page.id,
                    title=page.title,
                    error="empty content",
                )

            # 2. Build prompt and call Claude
            taxonomy = self._load_taxonomy()
            system, user_msg = self._build_atomize_prompt(
                title=page.title, content=content, taxonomy=taxonomy
            )
            raw_response = self.claude.prompt(
                user_message=user_msg,
                system=system,
                max_tokens=4096,
                model=self.config.model,
            )

            # 3. Parse concepts
            concepts = self._parse_concepts(raw_response, system, user_msg)

            if not concepts:
                # Already atomic or too short — mark as atomized without children
                logger.info(
                    "No concepts extracted for %s — marking as Atomized",
                    page.title,
                )
                if not self.dry_run:
                    self._update_status(page.id, "Atomized")
                return AtomizeResult(
                    page_id=page.id,
                    title=page.title,
                )

            # 4. Get the database_id from the page's parent
            database_id = self._get_page_database_id(page)

            # 5. Create concept pages
            created: list[dict[str, str]] = []
            for concept in concepts:
                if self.dry_run:
                    logger.info(
                        "[DRY RUN] Would create concept: %s", concept.title
                    )
                    created.append({"title": concept.title, "new_page_id": ""})
                else:
                    new_page_id = self._create_concept_page(
                        concept, page.id, database_id
                    )
                    created.append({
                        "title": concept.title,
                        "new_page_id": new_page_id or "",
                    })

            # 6. Annotate source page and update status
            if not self.dry_run:
                self._annotate_source(page.id, created)
                self._update_status(page.id, "Atomized")

            logger.info(
                "Atomized: %s -> %d concept(s)", page.title, len(created)
            )
            return AtomizeResult(
                page_id=page.id,
                title=page.title,
                concepts_created=created,
            )

        except Exception as exc:
            logger.error("Error atomizing %s: %s", page.title, exc)
            return AtomizeResult(
                page_id=page.id,
                title=page.title,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_atomize_prompt(
        self,
        title: str,
        content: str,
        taxonomy: dict[str, Any],
    ) -> tuple[str, str]:
        """Build (system_prompt, user_message) for Claude."""
        domains = taxonomy.get("domains", {})
        domain_names = list(domains.keys())

        tags_by_domain_lines: list[str] = []
        for domain_name, domain_tags in domains.items():
            if domain_tags:
                tags_by_domain_lines.append(
                    f"  {domain_name}: {', '.join(domain_tags)}"
                )
            else:
                tags_by_domain_lines.append(f"  {domain_name}: (none yet)")

        system = _SYSTEM_PROMPT
        user_msg = _USER_PROMPT.format(
            domains=", ".join(domain_names),
            tags_by_domain="\n".join(tags_by_domain_lines),
            title=title,
            content=content[: self.config.max_content_chars],
            max_concepts=self.config.max_concepts,
        )
        return system, user_msg

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    def _parse_concepts(
        self,
        raw_text: str,
        system: str,
        user_msg: str,
    ) -> list[ConceptNote]:
        """Parse concept notes from Claude's JSON response, retrying on failure."""
        try:
            return self._extract_concepts(raw_text)
        except (json.JSONDecodeError, ValueError):
            if self.config.max_retries < 1:
                raise
            logger.debug("Bad JSON from atomize, retrying with explicit instruction")
            retry_response = self.claude.prompt(
                user_message=(
                    "Your previous response was not valid JSON. "
                    "Please respond with ONLY a valid JSON object, "
                    "no markdown fences or extra text:\n"
                    '{"concepts": [{"title": "...", "content": "...", '
                    '"domain": "...", "tags": ["..."]}]}'
                ),
                system=system,
                max_tokens=4096,
                model=self.config.model,
            )
            return self._extract_concepts(retry_response)

    @staticmethod
    def _extract_concepts(text: str) -> list[ConceptNote]:
        """Strip markdown fences and parse JSON into ConceptNote list."""
        cleaned = text.strip()

        # Strip ```json ... ``` or ``` ... ``` fences
        fence_pattern = re.compile(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL
        )
        match = fence_pattern.match(cleaned)
        if match:
            cleaned = match.group(1).strip()

        parsed = json.loads(cleaned)

        if "concepts" not in parsed:
            raise ValueError("Missing 'concepts' key in response")

        concepts: list[ConceptNote] = []
        for item in parsed["concepts"]:
            if not item.get("title") or not item.get("content"):
                continue
            concepts.append(ConceptNote(
                title=item["title"],
                content_markdown=item["content"],
                domain=item.get("domain", ""),
                tags=item.get("tags", []),
            ))
        return concepts

    # ------------------------------------------------------------------
    # Notion operations
    # ------------------------------------------------------------------

    def _create_concept_page(
        self,
        concept: ConceptNote,
        source_page_id: str,
        database_id: str,
    ) -> str | None:
        """Create a new atomic concept page in Notion.

        Returns the new page ID, or None in dry-run mode.
        """
        content_blocks = NotionClientWrapper.markdown_to_blocks(
            concept.content_markdown
        )

        properties: dict[str, Any] = {
            "Name": {
                "title": [
                    {"type": "text", "text": {"content": concept.title}}
                ]
            },
            "Is Atomic": {"checkbox": True},
            "Status": {"select": {"name": "Atomized"}},
            "Source Note": {
                "relation": [{"id": source_page_id}]
            },
        }

        if concept.domain:
            properties["Domain"] = {"select": {"name": concept.domain}}
        if concept.tags:
            properties["Tags"] = {
                "multi_select": [{"name": tag} for tag in concept.tags]
            }

        return self.notion.create_page(
            database_id=database_id,
            properties=properties,
            children=content_blocks,
        )

    def _annotate_source(
        self,
        page_id: str,
        created_concepts: list[dict[str, str]],
    ) -> None:
        """Prepend a callout block to the source page listing created concepts."""
        titles = [c["title"] for c in created_concepts]
        callout_text = (
            f"Atomized into {len(titles)} concept note(s): "
            + ", ".join(titles)
        )

        callout_block: dict[str, Any] = {
            "type": "callout",
            "callout": {
                "rich_text": [
                    {"type": "text", "text": {"content": callout_text}}
                ],
                "icon": {"type": "emoji", "emoji": "💡"},
            },
        }

        self.notion.append_blocks(page_id, [callout_block])

    def _update_status(self, page_id: str, status: str) -> None:
        """Update the Status property of a page."""
        self.notion.update_page_properties(
            page_id,
            {"Status": {"select": {"name": status}}},
        )

    def _get_page_database_id(self, page: NotionPage) -> str:
        """Extract the parent database ID from a page.

        Falls back to querying the Notion API if not available in properties.
        """
        # The page object might have parent info from the raw API response.
        # Since our NotionPage doesn't store parent, we fetch the page again.
        raw = self.notion._rate_limited_call(
            self.notion.client.pages.retrieve, page_id=page.id
        )
        parent = raw.get("parent", {})
        db_id = parent.get("database_id", "")
        if not db_id:
            raise ValueError(
                f"Page {page.id} has no parent database — cannot create "
                f"concept pages in the same database."
            )
        return db_id

    # ------------------------------------------------------------------
    # Taxonomy I/O
    # ------------------------------------------------------------------

    def _load_taxonomy(self) -> dict[str, Any]:
        """Load taxonomy from JSON file."""
        path = self.config.taxonomy_path
        if not path.is_file():
            logger.warning("Taxonomy file not found: %s", path)
            return {"domains": {}}
        return json.loads(path.read_text(encoding="utf-8"))
