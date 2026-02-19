"""Tag command — classify Notion KB pages by domain and tags using Claude.

Reads page content, asks Claude to assign a domain + tags from a taxonomy,
updates Notion page properties, and grows the taxonomy with new tags.
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
class TagConfig:
    """Configuration for the tagging command."""

    taxonomy_path: Path = Path("taxonomy.json")
    model: str = "claude-haiku-4-5-20251001"
    max_tags: int = 4
    max_retries: int = 1


@dataclass
class TagResult:
    """Result of tagging a single page."""

    page_id: str
    title: str
    domain: str
    tags: list[str]
    new_tags: list[str]
    error: str | None = None


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a note categorizer for a university knowledge base. "
    "Your job is to assign exactly ONE domain and up to {max_tags} tags to each note. "
    "Prefer existing tags from the taxonomy. Only suggest new tags when nothing fits."
)

_USER_PROMPT = """\
Categorize this note.

## Taxonomy

Domains: {domains}

Existing tags by domain:
{tags_by_domain}

## Note content

Title: {title}

{content}

## Instructions

1. Pick exactly ONE domain from the list above.
2. Pick 1-{max_tags} tags. Prefer existing tags for the chosen domain.
3. If no existing tag fits, suggest a new short tag (2-3 words max).
4. Put new tags in the "new_tags" array (empty if all tags already exist).

Respond with ONLY valid JSON, no markdown fences:
{{"domain": "...", "tags": ["..."], "new_tags": ["..."]}}"""


# ------------------------------------------------------------------
# Tagger
# ------------------------------------------------------------------


class Tagger:
    """Tags Notion KB pages using Claude and a growing taxonomy.

    Parameters
    ----------
    notion : NotionClientWrapper
        Notion API wrapper.
    claude : ClaudeClient
        Claude API wrapper.
    config : TagConfig
        Tagging configuration.
    dry_run : bool
        When True, compute tags but don't write to Notion.
    """

    def __init__(
        self,
        notion: NotionClientWrapper,
        claude: ClaudeClient,
        config: TagConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        self.notion = notion
        self.claude = claude
        self.config = config or TagConfig()
        self.dry_run = dry_run

    def tag_pages(self, pages: list[NotionPage]) -> list[TagResult]:
        """Tag a list of pages. Returns one TagResult per page.

        After processing all pages, saves any new tags to the taxonomy file.
        """
        taxonomy = self._load_taxonomy()
        results: list[TagResult] = []
        all_new_tags: dict[str, list[str]] = {}  # domain -> [new tags]

        for page in pages:
            result = self._tag_one_page(page, taxonomy)
            results.append(result)

            # Accumulate new tags for batch taxonomy save
            if result.new_tags and not result.error:
                domain_new = all_new_tags.setdefault(result.domain, [])
                for tag in result.new_tags:
                    if tag not in domain_new:
                        domain_new.append(tag)

        # Save taxonomy with new tags (once, not per page)
        if all_new_tags:
            self._grow_taxonomy(taxonomy, all_new_tags)

        return results

    def _tag_one_page(
        self, page: NotionPage, taxonomy: dict[str, Any]
    ) -> TagResult:
        """Tag a single page. Fetches content, prompts Claude, updates Notion."""
        try:
            # 1. Fetch content
            blocks = self.notion.get_page_blocks(page.id)
            content = NotionClientWrapper.blocks_to_markdown(blocks)

            if not content.strip():
                logger.warning("Skipping empty page: %s (%s)", page.title, page.id)
                return TagResult(
                    page_id=page.id,
                    title=page.title,
                    domain="",
                    tags=[],
                    new_tags=[],
                    error="empty content",
                )

            # 2. Build prompt and call Claude
            system, user_msg = self._build_tag_prompt(
                title=page.title, content=content, taxonomy=taxonomy
            )
            raw_response = self.claude.prompt(
                user_message=user_msg,
                system=system,
                max_tokens=256,
                model=self.config.model,
            )

            # 3. Parse response (with retry on bad JSON)
            parsed = self._extract_json_with_retry(
                raw_response, system, user_msg
            )

            # 4. Validate
            domain = parsed.get("domain", "")
            tags = parsed.get("tags", [])
            new_tags = parsed.get("new_tags", [])

            valid_domains = list(taxonomy.get("domains", {}).keys())
            if domain not in valid_domains:
                return TagResult(
                    page_id=page.id,
                    title=page.title,
                    domain=domain,
                    tags=tags,
                    new_tags=[],
                    error=f"invalid domain '{domain}', expected one of {valid_domains}",
                )

            # Enforce tag limit
            if len(tags) > self.config.max_tags:
                tags = tags[: self.config.max_tags]
                new_tags = [t for t in new_tags if t in tags]

            # Determine which tags are truly new
            existing_tags = set(taxonomy["domains"].get(domain, []))
            new_tags = [t for t in new_tags if t not in existing_tags]

            # 5. Update Notion page (unless dry-run)
            if not self.dry_run:
                self._update_page_tags(page.id, domain, tags)

            logger.info(
                "Tagged: %s -> %s %s%s",
                page.title,
                domain,
                tags,
                f" (new: {new_tags})" if new_tags else "",
            )

            return TagResult(
                page_id=page.id,
                title=page.title,
                domain=domain,
                tags=tags,
                new_tags=new_tags,
            )

        except Exception as exc:
            logger.error("Error tagging %s: %s", page.title, exc)
            return TagResult(
                page_id=page.id,
                title=page.title,
                domain="",
                tags=[],
                new_tags=[],
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_tag_prompt(
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

        system = _SYSTEM_PROMPT.format(max_tags=self.config.max_tags)
        user_msg = _USER_PROMPT.format(
            domains=", ".join(domain_names),
            tags_by_domain="\n".join(tags_by_domain_lines),
            title=title,
            content=content[:3000],  # Truncate long notes for Haiku
            max_tags=self.config.max_tags,
        )
        return system, user_msg

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    def _extract_json_with_retry(
        self,
        raw_text: str,
        system: str,
        user_msg: str,
    ) -> dict[str, Any]:
        """Parse JSON from Claude's response, retrying once on failure."""
        try:
            return self._extract_json(raw_text)
        except (json.JSONDecodeError, ValueError):
            if self.config.max_retries < 1:
                raise
            logger.debug("Bad JSON, retrying with explicit instruction")
            retry_response = self.claude.prompt(
                user_message=(
                    "Your previous response was not valid JSON. "
                    "Please respond with ONLY a valid JSON object, "
                    "no markdown fences or extra text:\n"
                    '{"domain": "...", "tags": ["..."], "new_tags": ["..."]}'
                ),
                system=system,
                max_tokens=256,
                model=self.config.model,
            )
            return self._extract_json(retry_response)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Strip markdown fences and parse JSON, validating required keys."""
        cleaned = text.strip()

        # Strip ```json ... ``` or ``` ... ``` fences
        fence_pattern = re.compile(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL
        )
        match = fence_pattern.match(cleaned)
        if match:
            cleaned = match.group(1).strip()

        parsed = json.loads(cleaned)

        # Validate required keys
        if "domain" not in parsed:
            raise ValueError("Missing 'domain' key in response")
        if "tags" not in parsed:
            raise ValueError("Missing 'tags' key in response")

        # Ensure new_tags exists (optional in response)
        if "new_tags" not in parsed:
            parsed["new_tags"] = []

        return parsed

    # ------------------------------------------------------------------
    # Notion updates
    # ------------------------------------------------------------------

    def _update_page_tags(
        self, page_id: str, domain: str, tags: list[str]
    ) -> None:
        """Write Domain (select) and Tags (multi_select) to a Notion page."""
        properties: dict[str, Any] = {
            "Domain": {"select": {"name": domain}},
            "Tags": {
                "multi_select": [{"name": tag} for tag in tags]
            },
        }
        self.notion.update_page_properties(page_id, properties)

    # ------------------------------------------------------------------
    # Taxonomy I/O
    # ------------------------------------------------------------------

    def _load_taxonomy(self) -> dict[str, Any]:
        """Load taxonomy from JSON file, creating seed if missing."""
        path = self.config.taxonomy_path
        if not path.is_file():
            logger.info("Taxonomy file not found, creating seed: %s", path)
            seed = _seed_taxonomy()
            path.write_text(json.dumps(seed, indent=2), encoding="utf-8")
            return seed
        return json.loads(path.read_text(encoding="utf-8"))

    def _grow_taxonomy(
        self,
        taxonomy: dict[str, Any],
        new_tags_by_domain: dict[str, list[str]],
    ) -> None:
        """Append new tags to taxonomy and save."""
        domains = taxonomy.get("domains", {})
        for domain, new_tags in new_tags_by_domain.items():
            existing = domains.get(domain, [])
            for tag in new_tags:
                if tag not in existing:
                    existing.append(tag)
            domains[domain] = existing
        taxonomy["domains"] = domains

        self.config.taxonomy_path.write_text(
            json.dumps(taxonomy, indent=2), encoding="utf-8"
        )
        logger.info("Taxonomy updated with new tags: %s", new_tags_by_domain)

    def _save_taxonomy(self, taxonomy: dict[str, Any]) -> None:
        """Save taxonomy to disk."""
        self.config.taxonomy_path.write_text(
            json.dumps(taxonomy, indent=2), encoding="utf-8"
        )


# ------------------------------------------------------------------
# Seed taxonomy
# ------------------------------------------------------------------


def _seed_taxonomy() -> dict[str, Any]:
    """Return the initial seed taxonomy."""
    return {
        "version": 1,
        "domains": {
            "Neuroscience": [
                "synaptic plasticity",
                "neural circuits",
                "neuroanatomy",
                "electrophysiology",
                "neurotransmitters",
            ],
            "Molecular Biology": [
                "gene expression",
                "protein structure",
                "signal transduction",
                "DNA replication",
                "RNA processing",
            ],
            "Pharmacology": [
                "drug mechanisms",
                "pharmacokinetics",
                "receptor biology",
                "dose-response",
                "clinical trials",
            ],
            "Psychology": [
                "cognitive processes",
                "behavioral analysis",
                "developmental psychology",
                "psychopathology",
                "research methods",
            ],
            "Physiology": [
                "cardiovascular",
                "respiratory",
                "endocrine",
                "renal",
                "homeostasis",
            ],
            "Biostatistics": [
                "experimental design",
                "hypothesis testing",
                "regression",
                "survival analysis",
                "meta-analysis",
            ],
        },
    }
