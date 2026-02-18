"""Link command — discover semantic connections between KB pages using Claude.

Uses two-stage filtering: fast pre-filtering by tag/domain overlap, then
semantic comparison via Claude (Sonnet) for plausible candidates. A local
cache prevents O(n^2) re-evaluation of already-checked pairs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from notion_notes.claude_client import ClaudeClient
from notion_notes.notion_client import NotionClientWrapper, NotionPage

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class LinkConfig:
    """Configuration for the link command."""

    model: str = "claude-sonnet-4-6"
    max_candidates: int = 20
    min_tag_overlap: int = 1
    cache_path: Path = Path(".cache/link_pairs.json")
    connection_types: tuple[str, ...] = (
        "same-concept-different-context",
        "prerequisite-builds-on",
        "contradicts-alternative-view",
        "shares-mechanism-principle",
        "part-of-same-system",
    )
    max_content_chars: int = 6000


@dataclass(frozen=True)
class Connection:
    """A discovered semantic connection between two pages."""

    page_a_id: str
    page_b_id: str
    connection_type: str
    explanation: str


@dataclass
class LinkResult:
    """Result of linking a single page."""

    page_id: str
    page_title: str
    connections_found: list[Connection] = field(default_factory=list)
    candidates_evaluated: int = 0
    skipped_cached: int = 0
    error: str | None = None


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are analyzing a knowledge base to find genuine semantic connections "
    "between notes. Only identify connections that would be useful for "
    "cross-domain understanding. Skip trivial connections."
)

_USER_PROMPT = """\
Given this note and these candidates, identify which candidates are \
genuinely related in a way useful for cross-domain understanding.

## Target note

Title: {title}

{content}

## Candidates

{candidates}

## Instructions

1. For each candidate that has a genuine semantic connection to the target \
note, identify the connection type and explain it in one sentence.
2. Connection types: {connection_types}
3. Skip trivial connections (e.g., same course, adjacent lecture topics \
with no deeper link).
4. Only include connections that would help a student see cross-cutting \
themes or deeper relationships.
5. If no candidates have genuine connections, return an empty array.

Respond with ONLY valid JSON, no markdown fences:
{{"connections": [{{"candidate_title": "...", "connection_type": "...", \
"explanation": "..."}}]}}"""


# ------------------------------------------------------------------
# Linker
# ------------------------------------------------------------------


class Linker:
    """Discover and create semantic connections between KB pages.

    Parameters
    ----------
    notion : NotionClientWrapper
        Notion API wrapper.
    claude : ClaudeClient
        Claude API wrapper.
    config : LinkConfig
        Link configuration.
    dry_run : bool
        When True, compute connections but don't create relations in Notion.
    """

    def __init__(
        self,
        notion: NotionClientWrapper,
        claude: ClaudeClient,
        config: LinkConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        self.notion = notion
        self.claude = claude
        self.config = config or LinkConfig()
        self.dry_run = dry_run

    def link_pages(
        self, pages: list[NotionPage], all_pages: list[NotionPage]
    ) -> list[LinkResult]:
        """Link a list of pages against the full KB. Returns one LinkResult per page."""
        cache = self._load_cache()
        results: list[LinkResult] = []

        for page in pages:
            result = self._link_one_page(page, all_pages, cache)
            results.append(result)

        self._save_cache(cache)
        return results

    def _link_one_page(
        self,
        page: NotionPage,
        all_pages: list[NotionPage],
        cache: set[tuple[str, str]],
    ) -> LinkResult:
        """Link a single page: find candidates, call Claude, create relations."""
        try:
            # 1. Find candidates via fast pre-filtering
            candidates, skipped_cached = self._find_candidates(
                page, all_pages, cache
            )

            if not candidates:
                logger.info(
                    "No candidates for %s (skipped %d cached)",
                    page.title,
                    skipped_cached,
                )
                # Mark all candidate pairs as evaluated even if empty
                return LinkResult(
                    page_id=page.id,
                    page_title=page.title,
                    candidates_evaluated=0,
                    skipped_cached=skipped_cached,
                )

            # 2. Fetch target page content
            blocks = self.notion.get_page_blocks(page.id)
            content = NotionClientWrapper.blocks_to_markdown(blocks)

            if not content.strip():
                logger.warning(
                    "Skipping empty page: %s (%s)", page.title, page.id
                )
                return LinkResult(
                    page_id=page.id,
                    page_title=page.title,
                    error="empty content",
                )

            # 3. Build prompt and call Claude
            candidate_map = {c.title: c.id for c in candidates}
            system, user_msg = self._build_link_prompt(
                content, page.title, candidates
            )
            raw_response = self.claude.prompt(
                user_message=user_msg,
                system=system,
                max_tokens=2048,
                model=self.config.model,
            )

            # 4. Parse connections
            connections = self._parse_connections(
                raw_response, page.id, candidate_map
            )

            # 5. Create relations (bidirectional)
            if not self.dry_run:
                for conn in connections:
                    self.notion.add_relation(
                        conn.page_a_id, "Related Notes", [conn.page_b_id]
                    )
                    self.notion.add_relation(
                        conn.page_b_id, "Related Notes", [conn.page_a_id]
                    )

                # Update status to Linked
                self.notion.update_page_properties(
                    page.id,
                    {"Status": {"select": {"name": "Linked"}}},
                )

            # 6. Update cache with all evaluated pairs
            for candidate in candidates:
                cache.add(self._pair_key(page.id, candidate.id))

            logger.info(
                "Linked: %s -> %d connection(s) from %d candidates",
                page.title,
                len(connections),
                len(candidates),
            )
            return LinkResult(
                page_id=page.id,
                page_title=page.title,
                connections_found=connections,
                candidates_evaluated=len(candidates),
                skipped_cached=skipped_cached,
            )

        except Exception as exc:
            logger.error("Error linking %s: %s", page.title, exc)
            return LinkResult(
                page_id=page.id,
                page_title=page.title,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        page: NotionPage,
        all_pages: list[NotionPage],
        cache: set[tuple[str, str]],
    ) -> tuple[list[NotionPage], int]:
        """Fast pre-filtering by tag overlap and domain match.

        Returns (candidates, skipped_cached_count).
        """
        page_tags = set(
            NotionClientWrapper.extract_property(page.properties, "Tags") or []
        )
        page_domain = (
            NotionClientWrapper.extract_property(page.properties, "Domain") or ""
        )

        scored: list[tuple[int, NotionPage]] = []
        skipped_cached = 0

        for candidate in all_pages:
            # Skip self
            if candidate.id == page.id:
                continue

            # Skip cached pairs
            if self._pair_key(page.id, candidate.id) in cache:
                skipped_cached += 1
                continue

            # Compute tag overlap
            cand_tags = set(
                NotionClientWrapper.extract_property(
                    candidate.properties, "Tags"
                )
                or []
            )
            tag_overlap = len(page_tags & cand_tags)

            # Check domain match
            cand_domain = (
                NotionClientWrapper.extract_property(
                    candidate.properties, "Domain"
                )
                or ""
            )
            same_domain = (
                page_domain and cand_domain and page_domain == cand_domain
            )

            # Include if tag overlap meets threshold OR same domain
            if tag_overlap >= self.config.min_tag_overlap or same_domain:
                scored.append((tag_overlap, candidate))

        # Sort by tag overlap descending, limit to max_candidates
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [c for _, c in scored[: self.config.max_candidates]]

        return candidates, skipped_cached

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_link_prompt(
        self,
        content: str,
        title: str,
        candidates: list[NotionPage],
    ) -> tuple[str, str]:
        """Build (system_prompt, user_message) for Claude."""
        candidate_lines: list[str] = []
        for i, cand in enumerate(candidates, 1):
            cand_tags = (
                NotionClientWrapper.extract_property(
                    cand.properties, "Tags"
                )
                or []
            )
            cand_domain = (
                NotionClientWrapper.extract_property(
                    cand.properties, "Domain"
                )
                or "untagged"
            )
            tags_str = ", ".join(cand_tags) if cand_tags else "none"
            candidate_lines.append(
                f"{i}. **{cand.title}** (Domain: {cand_domain}, Tags: {tags_str})"
            )

        system = _SYSTEM_PROMPT
        user_msg = _USER_PROMPT.format(
            title=title,
            content=content[: self.config.max_content_chars],
            candidates="\n".join(candidate_lines),
            connection_types=", ".join(self.config.connection_types),
        )
        return system, user_msg

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    def _parse_connections(
        self,
        raw_text: str,
        page_id: str,
        candidate_map: dict[str, str],
    ) -> list[Connection]:
        """Parse Connection objects from Claude's JSON response."""
        cleaned = raw_text.strip()

        # Strip ```json ... ``` or ``` ... ``` fences
        fence_pattern = re.compile(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL
        )
        match = fence_pattern.match(cleaned)
        if match:
            cleaned = match.group(1).strip()

        parsed = json.loads(cleaned)

        if "connections" not in parsed:
            raise ValueError("Missing 'connections' key in response")

        connections: list[Connection] = []
        for item in parsed["connections"]:
            cand_title = item.get("candidate_title", "")
            cand_id = candidate_map.get(cand_title)
            if not cand_id:
                logger.warning(
                    "Unknown candidate title in response: %s", cand_title
                )
                continue

            conn_type = item.get("connection_type", "")
            explanation = item.get("explanation", "")

            if not conn_type or not explanation:
                continue

            connections.append(
                Connection(
                    page_a_id=page_id,
                    page_b_id=cand_id,
                    connection_type=conn_type,
                    explanation=explanation,
                )
            )
        return connections

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _load_cache(self) -> set[tuple[str, str]]:
        """Load evaluated pair cache from disk."""
        path = self.config.cache_path
        if not path.is_file():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {tuple(pair) for pair in data}
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Corrupted cache file, starting fresh: %s", path)
            return set()

    def _save_cache(self, cache: set[tuple[str, str]]) -> None:
        """Save evaluated pair cache to disk."""
        path = self.config.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [list(pair) for pair in sorted(cache)]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _pair_key(id_a: str, id_b: str) -> tuple[str, str]:
        """Canonical pair key (sorted) so A-B == B-A."""
        return tuple(sorted([id_a, id_b]))
