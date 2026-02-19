"""Notes-to-KB migration for Notion Atomic Notes.

One-time migration script that copies qualifying pages from the Notes database
to the Knowledge Base, maps properties, re-establishes relations, and produces
a report.

Two-pass approach:
  1. Create all pages in KB, building an old_id -> new_id mapping.
  2. Re-establish Related Notes relations using the mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from notion_notes.notion_client import NotionClientWrapper, NotionPage

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Read-only block fields that must be stripped before writing
# ------------------------------------------------------------------
_BLOCK_READONLY_FIELDS = frozenset({
    "id",
    "parent",
    "created_time",
    "last_edited_time",
    "created_by",
    "last_edited_by",
    "has_children",
    "archived",
    "in_trash",
    "object",
    "request_id",
})


# ------------------------------------------------------------------
# Status value mapping
# ------------------------------------------------------------------


def map_status_value(value: str | None) -> str:
    """Map Notes DB status values to KB status values.

    Returns "Raw" for None, empty, or unrecognised values.
    """
    mapping = {
        "Not started": "Raw",
        "In progress": "In Progress",
        "Done": "Processed",
    }
    if not value:
        return "Raw"
    return mapping.get(value, "Raw")


# ------------------------------------------------------------------
# Property mapping
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PropertyMapping:
    """Describes how a source property maps to a target property."""

    target_name: str
    transform: Callable[[Any], Any] | None = None


PROPERTY_MAP: dict[str, PropertyMapping] = {
    "Tags": PropertyMapping(target_name="Tags"),
    "Evergreen": PropertyMapping(target_name="Evergreen"),
    "Status": PropertyMapping(target_name="Status", transform=map_status_value),
}


def _find_property_key(properties: dict[str, Any], name: str) -> str | None:
    """Find a property key by exact match or suffix match (handles emoji prefixes).

    Notion users often prepend emoji to property names (e.g. '🔬UNI Courses').
    This helper finds the actual key so lookups don't break.
    """
    if name in properties:
        return name
    for key in properties:
        if key.endswith(name):
            return key
    return None


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationConfig:
    """Configuration for a Notes-to-KB migration run."""

    source_db_id: str
    target_db_id: str
    dry_run: bool = False
    require_course: bool = True
    include_projects: tuple[str, ...] = ()
    include_pages: tuple[str, ...] = ()
    exclude_pages: tuple[str, ...] = ()


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


@dataclass
class MigrationReport:
    """Collects statistics and errors from a migration run."""

    total_source: int = 0
    filtered_out: int = 0
    migrated: int = 0
    skipped_dry_run: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    id_mapping: dict[str, str] = field(default_factory=dict)
    relations_remapped: int = 0
    relations_skipped: int = 0

    def summary(self) -> str:
        """Human-readable summary of the migration."""
        lines = [
            f"Source pages found: {self.total_source}",
            f"Filtered out:       {self.filtered_out}",
            f"Migrated:           {self.migrated}",
            f"Dry-run skipped:    {self.skipped_dry_run}",
            f"Errors:             {len(self.errors)}",
            f"Relations remapped: {self.relations_remapped}",
            f"Relations skipped:  {self.relations_skipped}",
        ]
        if self.errors:
            lines.append("Error details:")
            for err in self.errors:
                lines.append(f"  - {err.get('page_id', '?')}: {err.get('error', '?')}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Migrator
# ------------------------------------------------------------------


class NoteMigrator:
    """Migrates pages from Notes DB to Knowledge Base DB.

    Parameters
    ----------
    client : NotionClientWrapper
        Configured Notion client (respects dry_run internally).
    config : MigrationConfig
        Migration settings including source/target DB IDs and filters.
    """

    def __init__(self, client: NotionClientWrapper, config: MigrationConfig) -> None:
        self.client = client
        self.config = config

    def migrate(self) -> MigrationReport:
        """Run the full migration pipeline.

        1. Query source pages
        2. Filter
        3. Discover properties (log first page's property names)
        4. Migrate each page (first pass)
        5. Re-establish relations (second pass)
        """
        report = MigrationReport()

        # Step 1: Query all source pages
        source_pages = self._query_source_pages()
        report.total_source = len(source_pages)
        logger.info("Found %d pages in source database", len(source_pages))

        # Step 2: Discovery — log properties from first page
        if source_pages:
            prop_names = list(source_pages[0].properties.keys())
            logger.info("Source DB properties: %s", prop_names)

        # Step 3: Filter and migrate (first pass)
        old_relations: dict[str, list[str]] = {}  # new_page_id -> old related IDs

        for page in source_pages:
            if not self._should_migrate(page):
                report.filtered_out += 1
                continue

            if self.config.dry_run:
                report.skipped_dry_run += 1
                report.id_mapping[page.id] = f"dry-run-{page.id}"
                # Collect relations even in dry-run for reporting
                related = self.client.extract_property(page.properties, "Related Notes")
                if related:
                    old_relations[f"dry-run-{page.id}"] = related
                continue

            try:
                new_id = self._migrate_page(page)
                if new_id:
                    report.migrated += 1
                    report.id_mapping[page.id] = new_id
                    # Collect old relations for second pass
                    related = self.client.extract_property(page.properties, "Related Notes")
                    if related:
                        old_relations[new_id] = related
            except Exception as exc:
                logger.error("Failed to migrate page %s (%s): %s", page.id, page.title, exc)
                report.errors.append({"page_id": page.id, "title": page.title, "error": str(exc)})

        # Step 4: Re-establish relations (second pass)
        if not self.config.dry_run and old_relations:
            remapped, skipped = self._remap_relations(report.id_mapping, old_relations)
            report.relations_remapped = remapped
            report.relations_skipped = skipped

        logger.info("Migration complete.\n%s", report.summary())
        return report

    def _query_source_pages(self) -> list[NotionPage]:
        """Query pages from the source database.

        When require_course is set, uses a server-side Notion API filter to
        only return pages with a non-empty UNI Courses relation. This is needed
        because dual-property relations return empty arrays in query results,
        so client-side filtering doesn't work.
        """
        api_filter = None
        if self.config.require_course:
            # Find the actual property name (may have emoji prefix)
            # Query one page to discover property names
            sample = self.client.query_database(self.config.source_db_id, page_size=1)
            if sample:
                course_key = _find_property_key(sample[0].properties, "UNI Courses")
                if course_key:
                    api_filter = {
                        "property": course_key,
                        "relation": {"is_not_empty": True},
                    }
                    logger.info(
                        "Using server-side filter: '%s' is_not_empty", course_key
                    )
        return self.client.query_database(self.config.source_db_id, filter=api_filter)

    def _should_migrate(self, page: NotionPage) -> bool:
        """Decide whether a page qualifies for migration.

        Priority: exclude list > include list > require_course check.
        """
        title = page.title

        # Exclude list has highest priority
        if self.config.exclude_pages and title in self.config.exclude_pages:
            logger.debug("Excluding page '%s' (in exclude list)", title)
            return False

        # Include-pages list: if set, only these specific pages migrate
        if self.config.include_pages:
            if title in self.config.include_pages:
                return True
            # If include_pages is set but this page isn't in it, skip
            # (unless include_projects matches)
            if not self.config.include_projects:
                return False

        # Include-projects: check if page has a matching UNI Course
        if self.config.include_projects:
            course_key = _find_property_key(page.properties, "UNI Courses")
            courses = self.client.extract_property(page.properties, course_key) if course_key else []
            courses = courses or []
            if isinstance(courses, str):
                courses = [courses]
            if any(c in self.config.include_projects for c in courses):
                return True
            # If include_projects set but no match and no include_pages match, skip
            if self.config.include_pages:
                return False
            # If only include_projects set (no include_pages), no match = skip
            return False

        # require_course: handled by server-side filter in _query_source_pages.
        # No client-side check needed (dual relations return empty in queries).

        return True

    def _migrate_page(self, page: NotionPage) -> str | None:
        """Migrate a single page: map properties, create in KB, copy content.

        Returns the new page ID or None if creation failed.
        """
        # Map properties
        target_props = self._map_properties(page.properties)

        # Set the title (KB uses "Title" as the title property name)
        target_props["Title"] = {
            "title": [{"text": {"content": page.title}}],
        }

        # Fetch source blocks
        blocks = self._fetch_blocks_recursive(page.id)

        # Strip read-only metadata from blocks
        clean_blocks = [self._strip_block_metadata(b) for b in blocks]

        # Create page in KB (with first 100 blocks if any)
        initial_blocks = clean_blocks[:100] if clean_blocks else None
        new_id = self.client.create_page(
            self.config.target_db_id,
            target_props,
            children=initial_blocks,
        )

        if not new_id:
            return None

        # Append remaining blocks in chunks of 100
        if len(clean_blocks) > 100:
            self._copy_content_blocks(new_id, clean_blocks[100:])

        # Mark source page as migrated
        self._mark_source_migrated(page.id, new_id)

        logger.info("Migrated '%s' -> %s", page.title, new_id)
        return new_id

    def _map_properties(self, source_props: dict[str, Any]) -> dict[str, Any]:
        """Map source properties to target format using PROPERTY_MAP.

        Also sets KB defaults: Is Atomic = False, Status = "Raw" (if not mapped).
        """
        target: dict[str, Any] = {}

        for source_name, mapping in PROPERTY_MAP.items():
            actual_key = _find_property_key(source_props, source_name)
            if actual_key is None:
                continue

            prop = source_props[actual_key]
            prop_type = prop.get("type", "")

            if prop_type == "multi_select":
                tags = [opt["name"] for opt in prop.get("multi_select", [])]
                target[mapping.target_name] = {
                    "multi_select": [{"name": t} for t in tags],
                }

            elif prop_type == "select":
                sel = prop.get("select")
                if sel:
                    value = sel["name"]
                    if mapping.transform:
                        value = mapping.transform(value)
                    target[mapping.target_name] = {"select": {"name": value}}

            elif prop_type == "checkbox":
                target[mapping.target_name] = {
                    "checkbox": prop.get("checkbox", False),
                }

            elif prop_type == "status":
                raw_value = prop.get("status", {}).get("name") if prop.get("status") else None
                value = mapping.transform(raw_value) if mapping.transform else raw_value
                # KB uses select (not status) for this property
                target[mapping.target_name] = {"select": {"name": value}}

            elif prop_type == "relation":
                # Relations are handled in the second pass; skip here
                pass

            else:
                logger.debug(
                    "Skipping property '%s' (unmapped type: %s)", source_name, prop_type
                )

        # KB defaults
        if "Is Atomic" not in target:
            target["Is Atomic"] = {"checkbox": False}
        if "Status" not in target:
            target["Status"] = {"select": {"name": "Raw"}}

        # Log unmapped properties
        for name in source_props:
            if name not in PROPERTY_MAP and source_props[name].get("type") != "title":
                logger.debug("Unmapped source property: '%s' (type: %s)",
                             name, source_props[name].get("type"))

        return target

    def _fetch_blocks_recursive(self, page_id: str) -> list[dict]:
        """Fetch all blocks for a page, recursively fetching children."""
        blocks = self.client.get_page_blocks(page_id)
        result: list[dict] = []
        for block in blocks:
            result.append(block)
            if block.get("has_children"):
                children = self._fetch_blocks_recursive(block["id"])
                # Nest children under parent block's type key
                block_type = block.get("type", "")
                if block_type in block and isinstance(block[block_type], dict):
                    block[block_type]["children"] = [
                        self._strip_block_metadata(c) for c in children
                    ]
        return result

    @staticmethod
    def _strip_block_metadata(block: dict) -> dict:
        """Remove read-only fields from a block, keeping type + content.

        Also converts Notion-hosted image blocks (type: file) to external
        format so they can be re-created via the API. Notion will download
        and re-host the image from the external URL during page creation.
        """
        cleaned: dict[str, Any] = {}
        for key, value in block.items():
            if key not in _BLOCK_READONLY_FIELDS:
                cleaned[key] = value

        # Convert image.file -> image.external (file type is read-only)
        if cleaned.get("type") == "image":
            img = cleaned.get("image", {})
            if img.get("type") == "file" and "file" in img:
                url = img["file"].get("url", "")
                if url:
                    cleaned["image"] = {
                        "type": "external",
                        "external": {"url": url},
                        "caption": img.get("caption", []),
                    }

        # Strip list_start_index from numbered_list_item (read-only)
        block_type = cleaned.get("type", "")
        if block_type == "numbered_list_item" and block_type in cleaned:
            cleaned[block_type].pop("list_start_index", None)

        return cleaned

    def _copy_content_blocks(self, page_id: str, blocks: list[dict]) -> None:
        """Append blocks to a page in chunks of 100 (Notion API limit)."""
        for i in range(0, len(blocks), 100):
            chunk = blocks[i : i + 100]
            self.client.append_blocks(page_id, chunk)

    def _mark_source_migrated(self, source_page_id: str, new_page_id: str) -> None:
        """Append a callout block to the source page noting migration."""
        callout = {
            "type": "callout",
            "callout": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": f"Migrated to KB: {new_page_id}",
                        },
                    }
                ],
                "icon": {"type": "emoji", "emoji": "➡️"},
            },
        }
        self.client.append_blocks(source_page_id, [callout])

    def _remap_relations(
        self,
        id_mapping: dict[str, str],
        old_relations: dict[str, list[str]],
    ) -> tuple[int, int]:
        """Re-establish Related Notes using the old-to-new ID mapping.

        Returns (remapped_count, skipped_count).
        """
        remapped = 0
        skipped = 0

        for new_page_id, old_related_ids in old_relations.items():
            new_related: list[str] = []
            for old_id in old_related_ids:
                if old_id in id_mapping:
                    new_related.append(id_mapping[old_id])
                    remapped += 1
                else:
                    logger.warning(
                        "Relation target %s not in migrated set, skipping", old_id
                    )
                    skipped += 1

            if new_related:
                try:
                    self.client.add_relation(new_page_id, "Related Notes", new_related)
                except Exception as exc:
                    logger.error("Failed to remap relations for %s: %s", new_page_id, exc)
                    skipped += len(new_related)
                    remapped -= len(new_related)

        return remapped, skipped
