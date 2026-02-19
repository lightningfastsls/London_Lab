"""Tests for Notion Notes Phase 1.2 — Notes-to-KB migration.

All tests use mocks — no live API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Ensure notion_notes package is importable from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notion_notes.migration import (
    MigrationConfig,
    MigrationReport,
    NoteMigrator,
    PROPERTY_MAP,
    map_status_value,
    _BLOCK_READONLY_FIELDS,
)
from notion_notes.notion_client import NotionPage


# ===================================================================
# Helpers
# ===================================================================


def _make_page(
    page_id: str = "page-1",
    title: str = "Test Page",
    properties: dict | None = None,
) -> NotionPage:
    """Build a minimal NotionPage for testing."""
    return NotionPage(
        id=page_id,
        title=title,
        properties=properties or {},
    )


def _make_client(dry_run: bool = False) -> MagicMock:
    """Build a MagicMock that behaves like NotionClientWrapper."""
    client = MagicMock()
    client.dry_run = dry_run
    client.extract_property = MagicMock(return_value=None)
    return client


def _make_config(**overrides) -> MigrationConfig:
    """Build a MigrationConfig with defaults."""
    defaults = {
        "source_db_id": "src-db",
        "target_db_id": "tgt-db",
    }
    defaults.update(overrides)
    return MigrationConfig(**defaults)


# ===================================================================
# 1. TestMigrationConfig — frozen dataclass, default values
# ===================================================================


class TestMigrationConfig:
    def test_default_values(self) -> None:
        cfg = MigrationConfig(source_db_id="src", target_db_id="tgt")
        assert cfg.source_db_id == "src"
        assert cfg.target_db_id == "tgt"
        assert cfg.dry_run is False
        assert cfg.require_course is True
        assert cfg.include_projects == ()
        assert cfg.include_pages == ()
        assert cfg.exclude_pages == ()

    def test_frozen(self) -> None:
        cfg = MigrationConfig(source_db_id="src", target_db_id="tgt")
        with pytest.raises(AttributeError):
            cfg.source_db_id = "other"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = MigrationConfig(
            source_db_id="src",
            target_db_id="tgt",
            dry_run=True,
            require_course=False,
            include_projects=("PSYC101",),
            exclude_pages=("Scratch",),
        )
        assert cfg.dry_run is True
        assert cfg.require_course is False
        assert cfg.include_projects == ("PSYC101",)
        assert cfg.exclude_pages == ("Scratch",)


# ===================================================================
# 2. TestMigrationReport — summary() output, default counts
# ===================================================================


class TestMigrationReport:
    def test_default_counts(self) -> None:
        report = MigrationReport()
        assert report.total_source == 0
        assert report.migrated == 0
        assert report.errors == []
        assert report.id_mapping == {}

    def test_summary_no_errors(self) -> None:
        report = MigrationReport(total_source=10, filtered_out=3, migrated=7)
        s = report.summary()
        assert "Source pages found: 10" in s
        assert "Filtered out:       3" in s
        assert "Migrated:           7" in s
        assert "Error details" not in s

    def test_summary_with_errors(self) -> None:
        report = MigrationReport(
            total_source=5,
            migrated=3,
            errors=[{"page_id": "abc", "error": "API failed"}],
        )
        s = report.summary()
        assert "Errors:             1" in s
        assert "abc: API failed" in s


# ===================================================================
# 3. TestShouldMigrate — filtering logic
# ===================================================================


class TestShouldMigrate:
    def test_exclude_list_has_priority(self) -> None:
        config = _make_config(exclude_pages=("Bad Page",))
        client = _make_client()
        migrator = NoteMigrator(client, config)
        page = _make_page(title="Bad Page")
        assert migrator._should_migrate(page) is False

    def test_include_pages_allows_match(self) -> None:
        config = _make_config(include_pages=("Good Page",), require_course=False)
        client = _make_client()
        migrator = NoteMigrator(client, config)
        assert migrator._should_migrate(_make_page(title="Good Page")) is True
        assert migrator._should_migrate(_make_page(title="Other Page")) is False

    def test_include_projects_matches_course(self) -> None:
        config = _make_config(include_projects=("PSYC101",), require_course=False)
        client = _make_client()
        client.extract_property = MagicMock(return_value=["PSYC101", "NEURO200"])
        migrator = NoteMigrator(client, config)
        page = _make_page(title="Any", properties={
            "UNI Courses": {"type": "relation", "relation": [{"id": "course-1"}]},
        })
        assert migrator._should_migrate(page) is True

    def test_include_projects_no_match(self) -> None:
        config = _make_config(include_projects=("PSYC101",), require_course=False)
        client = _make_client()
        client.extract_property = MagicMock(return_value=["BIO300"])
        migrator = NoteMigrator(client, config)
        assert migrator._should_migrate(_make_page(title="Any")) is False

    def test_require_course_handled_by_server_filter(self) -> None:
        """require_course filtering is done server-side in _query_source_pages,
        not in _should_migrate. Pages returned by query already pass the filter."""
        config = _make_config(require_course=True)
        client = _make_client()
        migrator = NoteMigrator(client, config)
        # _should_migrate always passes for require_course (server does the filtering)
        assert migrator._should_migrate(_make_page(title="Any Page")) is True

    def test_no_filters_allows_all(self) -> None:
        config = _make_config(require_course=False)
        client = _make_client()
        migrator = NoteMigrator(client, config)
        assert migrator._should_migrate(_make_page(title="Anything")) is True

    def test_exclude_overrides_include(self) -> None:
        config = _make_config(
            include_pages=("Page A",),
            exclude_pages=("Page A",),
            require_course=False,
        )
        client = _make_client()
        migrator = NoteMigrator(client, config)
        assert migrator._should_migrate(_make_page(title="Page A")) is False


# ===================================================================
# 4. TestMapProperties — Tags, UNI Courses, Evergreen, Status, unknowns
# ===================================================================


class TestMapProperties:
    def _make_migrator(self) -> NoteMigrator:
        return NoteMigrator(_make_client(), _make_config())

    def test_maps_multi_select_tags(self) -> None:
        m = self._make_migrator()
        source = {
            "Tags": {
                "type": "multi_select",
                "multi_select": [{"name": "neuro"}, {"name": "memory"}],
            }
        }
        result = m._map_properties(source)
        assert result["Tags"] == {
            "multi_select": [{"name": "neuro"}, {"name": "memory"}],
        }

    def test_uni_courses_not_mapped(self) -> None:
        """UNI Courses is not in PROPERTY_MAP (KB has no UNI Course property)."""
        m = self._make_migrator()
        source = {
            "UNI Courses": {
                "type": "multi_select",
                "multi_select": [{"name": "PSYC101"}],
            }
        }
        result = m._map_properties(source)
        assert "UNI Course" not in result
        assert "UNI Courses" not in result

    def test_maps_checkbox_evergreen(self) -> None:
        m = self._make_migrator()
        source = {
            "Evergreen": {"type": "checkbox", "checkbox": True},
        }
        result = m._map_properties(source)
        assert result["Evergreen"] == {"checkbox": True}

    def test_maps_status_with_transform(self) -> None:
        m = self._make_migrator()
        source = {
            "Status": {
                "type": "status",
                "status": {"name": "Done"},
            }
        }
        result = m._map_properties(source)
        assert result["Status"] == {"select": {"name": "Processed"}}

    def test_sets_kb_defaults(self) -> None:
        m = self._make_migrator()
        result = m._map_properties({})
        assert result["Is Atomic"] == {"checkbox": False}
        assert result["Status"] == {"select": {"name": "Raw"}}

    def test_unknown_properties_skipped(self) -> None:
        m = self._make_migrator()
        source = {
            "Weird Prop": {"type": "number", "number": 42},
        }
        result = m._map_properties(source)
        assert "Weird Prop" not in result


# ===================================================================
# 5. TestMapStatusValue — value transforms
# ===================================================================


class TestMapStatusValue:
    def test_not_started_maps_to_raw(self) -> None:
        assert map_status_value("Not started") == "Raw"

    def test_in_progress_maps(self) -> None:
        assert map_status_value("In progress") == "In Progress"

    def test_done_maps_to_processed(self) -> None:
        assert map_status_value("Done") == "Processed"

    def test_none_maps_to_raw(self) -> None:
        assert map_status_value(None) == "Raw"

    def test_empty_maps_to_raw(self) -> None:
        assert map_status_value("") == "Raw"

    def test_unknown_maps_to_raw(self) -> None:
        assert map_status_value("Something Else") == "Raw"


# ===================================================================
# 6. TestStripBlockMetadata — removes read-only fields
# ===================================================================


class TestStripBlockMetadata:
    def test_removes_readonly_fields(self) -> None:
        block = {
            "id": "block-1",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": "hello"}}]},
            "created_time": "2026-01-01",
            "last_edited_time": "2026-01-02",
            "has_children": False,
            "parent": {"page_id": "page-1"},
            "object": "block",
            "archived": False,
            "in_trash": False,
        }
        cleaned = NoteMigrator._strip_block_metadata(block)
        assert "id" not in cleaned
        assert "created_time" not in cleaned
        assert "parent" not in cleaned
        assert "has_children" not in cleaned
        assert "object" not in cleaned
        assert "archived" not in cleaned
        assert "in_trash" not in cleaned

    def test_preserves_type_and_content(self) -> None:
        block = {
            "id": "block-1",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": "hello"}}]},
            "has_children": False,
        }
        cleaned = NoteMigrator._strip_block_metadata(block)
        assert cleaned["type"] == "paragraph"
        assert cleaned["paragraph"]["rich_text"][0]["text"]["content"] == "hello"

    def test_preserves_unknown_fields(self) -> None:
        block = {
            "id": "block-1",
            "type": "code",
            "code": {"language": "python", "rich_text": []},
            "custom_field": "keep_me",
        }
        cleaned = NoteMigrator._strip_block_metadata(block)
        assert cleaned["custom_field"] == "keep_me"

    def test_converts_image_file_to_external(self) -> None:
        block = {
            "id": "block-1",
            "type": "image",
            "image": {
                "type": "file",
                "file": {"url": "https://s3.amazonaws.com/signed-url"},
                "caption": [{"plain_text": "diagram"}],
            },
            "has_children": False,
        }
        cleaned = NoteMigrator._strip_block_metadata(block)
        assert cleaned["image"]["type"] == "external"
        assert cleaned["image"]["external"]["url"] == "https://s3.amazonaws.com/signed-url"
        assert cleaned["image"]["caption"] == [{"plain_text": "diagram"}]

    def test_keeps_external_image_unchanged(self) -> None:
        block = {
            "id": "block-1",
            "type": "image",
            "image": {
                "type": "external",
                "external": {"url": "https://example.com/img.png"},
                "caption": [],
            },
        }
        cleaned = NoteMigrator._strip_block_metadata(block)
        assert cleaned["image"]["type"] == "external"
        assert cleaned["image"]["external"]["url"] == "https://example.com/img.png"

    def test_strips_list_start_index(self) -> None:
        block = {
            "id": "block-1",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [{"text": {"content": "item"}}],
                "list_start_index": 2,
            },
        }
        cleaned = NoteMigrator._strip_block_metadata(block)
        assert "list_start_index" not in cleaned["numbered_list_item"]
        assert cleaned["numbered_list_item"]["rich_text"][0]["text"]["content"] == "item"


# ===================================================================
# 7. TestMigrateFullPipeline — end-to-end with mocked client
# ===================================================================


class TestMigrateFullPipeline:
    def test_migrates_two_pages(self) -> None:
        """Full pipeline: 2 pages queried, both pass filter, both created."""
        page1 = _make_page(
            "old-1", "Page One",
            properties={
                "Name": {"type": "title", "title": [{"plain_text": "Page One"}]},
                "Tags": {
                    "type": "multi_select",
                    "multi_select": [{"name": "bio"}],
                },
            },
        )
        page2 = _make_page(
            "old-2", "Page Two",
            properties={
                "Name": {"type": "title", "title": [{"plain_text": "Page Two"}]},
            },
        )

        client = _make_client()
        client.query_database.return_value = [page1, page2]
        client.get_page_blocks.return_value = [
            {
                "id": "b1",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": "text"}}]},
                "has_children": False,
            }
        ]
        client.create_page.side_effect = ["new-1", "new-2"]
        # extract_property returns None for Related Notes (no relations)
        client.extract_property.return_value = None

        config = _make_config(require_course=False)
        migrator = NoteMigrator(client, config)
        report = migrator.migrate()

        assert report.total_source == 2
        assert report.migrated == 2
        assert report.id_mapping == {"old-1": "new-1", "old-2": "new-2"}
        assert client.create_page.call_count == 2
        # Source pages should be marked as migrated
        assert client.append_blocks.call_count == 2

    def test_relations_remapped_in_second_pass(self) -> None:
        """Pages with Related Notes get relations re-established."""
        page1 = _make_page("old-1", "P1", properties={
            "Name": {"type": "title", "title": [{"plain_text": "P1"}]},
        })
        page2 = _make_page("old-2", "P2", properties={
            "Name": {"type": "title", "title": [{"plain_text": "P2"}]},
        })

        client = _make_client()
        client.query_database.return_value = [page1, page2]
        client.get_page_blocks.return_value = []
        client.create_page.side_effect = ["new-1", "new-2"]

        # page1 has relation to page2
        def extract_prop(props, name):
            if name == "Related Notes":
                # Check which page we're extracting from by checking the actual props dict
                if props is page1.properties:
                    return ["old-2"]
                return None
            return None

        client.extract_property = MagicMock(side_effect=extract_prop)

        config = _make_config(require_course=False)
        migrator = NoteMigrator(client, config)
        report = migrator.migrate()

        assert report.relations_remapped == 1
        # add_relation should have been called for new-1 -> new-2
        client.add_relation.assert_called_once_with("new-1", "Related Notes", ["new-2"])


# ===================================================================
# 8. TestDryRunMigration — no writes called
# ===================================================================


class TestDryRunMigration:
    def test_dry_run_skips_writes(self) -> None:
        page = _make_page("old-1", "Test", properties={
            "Name": {"type": "title", "title": [{"plain_text": "Test"}]},
        })

        client = _make_client(dry_run=True)
        client.query_database.return_value = [page]
        client.extract_property.return_value = None

        config = _make_config(dry_run=True, require_course=False)
        migrator = NoteMigrator(client, config)
        report = migrator.migrate()

        assert report.skipped_dry_run == 1
        assert report.migrated == 0
        assert "old-1" in report.id_mapping
        # No write methods should have been called
        client.create_page.assert_not_called()
        client.append_blocks.assert_not_called()
        client.add_relation.assert_not_called()

    def test_dry_run_with_filtering(self) -> None:
        pages = [
            _make_page("p1", "Include Me"),
            _make_page("p2", "Exclude Me"),
        ]

        client = _make_client(dry_run=True)
        client.query_database.return_value = pages
        client.extract_property.return_value = None

        config = _make_config(
            dry_run=True,
            require_course=False,
            exclude_pages=("Exclude Me",),
        )
        migrator = NoteMigrator(client, config)
        report = migrator.migrate()

        assert report.total_source == 2
        assert report.filtered_out == 1
        assert report.skipped_dry_run == 1


# ===================================================================
# 9. TestMigrationErrorHandling — failed page doesn't halt
# ===================================================================


class TestMigrationErrorHandling:
    def test_error_collected_and_continues(self) -> None:
        page1 = _make_page("fail-1", "Fail Page")
        page2 = _make_page("ok-2", "OK Page", properties={
            "Name": {"type": "title", "title": [{"plain_text": "OK Page"}]},
        })

        client = _make_client()
        client.query_database.return_value = [page1, page2]
        client.get_page_blocks.side_effect = [
            Exception("API error"),
            [],  # OK for page2
        ]
        client.create_page.return_value = "new-2"
        client.extract_property.return_value = None

        config = _make_config(require_course=False)
        migrator = NoteMigrator(client, config)
        report = migrator.migrate()

        assert len(report.errors) == 1
        assert report.errors[0]["page_id"] == "fail-1"
        assert "API error" in report.errors[0]["error"]
        assert report.migrated == 1

    def test_all_pages_fail_gracefully(self) -> None:
        pages = [_make_page(f"p{i}", f"Page {i}") for i in range(3)]

        client = _make_client()
        client.query_database.return_value = pages
        client.get_page_blocks.side_effect = Exception("Down")
        client.extract_property.return_value = None

        config = _make_config(require_course=False)
        migrator = NoteMigrator(client, config)
        report = migrator.migrate()

        assert len(report.errors) == 3
        assert report.migrated == 0
