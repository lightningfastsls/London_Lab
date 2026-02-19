"""Tests for Notion Notes Phase 1.1 — client wrappers and config.

All tests use mocks — no live API calls.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure notion_notes package is importable from repo root
# ---------------------------------------------------------------------------
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notion_notes.config import NotionNotesConfig, load_config, _parse_env_file
from notion_notes.claude_client import ClaudeClient
from notion_notes.notion_client import NotionClientWrapper, NotionPage


# ===================================================================
# 1. NotionClientWrapper initializes with a token
# ===================================================================


class TestNotionClientInit:
    def test_init_creates_client(self) -> None:
        with patch("notion_notes.notion_client.Client") as MockClient:
            wrapper = NotionClientWrapper(token="secret_test_token")
            MockClient.assert_called_once_with(
                auth="secret_test_token", notion_version="2022-06-28"
            )
            assert wrapper.dry_run is False

    def test_init_respects_dry_run(self) -> None:
        with patch("notion_notes.notion_client.Client"):
            wrapper = NotionClientWrapper(token="tok", dry_run=True)
            assert wrapper.dry_run is True


# ===================================================================
# 2. blocks_to_markdown converts all supported block types
# ===================================================================


class TestBlocksToMarkdown:
    def _block(self, block_type: str, rich_text: list[dict], **extra) -> dict:
        """Helper to build a Notion block dict."""
        content = {"rich_text": rich_text}
        content.update(extra)
        return {"type": block_type, block_type: content}

    def _rt(self, text: str) -> list[dict]:
        """Shortcut for a rich_text array with one plain_text item."""
        return [{"plain_text": text}]

    def test_paragraph(self) -> None:
        blocks = [self._block("paragraph", self._rt("Hello world"))]
        assert NotionClientWrapper.blocks_to_markdown(blocks) == "Hello world"

    def test_headings(self) -> None:
        blocks = [
            self._block("heading_1", self._rt("H1")),
            self._block("heading_2", self._rt("H2")),
            self._block("heading_3", self._rt("H3")),
        ]
        md = NotionClientWrapper.blocks_to_markdown(blocks)
        assert "# H1" in md
        assert "## H2" in md
        assert "### H3" in md

    def test_bulleted_list(self) -> None:
        blocks = [
            self._block("bulleted_list_item", self._rt("Item A")),
            self._block("bulleted_list_item", self._rt("Item B")),
        ]
        md = NotionClientWrapper.blocks_to_markdown(blocks)
        assert "- Item A" in md
        assert "- Item B" in md

    def test_numbered_list(self) -> None:
        blocks = [
            self._block("numbered_list_item", self._rt("First")),
            self._block("numbered_list_item", self._rt("Second")),
        ]
        md = NotionClientWrapper.blocks_to_markdown(blocks)
        assert "1. First" in md
        assert "2. Second" in md

    def test_code_block(self) -> None:
        blocks = [
            self._block("code", self._rt("print('hi')"), language="python"),
        ]
        md = NotionClientWrapper.blocks_to_markdown(blocks)
        assert "```python" in md
        assert "print('hi')" in md
        assert md.strip().endswith("```")

    def test_divider(self) -> None:
        blocks = [{"type": "divider", "divider": {}}]
        assert NotionClientWrapper.blocks_to_markdown(blocks) == "---"

    def test_unsupported_block(self) -> None:
        blocks = [{"type": "table", "table": {}}]
        md = NotionClientWrapper.blocks_to_markdown(blocks)
        assert "[unsupported: table]" in md

    def test_mixed_blocks(self) -> None:
        blocks = [
            self._block("heading_1", self._rt("Title")),
            self._block("paragraph", self._rt("Body text")),
            self._block("bulleted_list_item", self._rt("Bullet")),
            {"type": "divider", "divider": {}},
        ]
        md = NotionClientWrapper.blocks_to_markdown(blocks)
        lines = md.split("\n")
        assert lines[0] == "# Title"
        assert lines[1] == "Body text"
        assert lines[2] == "- Bullet"
        assert lines[3] == "---"


# ===================================================================
# 3. extract_title scans for type: "title"
# ===================================================================


class TestExtractTitle:
    def test_finds_title_by_type(self) -> None:
        props = {
            "Tags": {"type": "multi_select", "multi_select": []},
            "Name": {"type": "title", "title": [{"plain_text": "My Page"}]},
        }
        assert NotionClientWrapper.extract_title(props) == "My Page"

    def test_returns_empty_when_no_title(self) -> None:
        props = {"Tags": {"type": "multi_select", "multi_select": []}}
        assert NotionClientWrapper.extract_title(props) == ""

    def test_handles_custom_title_key(self) -> None:
        # Title property might not be called "Name"
        props = {
            "Concept": {"type": "title", "title": [{"plain_text": "Neural Plasticity"}]},
        }
        assert NotionClientWrapper.extract_title(props) == "Neural Plasticity"


# ===================================================================
# 4. extract_property handles multiple types
# ===================================================================


class TestExtractProperty:
    def test_multi_select(self) -> None:
        props = {
            "Tags": {
                "type": "multi_select",
                "multi_select": [{"name": "bio"}, {"name": "neuro"}],
            }
        }
        result = NotionClientWrapper.extract_property(props, "Tags")
        assert result == ["bio", "neuro"]

    def test_select(self) -> None:
        props = {"Domain": {"type": "select", "select": {"name": "Neuroscience"}}}
        assert NotionClientWrapper.extract_property(props, "Domain") == "Neuroscience"

    def test_select_none(self) -> None:
        props = {"Domain": {"type": "select", "select": None}}
        assert NotionClientWrapper.extract_property(props, "Domain") is None

    def test_checkbox(self) -> None:
        props = {"Is Atomic": {"type": "checkbox", "checkbox": True}}
        assert NotionClientWrapper.extract_property(props, "Is Atomic") is True

    def test_date(self) -> None:
        props = {
            "Last Processed": {
                "type": "date",
                "date": {"start": "2026-02-18"},
            }
        }
        assert NotionClientWrapper.extract_property(props, "Last Processed") == "2026-02-18"

    def test_relation(self) -> None:
        props = {
            "Related Notes": {
                "type": "relation",
                "relation": [{"id": "abc"}, {"id": "def"}],
            }
        }
        assert NotionClientWrapper.extract_property(props, "Related Notes") == ["abc", "def"]

    def test_missing_property(self) -> None:
        assert NotionClientWrapper.extract_property({}, "Nonexistent") is None


# ===================================================================
# 5. dry_run prevents all write operations
# ===================================================================


class TestDryRun:
    def _make_wrapper(self) -> NotionClientWrapper:
        with patch("notion_notes.notion_client.Client"):
            return NotionClientWrapper(token="tok", dry_run=True)

    def test_create_page_returns_none(self) -> None:
        w = self._make_wrapper()
        result = w.create_page("db_id", {"Title": {}})
        assert result is None

    def test_update_page_properties_noop(self) -> None:
        w = self._make_wrapper()
        # Should not raise
        w.update_page_properties("page_id", {"Status": {}})

    def test_append_blocks_noop(self) -> None:
        w = self._make_wrapper()
        w.append_blocks("page_id", [{"type": "paragraph"}])

    def test_add_relation_noop(self) -> None:
        w = self._make_wrapper()
        w.add_relation("page_id", "Related Notes", ["id1", "id2"])


# ===================================================================
# 6. Rate limiter sleeps to enforce 3 req/s
# ===================================================================


class TestRateLimiter:
    def test_sleeps_between_rapid_calls(self) -> None:
        with patch("notion_notes.notion_client.Client"):
            wrapper = NotionClientWrapper(token="tok", rate_limit_rps=3.0)

        mock_func = MagicMock(return_value={"results": [], "has_more": False})

        with patch("notion_notes.notion_client.time") as mock_time:
            # Simulate: first call at t=0, second call 0.1s later (too fast)
            mock_time.monotonic = MagicMock(side_effect=[0.0, 0.0, 0.1, 0.1 + 1 / 3])
            mock_time.sleep = MagicMock()

            wrapper._rate_limited_call(mock_func)
            wrapper._rate_limited_call(mock_func)

            # Should have slept at least once to respect rate limit
            assert mock_time.sleep.called


# ===================================================================
# 7. ClaudeClient.prompt returns text response
# ===================================================================


class TestClaudeClientPrompt:
    def test_prompt_returns_text(self) -> None:
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("notion_notes.claude_client.Anthropic", return_value=mock_anthropic):
            client = ClaudeClient(api_key="test_key")

        result = client.prompt("What is 2+2?")
        assert result == "Hello from Claude"
        mock_anthropic.messages.create.assert_called_once()

    def test_prompt_with_system(self) -> None:
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("notion_notes.claude_client.Anthropic", return_value=mock_anthropic):
            client = ClaudeClient(api_key="test_key")

        client.prompt("hello", system="You are a tagger.")
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a tagger."


# ===================================================================
# 8. ClaudeClient.prompt_json parses valid JSON
# ===================================================================


class TestClaudeClientPromptJson:
    def test_valid_json(self) -> None:
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"domain": "Neuroscience", "tags": ["synaptic"]}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("notion_notes.claude_client.Anthropic", return_value=mock_anthropic):
            client = ClaudeClient(api_key="test_key")

        result = client.prompt_json("tag this note")
        assert result == {"domain": "Neuroscience", "tags": ["synaptic"]}

    def test_invalid_json_raises_value_error(self) -> None:
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not JSON at all")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("notion_notes.claude_client.Anthropic", return_value=mock_anthropic):
            client = ClaudeClient(api_key="test_key")

        with pytest.raises(ValueError, match="not valid JSON"):
            client.prompt_json("tag this")


# ===================================================================
# 9. Config loads from .env file
# ===================================================================


class TestConfigLoad:
    def test_loads_from_env_vars(self) -> None:
        env = {
            "NOTION_TOKEN": "ntn_test123",
            "ANTHROPIC_API_KEY": "sk-ant-test456",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
        assert cfg.notion_token == "ntn_test123"
        assert cfg.anthropic_api_key == "sk-ant-test456"
        assert cfg.dry_run is False

    def test_loads_from_env_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            'NOTION_TOKEN="ntn_from_file"\n'
            "ANTHROPIC_API_KEY=sk-ant-from_file\n"
        )
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config(env_path=env_file)
        assert cfg.notion_token == "ntn_from_file"
        assert cfg.anthropic_api_key == "sk-ant-from_file"

    def test_env_file_takes_precedence_over_env_vars(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("NOTION_TOKEN=from_file\nANTHROPIC_API_KEY=from_file\n")
        with patch.dict(os.environ, {"NOTION_TOKEN": "from_env", "ANTHROPIC_API_KEY": "from_env"}, clear=True):
            cfg = load_config(env_path=env_file)
        # .env file wins over system env vars (see config.py)
        assert cfg.notion_token == "from_file"

    def test_dry_run_flag(self) -> None:
        env = {"NOTION_TOKEN": "tok", "ANTHROPIC_API_KEY": "key"}
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config(dry_run=True)
        assert cfg.dry_run is True


# ===================================================================
# 10. Config raises ValueError when required vars missing
# ===================================================================


class TestConfigValidation:
    def test_missing_notion_token(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=True):
            with pytest.raises(ValueError, match="NOTION_TOKEN"):
                load_config()

    def test_missing_anthropic_key(self) -> None:
        with patch.dict(os.environ, {"NOTION_TOKEN": "tok"}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                load_config()

    def test_missing_both(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="NOTION_TOKEN.*ANTHROPIC_API_KEY"):
                load_config()

    def test_empty_token_rejected(self) -> None:
        with pytest.raises(ValueError, match="notion_token must not be empty"):
            NotionNotesConfig(notion_token="", anthropic_api_key="key")

    def test_invalid_rate_limit(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            NotionNotesConfig(
                notion_token="tok",
                anthropic_api_key="key",
                notion_rate_limit_rps=-1.0,
            )


# ===================================================================
# 11. Bonus: _parse_env_file edge cases
# ===================================================================


class TestParseEnvFile:
    def test_handles_comments_and_blanks(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=value\n")
        result = _parse_env_file(env_file)
        assert result == {"KEY": "value"}

    def test_strips_quotes(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("A='single'\nB=\"double\"\n")
        result = _parse_env_file(env_file)
        assert result["A"] == "single"
        assert result["B"] == "double"

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        result = _parse_env_file(tmp_path / "nope")
        assert result == {}
