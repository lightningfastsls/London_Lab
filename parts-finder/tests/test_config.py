"""Tests for AppConfig."""

from __future__ import annotations

import unittest

from parts_finder.config import AppConfig


class TestAppConfig(unittest.TestCase):
    """Validate AppConfig construction, defaults, validation, and immutability."""

    def test_default_construction(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.gov_api_timeout_s, 10.0)
        self.assertEqual(cfg.server_port, 8000)
        self.assertEqual(cfg.server_host, "127.0.0.1")
        self.assertTrue(cfg.cache_enabled)
        self.assertEqual(cfg.cache_ttl_minutes, 60)
        self.assertEqual(cfg.ai_max_retries, 3)
        self.assertFalse(cfg.server_debug)

    def test_custom_override(self) -> None:
        cfg = AppConfig(
            gov_api_resource_id="abc-123",
            gov_api_timeout_s=30.0,
            server_port=9090,
            cache_ttl_minutes=120,
            ai_model="claude-opus-4-20250514",
        )
        self.assertEqual(cfg.gov_api_resource_id, "abc-123")
        self.assertEqual(cfg.gov_api_timeout_s, 30.0)
        self.assertEqual(cfg.server_port, 9090)
        self.assertEqual(cfg.cache_ttl_minutes, 120)
        self.assertEqual(cfg.ai_model, "claude-opus-4-20250514")

    # --- Validation error tests ---

    def test_empty_resource_id_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(gov_api_resource_id="")
        self.assertIn("gov_api_resource_id must not be empty", str(ctx.exception))

    def test_empty_base_url_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(gov_api_base_url="  ")
        self.assertIn("gov_api_base_url must not be empty", str(ctx.exception))

    def test_negative_timeout_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(gov_api_timeout_s=-1.0)
        self.assertIn("gov_api_timeout_s must be positive", str(ctx.exception))

    def test_zero_timeout_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(gov_api_timeout_s=0.0)
        self.assertIn("gov_api_timeout_s must be positive", str(ctx.exception))

    def test_invalid_port_zero_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(server_port=0)
        self.assertIn("server_port must be between 1 and 65535", str(ctx.exception))

    def test_invalid_port_too_high_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(server_port=70000)
        self.assertIn("server_port must be between 1 and 65535", str(ctx.exception))

    def test_negative_cache_ttl_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(cache_ttl_minutes=-5)
        self.assertIn("cache_ttl_minutes must be positive", str(ctx.exception))

    def test_negative_retries_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(ai_max_retries=-1)
        self.assertIn("ai_max_retries must be >= 0", str(ctx.exception))

    def test_empty_misses_path_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(misses_log_path="")
        self.assertIn("misses_log_path must not be empty", str(ctx.exception))

    def test_whitespace_misses_path_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AppConfig(misses_log_path="   ")
        self.assertIn("misses_log_path must not be empty", str(ctx.exception))

    # --- Immutability ---

    def test_frozen_immutability(self) -> None:
        cfg = AppConfig()
        with self.assertRaises(Exception):
            cfg.server_port = 9999  # type: ignore


if __name__ == "__main__":
    unittest.main()
