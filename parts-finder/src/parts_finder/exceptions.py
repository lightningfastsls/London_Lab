"""Exceptions for the Parts Finder service.

Two-level hierarchy so callers can catch broadly (GovApiError) or
narrowly (PlateNotFoundError) depending on their error-handling needs.
"""

from __future__ import annotations


class GovApiError(Exception):
    """Wraps HTTP errors, timeouts, and malformed responses from data.gov.il."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PlateNotFoundError(GovApiError):
    """Raised when a license plate number is not found in the registry."""

    def __init__(self, plate: str) -> None:
        super().__init__(f"Plate not found: {plate}")
        self.plate = plate
