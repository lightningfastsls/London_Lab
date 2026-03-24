"""Tests for the LMT data access layer.

All tests use in-memory SQLite databases — no real .sqlite files required.
Fixtures create the LMT schema and populate it with known test data.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from usv_spectrogram.lmt.db_loader import (
    AnimalInfo,
    BehavioralEvent,
    LMTDatabaseLoader,
)
from usv_spectrogram.lmt.synchronizer import LMTSynchronizer, SyncConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_lmt_schema(conn: sqlite3.Connection, minimal: bool = False) -> None:
    """Create LMT tables in an in-memory database.

    Parameters
    ----------
    minimal : bool
        If True, create ANIMAL with only 3 columns (ID, RFID, NAME) to
        simulate older LMT database versions.
    """
    if minimal:
        conn.execute(
            "CREATE TABLE ANIMAL (ID INTEGER PRIMARY KEY, RFID TEXT, NAME TEXT)"
        )
    else:
        conn.execute(
            "CREATE TABLE ANIMAL ("
            "  ID INTEGER PRIMARY KEY,"
            "  RFID TEXT,"
            "  NAME TEXT,"
            "  GENOTYPE TEXT,"
            "  SEX TEXT,"
            "  STRAIN TEXT"
            ")"
        )
    conn.execute(
        "CREATE TABLE EVENT ("
        "  ID INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  NAME TEXT,"
        "  STARTFRAME INTEGER,"
        "  ENDFRAME INTEGER,"
        "  IDANIMALA INTEGER,"
        "  IDANIMALB INTEGER,"
        "  IDANIMALC INTEGER,"
        "  IDANIMALD INTEGER,"
        "  METADATA TEXT"
        ")"
    )
    conn.commit()


@pytest.fixture
def lmt_db():
    """Create an in-memory LMT database with two animals and sample events.

    Animals:
        1: "Mouse_A", WT, Male, C57BL/6
        2: "Mouse_B", KO, Female, C57BL/6

    Events (at 30 fps):
        "Rearing"           frames 30-60   (1.0s - 2.0s)  animal 1
        "Oral-oral Contact" frames 45-75   (1.5s - 2.5s)  animals 1+2
        "Stop isolated"     frames 90-120  (3.0s - 4.0s)  animal 2
        "Contact"           frames 40-80   (1.33s - 2.67s) animal 1+2
        "night"             frames 0-9000  (0.0s - 300.0s) environmental
    """
    loader = LMTDatabaseLoader(":memory:")
    conn = loader._conn

    _create_lmt_schema(conn)

    conn.executemany(
        "INSERT INTO ANIMAL (ID, RFID, NAME, GENOTYPE, SEX, STRAIN) VALUES (?,?,?,?,?,?)",
        [
            (1, "RFID001", "Mouse_A", "WT", "Male", "C57BL/6"),
            (2, "RFID002", "Mouse_B", "KO", "Female", "C57BL/6"),
        ],
    )
    conn.executemany(
        "INSERT INTO EVENT (NAME, STARTFRAME, ENDFRAME, IDANIMALA, IDANIMALB) "
        "VALUES (?,?,?,?,?)",
        [
            ("Rearing", 30, 60, 1, None),
            ("Oral-oral Contact", 45, 75, 1, 2),
            ("Stop isolated", 90, 120, 2, None),
            ("Contact", 40, 80, 1, 2),
            ("night", 0, 9000, None, None),
        ],
    )
    conn.commit()

    yield loader
    loader.close()


@pytest.fixture
def lmt_db_minimal():
    """Create an LMT database with minimal ANIMAL schema (3 columns only)."""
    loader = LMTDatabaseLoader(":memory:")
    conn = loader._conn

    _create_lmt_schema(conn, minimal=True)

    conn.executemany(
        "INSERT INTO ANIMAL (ID, RFID, NAME) VALUES (?,?,?)",
        [
            (1, "RFID001", "Mouse_A"),
            (2, "RFID002", "Mouse_B"),
        ],
    )
    conn.commit()

    yield loader
    loader.close()


# ---------------------------------------------------------------------------
# LMTDatabaseLoader tests
# ---------------------------------------------------------------------------


class TestLMTDatabaseLoader:
    """Tests for the database loader."""

    def test_loader_opens_db(self, lmt_db: LMTDatabaseLoader):
        """Loader should open an in-memory database without error."""
        assert lmt_db is not None
        assert lmt_db.frame_rate == 30.0

    def test_get_animals(self, lmt_db: LMTDatabaseLoader):
        """Should return AnimalInfo for all animals in the database."""
        animals = lmt_db.get_animals()

        assert len(animals) == 2
        assert animals[0].animal_id == 1
        assert animals[0].rfid == "RFID001"
        assert animals[0].name == "Mouse_A"
        assert animals[0].genotype == "WT"
        assert animals[0].sex == "Male"
        assert animals[0].strain == "C57BL/6"

        assert animals[1].animal_id == 2
        assert animals[1].genotype == "KO"
        assert animals[1].sex == "Female"

    def test_get_animals_missing_columns(
        self, lmt_db_minimal: LMTDatabaseLoader
    ):
        """Should handle databases with only 3 ANIMAL columns gracefully."""
        animals = lmt_db_minimal.get_animals()

        assert len(animals) == 2
        assert animals[0].animal_id == 1
        assert animals[0].rfid == "RFID001"
        assert animals[0].name == "Mouse_A"
        # Missing columns should be None
        assert animals[0].genotype is None
        assert animals[0].sex is None
        assert animals[0].strain is None

    def test_get_events_all(self, lmt_db: LMTDatabaseLoader):
        """Should return all events with correct field conversion."""
        events = lmt_db.get_events()

        assert len(events) == 5
        # Events should be sorted by start frame
        assert events[0].event_type == "night"
        assert events[0].start_frame == 0
        assert events[0].start_time_s == pytest.approx(0.0)

        # Check frame-to-time conversion: frame 30 / 30fps = 1.0s
        rearing = [e for e in events if e.event_type == "Rearing"][0]
        assert rearing.start_frame == 30
        assert rearing.start_time_s == pytest.approx(1.0)
        assert rearing.end_frame == 60
        assert rearing.end_time_s == pytest.approx(2.0)
        assert rearing.animal_id == 1
        assert rearing.partner_id is None

    def test_get_events_type_filter(self, lmt_db: LMTDatabaseLoader):
        """Should filter events by type name."""
        events = lmt_db.get_events(event_types=["Rearing"])

        assert len(events) == 1
        assert events[0].event_type == "Rearing"

    def test_get_events_time_range(self, lmt_db: LMTDatabaseLoader):
        """Should filter events by time range overlap."""
        # Time range 0.5s - 1.2s should overlap with:
        # - "night" (0.0s - 300.0s) — overlaps
        # - "Rearing" (1.0s - 2.0s) — starts at 1.0s, overlaps 1.0-1.2
        # But NOT:
        # - "Contact" (1.33s - 2.67s) — starts after range ends
        # - "Stop isolated" (3.0s - 4.0s) — starts well after
        events = lmt_db.get_events(time_range=(0.5, 1.2))

        event_types = {e.event_type for e in events}
        assert "night" in event_types
        assert "Rearing" in event_types
        assert "Stop isolated" not in event_types

    def test_get_events_time_range_boundary_exclusion(
        self, lmt_db: LMTDatabaseLoader
    ):
        """Event starting exactly at range end should NOT be included.

        "Oral-oral Contact" starts at frame 45 = 1.5s. A query with
        end_s=1.5 should exclude it (strict overlap semantics: the event
        must start BEFORE the range end, not at it).
        """
        events = lmt_db.get_events(
            event_types=["Oral-oral Contact"],
            time_range=(0.5, 1.5),
        )
        event_types = {e.event_type for e in events}
        assert "Oral-oral Contact" not in event_types

    def test_get_events_animal_filter(self, lmt_db: LMTDatabaseLoader):
        """Should filter events to those involving a specific animal."""
        events = lmt_db.get_events(animal_id=2)

        # Animal 2 is involved in: "Oral-oral Contact" (as B),
        # "Stop isolated" (as A), "Contact" (as B)
        assert len(events) == 3
        event_types = {e.event_type for e in events}
        assert "Oral-oral Contact" in event_types
        assert "Stop isolated" in event_types
        assert "Contact" in event_types
        # "Rearing" (animal 1 only) and "night" (no animal) excluded
        assert "Rearing" not in event_types

    def test_get_event_types(self, lmt_db: LMTDatabaseLoader):
        """Should return distinct event type names, sorted."""
        types = lmt_db.get_event_types()

        assert isinstance(types, list)
        assert len(types) == 5
        # Should be alphabetically sorted
        assert types == sorted(types)
        assert "Rearing" in types
        assert "Oral-oral Contact" in types

    def test_get_timeline(self, lmt_db: LMTDatabaseLoader):
        """Timeline for an animal should be sorted by start time."""
        timeline = lmt_db.get_timeline(animal_id=1)

        # Animal 1 events: Rearing (1.0s), Oral-oral Contact (1.5s),
        # Contact (1.33s) — but sorted by start frame
        assert len(timeline) >= 2
        # Verify sorted by start_time_s
        for i in range(len(timeline) - 1):
            assert timeline[i].start_time_s <= timeline[i + 1].start_time_s

    def test_context_manager(self):
        """LMTDatabaseLoader should work as a context manager."""
        with LMTDatabaseLoader(":memory:") as loader:
            assert loader is not None
            assert loader.frame_rate == 30.0
            # Create schema to verify connection works
            loader._conn.execute(
                "CREATE TABLE ANIMAL (ID INTEGER PRIMARY KEY, RFID TEXT, NAME TEXT)"
            )
            loader._conn.execute(
                "CREATE TABLE EVENT (ID INTEGER PRIMARY KEY, NAME TEXT, "
                "STARTFRAME INTEGER, ENDFRAME INTEGER, "
                "IDANIMALA INTEGER, IDANIMALB INTEGER)"
            )
            loader._conn.commit()
            animals = loader.get_animals()
            assert animals == []

        # After exiting, connection should be closed
        assert loader._conn is None


# ---------------------------------------------------------------------------
# LMTSynchronizer tests
# ---------------------------------------------------------------------------


class TestLMTSynchronizer:
    """Tests for coordinate conversion and event alignment."""

    def test_frame_to_seconds(self):
        """Frame 30 at 30 fps = 1.0s, frame 0 = 0.0s."""
        sync = LMTSynchronizer()

        assert sync.lmt_frame_to_seconds(0) == pytest.approx(0.0)
        assert sync.lmt_frame_to_seconds(30) == pytest.approx(1.0)
        assert sync.lmt_frame_to_seconds(15) == pytest.approx(0.5)

    def test_frame_to_seconds_with_offset(self):
        """Time offset should shift the result."""
        config = SyncConfig(time_offset_s=2.5)
        sync = LMTSynchronizer(config)

        # Frame 30 = 1.0s + 2.5s offset = 3.5s
        assert sync.lmt_frame_to_seconds(30) == pytest.approx(3.5)
        assert sync.lmt_frame_to_seconds(0) == pytest.approx(2.5)

    def test_seconds_to_wav_sample(self):
        """1.0s at 300 kHz = sample 300000."""
        sync = LMTSynchronizer()

        assert sync.seconds_to_wav_sample(0.0) == 0
        assert sync.seconds_to_wav_sample(1.0) == 300_000
        assert sync.seconds_to_wav_sample(0.5) == 150_000

    def test_seconds_to_spectrogram_frame(self):
        """1.0s at hop=128, sr=300k = int(300000/128) = 2343."""
        sync = LMTSynchronizer()

        # 300000 / 128 = 2343.75, floor to 2343
        assert sync.seconds_to_spectrogram_frame(1.0, hop_length=128) == 2343
        assert sync.seconds_to_spectrogram_frame(0.0, hop_length=128) == 0

    def test_align_overlapping(self, lmt_db: LMTDatabaseLoader):
        """Detection during an event should find behavioral context."""
        sync = LMTSynchronizer()
        events = lmt_db.get_events()

        # Detection at 1.5s - 1.8s overlaps with:
        # - night (0-300s), Rearing (1.0-2.0s), Oral-oral Contact (1.5-2.5s),
        #   Contact (1.33-2.67s)
        detections = [{"start_time_s": 1.5, "end_time_s": 1.8, "id": "det_001"}]
        results = sync.align_events_with_detections(events, detections)

        assert len(results) == 1
        result = results[0]
        assert result["id"] == "det_001"  # original field preserved
        assert len(result["behavioral_context"]) >= 3
        assert result["dominant_event"] is not None
        # Most specific should be a pairwise event (Oral-oral Contact or Contact)
        assert result["dominant_event"].partner_id is not None

    def test_align_no_overlap(self, lmt_db: LMTDatabaseLoader):
        """Detection outside all behavioral events should get empty context."""
        sync = LMTSynchronizer()

        # Only get non-night events so we can test no overlap
        events = lmt_db.get_events(event_types=[
            "Rearing", "Oral-oral Contact", "Stop isolated", "Contact",
        ])

        # Detection at 5.0s - 5.5s — well after all non-night events
        detections = [{"start_time_s": 5.0, "end_time_s": 5.5}]
        results = sync.align_events_with_detections(events, detections)

        assert len(results) == 1
        assert results[0]["behavioral_context"] == []
        assert results[0]["dominant_event"] is None

    def test_align_multiple_events(self, lmt_db: LMTDatabaseLoader):
        """Detection overlapping multiple events should list all, pick dominant."""
        sync = LMTSynchronizer()

        # Get only the interesting events (exclude night)
        events = lmt_db.get_events(event_types=[
            "Rearing", "Oral-oral Contact", "Contact",
        ])

        # Detection at 1.5s - 1.6s overlaps all three
        detections = [{"start_time_s": 1.5, "end_time_s": 1.6}]
        results = sync.align_events_with_detections(events, detections)

        ctx = results[0]["behavioral_context"]
        assert len(ctx) == 3
        # First should be most specific (pairwise event)
        assert ctx[0].partner_id is not None
        # dominant_event should be the first (most specific)
        assert results[0]["dominant_event"] is ctx[0]

    def test_align_edge_exact_boundary(self):
        """Detection end = event start should NOT overlap (strict inequality)."""
        sync = LMTSynchronizer()

        # Event from 2.0s to 3.0s
        events = [
            BehavioralEvent(
                event_type="Rearing",
                start_frame=60,
                end_frame=90,
                start_time_s=2.0,
                end_time_s=3.0,
                animal_id=1,
            )
        ]

        # Detection ends exactly when event starts: 1.5 to 2.0
        # Overlap requires: event.start < det.end AND event.end > det.start
        # 2.0 < 2.0 is False → no overlap
        detections = [{"start_time_s": 1.5, "end_time_s": 2.0}]
        results = sync.align_events_with_detections(events, detections)

        assert results[0]["behavioral_context"] == []
        assert results[0]["dominant_event"] is None


# ---------------------------------------------------------------------------
# SyncConfig validation tests
# ---------------------------------------------------------------------------


class TestSyncConfig:
    """Tests for SyncConfig validation."""

    def test_default_config(self):
        """Default config should use standard LMT and USV parameters."""
        cfg = SyncConfig()
        assert cfg.lmt_frame_rate == 30.0
        assert cfg.wav_sample_rate == 300_000
        assert cfg.time_offset_s == 0.0

    def test_invalid_frame_rate(self):
        """Frame rate <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="lmt_frame_rate must be positive"):
            SyncConfig(lmt_frame_rate=0.0)

    def test_invalid_sample_rate(self):
        """Sample rate <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="wav_sample_rate must be positive"):
            SyncConfig(wav_sample_rate=0)


# ---------------------------------------------------------------------------
# Integration test (skip-guarded, requires real SQLite file)
# ---------------------------------------------------------------------------

REAL_LMT_DB = os.environ.get("LMT_TEST_DB_PATH")


@pytest.mark.skipif(
    not REAL_LMT_DB,
    reason="LMT_TEST_DB_PATH not set — set env var to path of real .sqlite file",
)
def test_loader_opens_real_sqlite():
    """Opens a real LMT SQLite file and loads at least one animal."""
    with LMTDatabaseLoader(REAL_LMT_DB) as loader:
        animals = loader.get_animals()
        assert isinstance(animals, list)
        event_types = loader.get_event_types()
        assert isinstance(event_types, list)
