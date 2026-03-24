"""LMT SQLite database loader for behavioral event data.

Loads behavioral annotations from Live Mouse Tracker (LMT) SQLite databases.
LMT stores frame-by-frame tracking and behavioral event classifications for
mice in a home-cage setup.

Schema reference (LMT SQLite):
    ANIMAL: ID, RFID, NAME [, GENOTYPE, AGE, SEX, STRAIN, SETUP, IND]
    EVENT: ID, NAME, STARTFRAME, ENDFRAME, IDANIMALA-D, METADATA
    FRAME: FRAMENUMBER, TIMESTAMP (Unix ms)
    DETECTION: per-frame tracking coordinates
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BehavioralEvent:
    """A single behavioral event from the LMT database.

    Represents a classified behavior (e.g., 'Oral-oral Contact', 'Rearing')
    with both frame-based and time-based coordinates.
    """

    event_type: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    animal_id: Optional[int] = None
    partner_id: Optional[int] = None


@dataclass(frozen=True)
class AnimalInfo:
    """Metadata for a tracked animal in the LMT database.

    The LMT ANIMAL table has variable column count (3-9 columns across
    different database versions). Fields beyond animal_id/rfid/name may
    be None if the column doesn't exist in the database.
    """

    animal_id: int
    rfid: Optional[str] = None
    name: Optional[str] = None
    genotype: Optional[str] = None
    sex: Optional[str] = None
    strain: Optional[str] = None


class LMTDatabaseLoader:
    """Read-only loader for LMT SQLite behavioral databases.

    Opens the database in read-only mode and provides methods to query
    animals, event types, and behavioral events with optional filtering.

    Supports context manager protocol for safe connection lifecycle:

        with LMTDatabaseLoader(db_path) as loader:
            animals = loader.get_animals()
            events = loader.get_events()

    Parameters
    ----------
    db_path : Path or str
        Path to the LMT SQLite database file. For testing, pass ":memory:"
        to use an in-memory database.
    frame_rate : float
        LMT camera frame rate in fps. Default is 30.0 (standard LMT).
    """

    def __init__(self, db_path: Path | str, frame_rate: float = 30.0) -> None:
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")

        self._frame_rate = frame_rate
        self._db_path = db_path

        # For in-memory databases (testing), connect directly.
        # For file databases, use URI mode with read-only flag.
        if str(db_path) == ":memory:":
            self._conn = sqlite3.connect(":memory:")
        else:
            db_path = Path(db_path)
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            uri = f"file:{db_path.as_posix()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)

        self._conn.row_factory = sqlite3.Row

    def __enter__(self) -> LMTDatabaseLoader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    def get_animals(self) -> list[AnimalInfo]:
        """Load all animals from the ANIMAL table.

        Handles variable column count gracefully — columns beyond
        ID/RFID/NAME are optional in older LMT database versions.
        """
        cursor = self._conn.execute("SELECT * FROM ANIMAL")
        columns = [desc[0].upper() for desc in cursor.description]
        rows = cursor.fetchall()

        animals = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            animals.append(AnimalInfo(
                animal_id=row_dict.get("ID", 0),
                rfid=row_dict.get("RFID"),
                name=row_dict.get("NAME"),
                genotype=row_dict.get("GENOTYPE"),
                sex=row_dict.get("SEX"),
                strain=row_dict.get("STRAIN"),
            ))
        return animals

    def get_event_types(self) -> list[str]:
        """Return all distinct event type names, sorted alphabetically."""
        cursor = self._conn.execute(
            "SELECT DISTINCT NAME FROM EVENT ORDER BY NAME"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_events(
        self,
        event_types: Optional[list[str]] = None,
        animal_id: Optional[int] = None,
        time_range: Optional[tuple[float, float]] = None,
    ) -> list[BehavioralEvent]:
        """Query behavioral events with optional filters.

        Parameters
        ----------
        event_types : list[str], optional
            Filter to these event type names only.
        animal_id : int, optional
            Filter to events involving this animal (as A or B).
        time_range : tuple[float, float], optional
            Filter to events overlapping this time window (start_s, end_s).
            Converted to frame numbers for efficient SQL filtering.

        Returns
        -------
        list[BehavioralEvent]
            Matching events sorted by start time.
        """
        clauses: list[str] = []
        params: list = []

        if event_types is not None:
            placeholders = ",".join("?" for _ in event_types)
            clauses.append(f"NAME IN ({placeholders})")
            params.extend(event_types)

        if animal_id is not None:
            clauses.append("(IDANIMALA = ? OR IDANIMALB = ?)")
            params.extend([animal_id, animal_id])

        if time_range is not None:
            start_s, end_s = time_range
            # Convert seconds to frames for SQL-level filtering.
            # An event overlaps the range if:
            #   event.end_frame > range_start_frame AND event.start_frame < range_end_frame
            start_frame = int(start_s * self._frame_rate)
            end_frame = math.ceil(end_s * self._frame_rate)  # ceil: exclude events starting at exactly end_s
            clauses.append("ENDFRAME > ? AND STARTFRAME < ?")
            params.extend([start_frame, end_frame])

        sql = "SELECT NAME, STARTFRAME, ENDFRAME, IDANIMALA, IDANIMALB FROM EVENT"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY STARTFRAME"

        cursor = self._conn.execute(sql, params)
        events = []
        for row in cursor.fetchall():
            name, sf, ef, id_a, id_b = row
            events.append(BehavioralEvent(
                event_type=name,
                start_frame=sf,
                end_frame=ef,
                start_time_s=sf / self._frame_rate,
                end_time_s=ef / self._frame_rate,
                animal_id=id_a,
                partner_id=id_b,
            ))
        return events

    def get_timeline(self, animal_id: int) -> list[BehavioralEvent]:
        """Get all events involving a specific animal, sorted by start time.

        This is a convenience wrapper around get_events with an animal_id
        filter, useful for building per-animal behavioral timelines.
        """
        return self.get_events(animal_id=animal_id)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
