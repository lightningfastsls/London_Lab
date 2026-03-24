# LMT Data Access Layer

**Phase:** LMT Data Access Layer
**ADRs:** ADR-001 (sample rate 300 kHz), ADR-002 (STFT hop_length=128)
**Tests:** `tests/test_lmt.py` -- 21 tests across 3 test classes

## Purpose

Provides a Python API for loading behavioral event annotations from Live Mouse Tracker (LMT) SQLite databases and aligning their timestamps with WAV/spectrogram coordinate systems. This is the bridge between "what the mice were doing" (LMT behavioral events at 30 fps) and "what they were vocalizing" (USV detections at 300 kHz).

LMT records frame-by-frame behavioral annotations (e.g., "Oral-oral Contact", "Rearing", "Social approach") in SQLite databases. This module:
1. Loads animal metadata and behavioral events from the database
2. Converts between LMT frame coordinates, wall-clock seconds, WAV samples, and spectrogram frames
3. Aligns behavioral events with USV detections by temporal overlap

## Public Interface

### `BehavioralEvent`

```python
@dataclass(frozen=True)
class BehavioralEvent:
    event_type: str              # e.g., "Oral-oral Contact"
    start_frame: int             # LMT frame number
    end_frame: int               # LMT frame number
    start_time_s: float          # Converted to seconds
    end_time_s: float            # Converted to seconds
    animal_id: Optional[int]     # Primary animal
    partner_id: Optional[int]    # Partner (pairwise events)
```

### `AnimalInfo`

```python
@dataclass(frozen=True)
class AnimalInfo:
    animal_id: int
    rfid: Optional[str]
    name: Optional[str]
    genotype: Optional[str]      # May be None in older DBs
    sex: Optional[str]           # May be None in older DBs
    strain: Optional[str]        # May be None in older DBs
```

### `LMTDatabaseLoader`

```python
class LMTDatabaseLoader:
    def __init__(db_path, frame_rate=30.0)   # read-only connection
    def get_animals() -> list[AnimalInfo]
    def get_event_types() -> list[str]
    def get_events(event_types?, animal_id?, time_range?) -> list[BehavioralEvent]
    def get_timeline(animal_id) -> list[BehavioralEvent]
    def close()
    # Also supports: with LMTDatabaseLoader(...) as loader:
```

### `SyncConfig`

```python
@dataclass(frozen=True)
class SyncConfig:
    lmt_frame_rate: float = 30.0      # LMT camera fps
    wav_sample_rate: int = 300_000    # Per ADR-001
    time_offset_s: float = 0.0       # LMT-to-WAV sync offset
```

### `LMTSynchronizer`

```python
class LMTSynchronizer:
    def __init__(config=SyncConfig())
    def lmt_frame_to_seconds(frame) -> float
    def seconds_to_wav_sample(time_s) -> int
    def seconds_to_spectrogram_frame(time_s, hop_length=128) -> int
    def align_events_with_detections(events, detections) -> list[dict]
```

## Key Decisions

- **Read-only database access**: Uses `?mode=ro` URI for file databases to protect experimental data.
- **Variable ANIMAL schema**: The LMT ANIMAL table has 3-9 columns across versions. The loader handles missing columns gracefully (returns None).
- **Frame rate as parameter**: Default 30.0 fps but configurable for non-standard LMT setups.
- **Time range filtering in SQL**: Converts seconds to frames before querying, letting SQLite do the filtering efficiently.
- **Detection dicts (not DetectedUSV)**: The alignment API accepts any dict with `start_time`/`end_time` keys, keeping it decoupled from our detection classes.
- **Floor for spectrogram frames**: `int(time_s * sr / hop)` uses truncation, matching the convention that a spectrogram frame represents the window starting at that sample.
- **Event specificity ranking**: Pairwise > behavioral action > general > environmental. Used to select the "dominant" behavioral context for each USV.

## Integration Points

- **Reads from:** LMT SQLite databases (`.sqlite` files)
- **Feeds into:** USV-behavior correlation analysis, behavioral context enrichment
- **Dependencies:** `sqlite3` (standard library only)

## Coordinate Systems

```
LMT frame (30 fps) ──→ seconds ──→ WAV sample (300 kHz)
                           └──→ spectrogram frame (hop-dependent)
```

| From | To | Formula |
|------|----|---------|
| LMT frame → seconds | `frame / 30.0 + offset` |
| seconds → WAV sample | `round(time_s * 300_000)` |
| seconds → spec frame | `int(time_s * 300_000 / hop_length)` |
