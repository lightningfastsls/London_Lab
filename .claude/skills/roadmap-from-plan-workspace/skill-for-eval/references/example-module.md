# Example: Plan Step → ROADMAP Module

This shows the complete transformation from a loose plan description to a structured ROADMAP module entry.

## Input (from a web Claude plan)

> **Step 3: Feature Cache**
> Add caching to avoid recomputing features when re-analyzing the same WAV file with the same config. Use a simple SQLite cache keyed on (wav_path_hash, config_hash, segment_hash). This is important because feature extraction is the bottleneck — each segment takes ~50ms on CPU.

## Output (ROADMAP module entry)

```markdown
### 10.3 Feature Cache

**What:** SQLite-backed cache for computed audio features, avoiding redundant extraction when re-analyzing WAV files with the same config and segments.
**Status:** BLOCKED
**Review Tier:** 2 (standard module with clear logic)
**Depends on:** Phase 10.2 (Feature Extraction Engine)

/implement Feature Cache

Build a SQLite-backed feature cache that stores extracted audio features keyed on the combination of WAV file, extraction config, and segment boundaries. This avoids redundant computation when re-analyzing recordings — feature extraction takes ~50ms per segment on CPU, making caching essential for interactive workflows.

**Context:** Part of the feature extraction pipeline (Phase 10). Follows the frozen dataclass pattern from `docs/architecture/patterns.md`. The cache lives alongside the WAV data directory to keep data locality. References ADR-001 for sample rate consistency in cache key hashing.

**Files to create:**

1. `src/usv_spectrogram/features/cache.py` (NEW) — Feature cache with SQLite backend

    ```python
    @dataclass(frozen=True)
    class CacheKey:
        wav_hash: str          # SHA-256 of WAV file path + mtime
        config_hash: str       # SHA-256 of FeatureConfig serialization
        segment_hash: str      # SHA-256 of (start_time, end_time) tuple

    @dataclass(frozen=True)
    class CacheEntry:
        key: CacheKey
        features: np.ndarray   # Shape: (n_features,)
        created_at: float      # Unix timestamp
        extraction_ms: float   # How long extraction took (for profiling)
    ```

    Core logic:
    - `FeatureCache.__init__(db_path)` — opens/creates SQLite DB with schema
    - `get(key: CacheKey) -> Optional[np.ndarray]` — returns cached features or None
    - `put(key: CacheKey, features: np.ndarray, extraction_ms: float)` — stores features as binary blob (np.tobytes)
    - `invalidate(wav_hash: str)` — removes all entries for a WAV file (use when file changes)
    - `stats() -> CacheStats` — returns hit rate, total entries, total size on disk

    The cache uses WAL mode for concurrent read access. Binary blob storage avoids JSON serialization overhead for large feature vectors.

2. `tests/test_feature_cache.py` (NEW) — Cache tests

**Test plan:**
    ```
    1. Store and retrieve a feature vector — verify exact float equality via np.array_equal
    2. Cache miss returns None (not an error)
    3. Different configs for same WAV produce different cache entries
    4. invalidate() removes all entries for a WAV file but preserves others
    5. stats() reports correct hit/miss counts after a sequence of get/put operations
    6. Cache handles concurrent reads (two threads reading same key)
    7. Corrupt DB file raises clear error, not cryptic SQLite exception
    ```

**Exit criteria:**
- [ ] Cache hit returns array identical to what was stored (np.array_equal)
- [ ] Cache operations complete in < 5ms (excluding first-time DB creation)
- [ ] All 7 tests pass
- [ ] py_compile passes on cache.py
```

## What makes this example good

1. **Self-contained** — a fresh session can implement this without the original plan
2. **Data structures shown** — CacheKey and CacheEntry dataclasses with field types and comments
3. **Logic described** — each method's behavior is specified, not just listed
4. **Design decisions explained** — WAL mode, binary blobs, why cache key includes config hash
5. **Tests are specific** — each one says WHAT is verified, not just "test caching"
6. **Exit criteria are measurable** — performance targets, exact equality checks
7. **Context connects to the project** — references patterns.md, ADR-001, the pipeline it belongs to
