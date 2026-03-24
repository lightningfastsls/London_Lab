# Testing Audit Plan

## Current State

| Category | Status | Test Count |
|----------|--------|------------|
| Well-tested (energy detector, splits, CNN, zarr, etc.) | Good | ~350 |
| Thin tests (1-2 functions, import-only) | Needs expansion | ~20 |
| Completely untested persistence/I/O | **Critical gap** | 0 |
| Completely untested business logic | Gap | 0 |
| **Total existing** | | **~461** |

## Target: ~247 New Tests Across 14 Work Items

### Conventions for All New Test Files

Every new test file follows the established project pattern:
- `sys.path.insert(0, str(SRC_ROOT))` at top
- `tempfile.NamedTemporaryFile(delete=False)` + `yield` + `unlink()` for WAV files (Windows)
- `tempfile.TemporaryDirectory()` for output directories
- Class-per-behavior (`class TestXxx:`) organization
- Plain `assert` with f-string failure messages
- `np.testing.assert_allclose(rtol=1e-6)` for float arrays
- Explicit config params (never rely on defaults for critical values)

---

## TIER 1 — Persistence Round-Trip Tests (~117 tests)

These prevent data loss. The labeling annotation loss incident is the exact class of bug these catch.

---

### T1-A: `labeling_app.py` persistence (~28 tests)

**Source:** `src/usv_spectrogram/labeling/labeling_app.py`
**New test file:** `tests/test_labeling_app.py`
**Setup:** No special setup. Pure I/O functions, no Streamlit/model/WAV needed for most tests.

#### `save_label` + `load_existing_labels` round-trip (9 tests)

| Test | What it catches |
|------|----------------|
| `test_single_label_survives_reload` | The exact scenario that failed in production |
| `test_multiple_labels_survive_reload` | Full-rewrite strategy doesn't drop entries |
| `test_update_existing_label` | Re-labeling same candidate overwrites cleanly |
| `test_expand_ms_none_preserves_existing_value` | **Key fragility**: None falls back to old value, not 0.0 |
| `test_expand_ms_none_with_no_existing_value` | None defaults to 0.0 for new entries |
| `test_all_label_options_accepted` | "USV", "Not USV", "Uncertain" all roundtrip |
| `test_csv_fieldnames_are_canonical` | Header is exactly `candidate_id,label,labeled_at,expand_ms` |
| `test_labeled_at_is_iso_format` | Timestamp parses as ISO datetime |
| `test_candidate_id_with_special_characters` | IDs with underscores/digits survive |

#### `load_existing_labels` edge cases (5 tests)

| Test | What it catches |
|------|----------------|
| `test_nonexistent_file_returns_empty_dict` | Graceful first-run behavior |
| `test_expand_ms_empty_string_defaults_to_zero` | CSV with empty expand_ms field |
| `test_expand_ms_invalid_string_defaults_to_zero` | Corrupt expand_ms silently defaults |
| `test_expand_ms_none_string_defaults_to_zero` | Literal "None" string in CSV |
| `test_float_expand_ms_preserved` | 15.5 roundtrips as 15.5 |

#### `archive_labels_and_spectrograms` (5 tests)

| Test | What it catches |
|------|----------------|
| `test_archive_creates_timestamped_directory` | Archive dir naming convention |
| `test_archive_copies_labels_csv` | Labels preserved in archive |
| `test_archive_moves_existing_pngs` | PNGs moved (not copied) to archive |
| `test_archive_counts_missing_pngs` | Missing PNG counted, not crashed |
| `test_empty_labels_returns_early` | No crash on empty labels dict |

#### Pure helper functions (9 tests)

| Test | Function |
|------|----------|
| `test_all_returns_all` | `filter_candidates` |
| `test_unlabeled_excludes_labeled` | `filter_candidates` |
| `test_only_usv_filter` | `filter_candidates` |
| `test_finds_first_unlabeled_from_start` | `find_next_unlabeled` |
| `test_skips_labeled_entries` | `find_next_unlabeled` |
| `test_returns_none_when_all_labeled` | `find_next_unlabeled` |
| `test_absolute_path_unchanged` | `resolve_input_path` |
| `test_relative_path_joined_to_repo_root` | `resolve_input_path` |
| `test_count_labeled_correct` | `count_labeled` |

**Known fragile points to document in test comments:**
- `expand_ms=None` silently carries old value (design choice, not bug)
- `load_existing_labels` uses `csv.DictReader` not pandas — expand_ms parse errors default to 0.0
- `archive_labels_and_spectrograms` uses `shutil.move` (destructive) with 1-second timestamp granularity
- `filter_candidates("Unlabeled")` resets index, breaking alignment with original DataFrame

---

### T1-B: `noise_review_app.py` persistence (~15 tests)

**Source:** `src/usv_spectrogram/labeling/noise_review_app.py`
**New test file:** `tests/test_noise_review_app.py`
**Setup:** No special setup needed.

#### `save_review` + `load_existing_reviews` round-trip (6 tests)

| Test | What it catches |
|------|----------------|
| `test_clean_status_survives_reload` | Basic roundtrip |
| `test_trimmed_status_survives_reload` | **Key**: "Trimmed" not in LABEL_OPTIONS but used at runtime |
| `test_multiple_reviews_survive_reload` | Full-rewrite doesn't drop entries |
| `test_update_existing_review` | Re-review overwrites cleanly |
| `test_csv_fieldnames_are_canonical` | Header: `candidate_id,status,trim_ms,reviewed_at` |
| `test_reviewed_at_is_iso_format` | Timestamp parsing |

#### `load_existing_reviews` edge cases (3 tests)

| Test | What it catches |
|------|----------------|
| `test_nonexistent_file_returns_empty_dict` | Graceful first-run |
| `test_negative_trim_ms_preserved` | -20.0 roundtrips correctly |
| `test_zero_trim_ms_preserved` | 0 loads as 0.0 |

**Note:** Unlike `labeling_app.py`, this uses `float()` with NO try/except — a corrupt `trim_ms` would crash at runtime.

#### `trim_and_reextract` + helpers (6 tests)

| Test | What it catches |
|------|----------------|
| `test_zero_trim_returns_none` | No-op case |
| `test_positive_trim_shrinks_from_start` | Arithmetic correctness (mock extractor) |
| `test_negative_trim_shrinks_from_end` | Negative trim sign convention |
| `test_too_short_result_returns_none` | Duration < 10ms threshold |
| `test_counts_only_reviewed_candidates` | `count_reviewed` |
| `test_returns_none_when_all_reviewed` | `find_next_unreviewed` |

---

### T1-C: `app/core/label_storage.py` round-trip (~20 tests)

**Source:** `src/usv_spectrogram/app/core/label_storage.py`
**New test file:** `tests/test_label_storage.py`
**Setup:** Must construct `AudioData`, `DetectionResult`, `DetectedUSV` dataclasses with synthetic numpy arrays. No model files needed.

#### `LabelStorage.save` + `LabelStorage.load` round-trip (9 tests)

| Test | What it catches |
|------|----------------|
| `test_metadata_fields_survive_reload` | wav_file, model_file, duration_s, sample_rate, n_detections, file_label |
| `test_detection_params_survive_reload` | high_threshold, low_threshold |
| `test_probability_curve_arrays_survive_reload` | times, probabilities, column_indices (use assert_allclose) |
| `test_detection_count_matches` | 3 detections saved → n_detections=3 in metadata |
| `test_usv_times_survive_reload` | start_time_s, end_time_s per detection |
| `test_user_adjusted_flag_survives` | user_adjusted=True + original times |
| `test_user_action_survives` | user_action="added_manually" |
| `test_empty_detections_survives` | Zero detections is valid |
| `test_file_label_noise_survives` | file_label="noise" |

#### `reconstruct_detected_usv` known behaviors (4 tests)

| Test | What it documents |
|------|-------------------|
| `test_save_state_always_unsaved_after_load` | save_state NOT in JSON — always reconstructs as "unsaved" |
| `test_original_times_default_to_zero_when_absent` | Ambiguous 0.0 default |
| `test_optional_user_action_is_none_when_absent` | Missing → None |
| `test_original_cnn_probability_absent_gives_none` | Missing → None |

#### Error conditions + image export (7 tests)

| Test | What it catches |
|------|----------------|
| `test_full_roundtrip_via_reconstruct` | reconstruct_detection_result end-to-end |
| `test_file_label_backward_compatible_absent` | Old JSON without file_label loads as None |
| `test_missing_file_raises_file_not_found` | FileNotFoundError |
| `test_malformed_json_raises_json_decode_error` | json.JSONDecodeError |
| `test_creates_png_file` | export_annotated_image creates file |
| `test_no_figure_leak_on_success` | plt.get_fignums() unchanged |
| `test_creates_parent_dirs` | Nested output path auto-created |

---

### T1-D: `app/core/detection_exporter.py` (~18 tests)

**Source:** `src/usv_spectrogram/app/core/detection_exporter.py`
**New test file:** `tests/test_detection_exporter.py`
**Setup:** Synthetic `AudioData` + `DetectedUSV`. No model files.

#### `export_detection` outputs (8 tests)

Tests that PNG, JSON, CSV files are created with correct schema and content.

#### `_append_to_csv` fragility (3 tests)

| Test | What it catches |
|------|----------------|
| `test_first_write_adds_header` | Header written on first call |
| `test_second_write_no_duplicate_header` | Only 1 header row after 2 calls |
| `test_csv_row_count_matches_calls` | 5 calls → 1 header + 5 data rows |

#### Context windowing + output structure (7 tests)

Tests for context clamping at boundaries and correct directory/filename structure.

---

### T1-E: `app/core/saved_detection_tracker.py` (~20 tests)

**Source:** `src/usv_spectrogram/app/core/saved_detection_tracker.py`
**New test file:** `tests/test_saved_detection_tracker.py`
**Setup:** Pure JSON + dataclasses. No special setup.

#### Persistence round-trip (4 tests)
#### `is_saved` logic (7 tests) — including strict inequality boundary test
#### `_time_ranges_overlap` (4 tests)
#### Error handling (2 tests) — corrupt JSON, old format
#### `get_unsaved_detections` (1 test)
#### Known limitation documentation (1 test) — save_state not persisted
#### `test_save_state_not_persisted_in_tracking_json` — Documents why detections appear unsaved after restart

---

### T1-F: `app/core/preset_config.py` (~16 tests)

**Source:** `src/usv_spectrogram/app/core/preset_config.py`
**New test file:** `tests/test_preset_config.py`
**Setup:** Use tmp dir for config_dir to avoid writing into source tree.

#### `ThresholdPreset.validate` (6 tests) — boundary conditions
#### `PresetManager` persistence round-trip (2 tests)
#### `PresetManager` fallback behavior (4 tests) — invalid preset wipes ALL to defaults
#### `get_preset` (2 tests)
#### Config dir isolation (2 tests) — document that default writes into source tree

---

## TIER 2 — Core Logic Without Coverage (~92 tests)

---

### T2-A: `io_wav.py` (~20 tests)

**Source:** `src/usv_spectrogram/io_wav.py`
**New test file:** `tests/test_io_wav.py`
**Setup:** Synthetic WAVs via `soundfile.write`.

- `load_wav_mono`: mono/stereo handling, float32 dtype, sample rate, file-not-found
- `load_wav_segment_mono`: negative start (ValueError), zero/negative duration (empty), beyond-end (empty), truncation, stereo segment
- `get_default_wav_dir`: env var set vs. fallback (use `monkeypatch.setenv`)

---

### T2-B: `spectrogram.py` (~15 tests)

**Source:** `src/usv_spectrogram/spectrogram.py`
**New test file:** `tests/test_spectrogram_compute.py`

- Output shape: freq axis filtered to [f_min, f_max], time axis from hop
- Short/empty signal → empty spectrogram (not error)
- Sample rate enforcement
- Tone at known frequency appears in correct bin
- Zero padding doubles freq resolution
- Hop validation

---

### T2-C: `param_lab/state.py` — pure functions only (~14 tests)

**Source:** `src/usv_spectrogram/param_lab/state.py`
**New test file:** `tests/test_param_lab_state.py`

**Only 3 testable functions** (others are `@st.cache_data` decorated):
- `config_from_controls`: pure dataclass replacement
- `stft_cache_key`: tuple of 9 fields, **gain_db excluded** (critical cache correctness)
- `parse_sweep_values`: comma-separated parsing with type dispatch (int/float/str)

---

### T2-D: `app/core/detection_logic.py` — HysteresisDetector (~28 tests)

**Source:** `src/usv_spectrogram/app/core/detection_logic.py`
**New test file:** `tests/test_detection_logic.py`
**Setup:** Pure numpy — synthetic probability arrays.

All 5 stages of the pipeline tested independently:
1. `_hysteresis_detect` — high entry, low exit state machine (4 tests)
2. `_filter_by_duration` — min/max ms cutoffs (4 tests)
3. `_merge_nearby` — column gap merging (6 tests)
4. `_filter_by_probability_stability` — sustained prob check (3 tests)
5. `_filter_by_temporal_position` — start/end exclusion zones (4 tests)
6. Full `detect()` pipeline integration (3 tests)
7. `mark_as_noise` / `clear_noise_label` (4 tests)

---

### T2-E: `app/core/sliding_inference.py` — pure helpers only (~15 tests)

**Source:** `src/usv_spectrogram/app/core/sliding_inference.py`
**New test file:** `tests/test_sliding_inference.py`
**Setup:** Mock-based — `SlidingInference.__init__` loads a model file. Test only the pure helper methods via a lightweight stub.

- `_apply_mad_normalization`: output in [0,1], uniform input → zeros, shape preserved (5 tests)
- `_should_skip_window_by_energy`: threshold comparison (3 tests)
- `_normalize_window_to_training_distribution`: max→1.0, low-energy guard (3 tests)
- `_prepare_batch`: shape (batch,1,256,512), padding, dtype (4 tests)

---

## TIER 3 — Nice-to-Have Expansions (~38 tests)

---

### T3-A: Expand existing thin tests (~12 tests)

**Modify:** `test_param_lab_heuristic.py` (+4), `test_param_lab_segment.py` (+2), `test_streaming_equivalence.py` (+2), `test_param_lab_imports.py` (+4 behavior tests)

### T3-B: `clustering/` subsystem (~18 tests)

**New test file:** `tests/test_clustering.py`
- `USVClusterer`: fit_predict, metrics, invalid method (6 tests)
- `ClusterAnalyzer`: exemplars, diversity, noise exclusion (6 tests)
- `EmbeddingVisualizer`: tsne output shape, invalid method, plot creation (5 tests)
- Uses `@pytest.importorskip("sklearn")` for graceful skip

### T3-C: `param_lab/sweep.py` (~8 tests)

**New test file:** `tests/test_param_lab_sweep.py`
- `run_sweep`: output files, PNG count, return dict structure
- `_sanitize_label`: special chars, empty string

---

## Execution Order

```
Recommended order (simplest → most complex, respecting dependencies):

Session 1: T1-F (preset_config, 16 tests) + T1-A (labeling_app, 28 tests)
Session 2: T1-B (noise_review_app, 15 tests) + T2-A (io_wav, 20 tests)
Session 3: T2-B (spectrogram, 15 tests) + T2-C (param_lab state, 14 tests)
Session 4: T2-D (detection_logic, 28 tests) — largest single item
Session 5: T1-C (label_storage, 20 tests) + T1-D (detection_exporter, 18 tests)
Session 6: T1-E (saved_detection_tracker, 20 tests) + T2-E (sliding_inference, 15 tests)
Session 7: T3-A (expand existing, 12 tests) + T3-B (clustering, 18 tests) + T3-C (sweep, 8 tests)
```

**Dependencies:**
- T1-C, T1-D, T1-E all need `DetectedUSV`/`DetectionResult` from `detection_logic.py` — do T2-D first
- T3-C depends on T2-A and T2-B understanding
- All other items are independent

---

## Definitions of Done

**Tier 1 Done:** All 6 test files pass, ~117 new tests, no test expectations modified to pass

**Tier 2 Done:** All 5 test files pass, ~92 new tests, full suite doesn't regress

**Tier 3 Done:** 3 new + 3 expanded test files pass, ~38 new tests

**Overall:** ~247 new tests → project total ~708

---

## Excluded From This Plan

| Item | Why excluded |
|------|-------------|
| `clustering/feature_extractor.py` | Requires real model checkpoint |
| `sliding_inference.py` full `infer()` | Requires real model checkpoint |
| PyQt6 widgets (`app/widgets/`) | Requires pytest-qt + display |
| Streamlit render functions | Requires running Streamlit server |
| Scripts in `scripts/` (CLI wrappers) | Low risk — underlying modules are tested |
| `usv_language/` tests | Separate test suite already exists |
| `notion_notes/` tests | Already comprehensive |

---

## Verification After Each Work Item

```powershell
.\.venv\Scripts\python.exe -m py_compile <new_test_file>
.\.venv\Scripts\python.exe -m pytest <new_test_file> -v --tb=short
.\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short  # full regression
```
