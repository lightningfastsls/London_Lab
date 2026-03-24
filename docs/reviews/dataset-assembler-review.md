# Dataset Assembler Module Review

**Module:** Training Data Assembly Pipeline (Phase 9.1)
**Review Tier:** 2
**Reviewer:** master-reviewer
**Date:** 2026-02-21
**Handoff:** `docs/reviews/dataset-assembler-handoff.md`

---

## Summary

The `DatasetAssembler` module correctly implements the core goal: a unified pipeline from LabelStorage JSONs to train/val/test CSVs. ADR-004 (recording-level splits), ADR-008 (3-source negatives), and ADR-010 (JSON format) are all implemented. DSP parameters match ADR-002 (n_fft=512, hop_length=128, Hann window, sr=300000). All 432 tests pass. The handoff is honest about limitations.

The review found **no blockers** and **four warnings**, the most important being an undocumented jitter failure condition that is more severe than the handoff indicates, a rounding overshoot in negative count allocation, and missing module documentation.

---

## Test Run

```
.venv/Scripts/python.exe -m pytest tests/test_dataset_assembler.py -v
8 passed in 5.44s

.venv/Scripts/python.exe -m pytest tests/ -v
432 passed, 52 warnings in 16.31s
```

No regressions introduced.

---

## Findings

### DSP CORRECTNESS

**No blockers found.** The scipy.signal.stft call in `_low_energy_negatives` correctly maps to ADR-002:

```python
# assembler.py line 579
scipy_signal.stft(samples, fs=sr, nperseg=self.N_FFT, noverlap=self.N_FFT - self.HOP_LENGTH, window="hann")
```

`noverlap = N_FFT - HOP_LENGTH = 512 - 128 = 384`, so step size = 128 = HOP_LENGTH. Verified correct.

The class constants `SAMPLE_RATE = 300_000`, `N_FFT = 512`, `HOP_LENGTH = 128` (assembler.py lines 135-137) match ADR-001 and ADR-002.

The low-energy threshold uses linear magnitude (not dB). This is correct because percentile ranking is monotone-preserving: the frames ranked below the 20th percentile by linear magnitude are identical to those ranked below the 20th percentile in dB. Verified with code.

---

### ML RIGOR

#### WARNING-1: Jitter failure threshold is 40ms, not ~80ms as stated in handoff

**What:** The handoff states "long USVs (~80ms+) will get no augmentation." The actual critical duration is exactly 40ms (= `jitter_window_ms / (2 * jitter_min_overlap)`). Any USV with duration >= 40ms generates zero jittered copies.

**Where:** `assembler.py` lines 322-331, `_jitter_candidate()`. The condition `if max_start <= min_start: return []` triggers when USV duration >= window_ms / (2 * min_overlap).

**Why it matters:** Mouse USVs commonly range up to 500ms. With the default 40ms window and 0.5 overlap, half the USV population may silently receive no augmentation. This reduces the effective jitter count below the expected `n_originals * jitter_n_samples`. The `TestJitteredCount` test only passes because all 8 synthetic detections are 15-30ms. A misleading understanding of "~80ms+" could lead a developer to not investigate when real-data jitter counts come back lower than expected.

**Fix:** Update the handoff and any documentation to say "USVs >= 40ms get no jitter (= `jitter_window_ms / (2 * jitter_min_overlap)`)." Consider making `jitter_window_ms` adaptive: if `USV_duration > jitter_window_ms`, set `jitter_window_ms = USV_duration * 1.5` for that candidate. Or document the formula clearly in `_jitter_candidate()`:

```python
# Critical duration above which no jitter is possible:
# max_start == min_start when USV_duration == window_ms / (2 * min_overlap)
# i.e., when window_ms < 2 * min_overlap * USV_duration
```

#### WARNING-2: Negative count can exceed `neg_ratio * n_positives` due to `max(1, ...)`

**What:** In `_create_negative_candidates`, each recording gets `n_rec = max(1, round(n_total * proportion))`. The `max(1, ...)` floor ensures low-detection recordings still get at least one negative. When many recordings have very low detection counts, the sum of `n_rec` across recordings can substantially exceed `n_total`.

**Where:** `assembler.py` lines 376-377.

**Why it matters:** `report.total_negatives` reflects how many candidates were actually generated, not `n_total`. The `neg_ratio` parameter becomes misleading — you can request `neg_ratio=1.0` and get `neg_ratio=1.4` in practice. For the 3-recording test case this causes no problem (equal-ish distributions), but with real data (e.g., 20 recordings where 15 have only 1-2 detections), the overshoot is material. The `TestReportAccuracy` test verifies the report matches the CSV counts (it does), but does not verify the report matches `neg_ratio * total_positives`.

**Verified:** `n_recs = {'rec_001': 5, 'rec_002': 1, 'rec_003': 1}` for a 5-negative request → sum = 7 (40% overshoot).

**Fix:** After the per-recording allocation loop, optionally clamp or log when `sum(n_recs.values()) > n_total`:

```python
actual_total = sum(n_recs[r] for r in recordings)
if actual_total > n_total * 1.2:
    logger.warning(
        "Negative count overshoot: requested %d, generating ~%d "
        "(due to min-1 floor across %d recordings)",
        n_total, actual_total, len(recordings)
    )
```

This is a WARNING not a BLOCKER because the excess negatives are still valid (no leakage), and `neg_ratio` is a soft target. But it should be documented.

#### CONFIRMED SAFE: Data leakage (ADR-004)

Jitter is performed before split (handoff Decision 5). Verified: `_create_positive_candidates` generates jittered copies → `_create_splits` splits by recording. All jittered variants of a USV from recording X stay with recording X in the same split. No leakage path exists.

#### CONFIRMED SAFE: Negative generation (ADR-008)

All 3 sources implemented. `TestNegativeSourceTypes` verifies all 3 prefixes appear in the output CSVs. The low-energy detection uses frame-level buffer masking before region grouping (handoff Decision 3), which is the correct architecture — regions are safe by construction.

---

### SPEC COMPLIANCE

#### WARNING-3: Missing module documentation

**What:** No `docs/modules/dataset-assembler.md` exists. The spec (Phase 9.1 handoff instructions, CLAUDE.md "After implementing a module") requires a module doc.

**Where:** `docs/modules/` contains only `cnn-classifier.md` and `energy-detector.md`.

**Why it matters:** Future developers (and the master-reviewer workflow) expect module docs to describe the public interface, key decisions, and ADR references.

**Fix:** Create `docs/modules/dataset-assembler.md` documenting: `AssemblyConfig`, `AssemblyReport`, `DatasetAssembler.assemble()`, key decisions (direct JSON parsing, Hamilton allocation, jitter-before-split, frame-level buffer masking), and ADR references (ADR-001, ADR-002, ADR-004, ADR-008, ADR-010).

#### CONFIRMED: IMPLEMENTATION_PROGRESS.md not updated

**What:** `IMPLEMENTATION_PROGRESS.md` line 51 refers to "Phase 9.1: Spectrogram Autoregressive Transformer (v2 Phase 1)" — the old Phase 8.2. The Training Data Assembly Pipeline work is not recorded. The handoff acknowledged this ("not yet updated — pending review").

**Fix:** Update after fixes are applied (covered by Fix Documentation Requirement below).

#### NOTE: Split ratio inconsistency (known, pre-existing)

The knowledge graph note `[[split ratio inconsistency between DECISIONS.md 80-10-10 and ROADMAP Phase 9 70-15-15 needs resolution]]` correctly identifies that ADR-004 says 80/10/10 while Phase 9 spec and the implementation use 70/15/15. The implementation chose 70/15/15 (matching the ROADMAP spec and `SplitConfig` defaults). This is an **existing unresolved inconsistency**, not introduced by this module. ADR-004 should be updated to reflect the deliberate 70/15/15 choice and when to switch back to 80/10/10 at scale. Not a blocker for this review.

---

### INTEGRATION CORRECTNESS

#### CONFIRMED: Pattern 1 (Frozen Config)

`AssemblyConfig` uses `@dataclass(frozen=True)` with `__post_init__` validation (assembler.py lines 48-110). Validates split ratios, negative fractions, and all positive fields. Converts Path fields from strings. Compliant.

#### CONFIRMED: Pattern 4 (CLI Script)

`scripts/assemble_training_data.py` has `parse_args()` and `main() -> int`, uses `sys.exit(main())`, includes `--dry-run` support, and has path bootstrap. Minor: the bootstrap at line 21 is missing the `if str(SRC_ROOT) not in sys.path:` guard required by Pattern 8. Not a blocker (Python caches imports), but not compliant.

```python
# Current (line 21):
sys.path.insert(0, str(REPO_ROOT / "src"))

# Should be (Pattern 8):
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

#### CONFIRMED: Direct JSON parsing avoids PyQt6 dependency

Handoff Decision 1 is sound. The pipeline correctly reads `metadata.wav_file` and `detections[]` directly from JSON without importing `LabelStorage`. This is a proper architectural boundary.

#### CONFIRMED: SpectrogramExtractor integration

The `ExtractionConfig(default_render_mode="training")` call correctly sets training mode (no axes, exact dimensions). The extractor uses `wav_dir / candidate.source_file.name` (assembler.py line 679, extractor line 56), which is compatible with how the assembler stores `source_file` (full filename with `.wav` extension per handoff Decision 4).

#### CONFIRMED: Candidate mutability relied on correctly

`Candidate` is not frozen (verified), so `candidate.spectrogram_path = result` in `_extract_spectrograms` line 682 works. The mutation is visible when `_candidates_to_samples` reads `c.spectrogram_path` in step 5, because `positive_candidates` is a list of the same objects mutated in step 4 (Python reference semantics). Correct.

#### WARNING-4: Sample rate mismatch handling inconsistent between assembler and extractor

**What:** In `_low_energy_negatives`, if `load_wav_mono` returns `sr != 300000`, the code logs a warning and continues the STFT with the wrong `fs` value in `scipy_signal.stft(samples, fs=sr, ...)` (assembler.py lines 573-575). The frame-time mapping (`times * 1000.0`) is then computed with the wrong sample rate, creating mis-timed frame labels. Subsequent extraction of those candidates will raise `ValueError` in `SpectrogramExtractor` (extractor.py lines 72-76).

**Where:** `assembler.py` lines 572-576, `spectrogram_extractor.py` lines 72-76.

**Why it matters:** The assembler creates candidates with timing derived from a wrong-sr STFT, then the extractor raises when trying to extract them. The assembler silently generates bad candidates rather than skipping them. The quality check will report missing spectrogram files but the root cause (wrong sample rate) is obscured.

**Fix:** In `_low_energy_negatives`, return early if `sr != self.SAMPLE_RATE`:

```python
samples, sr = load_wav_mono(wav_path)
if sr != self.SAMPLE_RATE:
    logger.warning(
        "Skipping low-energy negatives for %s: sample rate %d != expected %d",
        source_stem, sr, self.SAMPLE_RATE
    )
    return []
```

---

### CODE QUALITY

#### Lazy imports inside loops (style issue)

`import soundfile as sf` appears inside the iteration body of `_create_positive_candidates` (line 290) and `_create_negative_candidates` (line 401). Python caches modules so this is not a performance bug, but it is an anti-pattern — imports belong at the top of the module. `soundfile` is already a transitive dependency.

**Fix:** Add `import soundfile as sf` to the top-level imports in `assembler.py`.

#### soundfile.info() N+1 in `_create_positive_candidates`

`_create_positive_candidates` calls `sf.info(wav_path)` once per detection row (O(n_detections)), not once per recording. `_create_negative_candidates` correctly groups by recording first and calls `sf.info` once per recording. With 840 detections across 6 recordings, this is ~140x more calls than needed.

This is acknowledged in the handoff ("kept simple for now since it's just metadata — fast I/O"). With the current dataset size (~840 labels) the impact is trivial (~840ms). At 30K labels it becomes ~30 seconds wasted in a pipeline step. Worth fixing before 30K scale, but not a blocker now.

**Fix when dataset exceeds ~5K labels:** Build a `recording_durations` dict keyed by `source_file` before the loop.

#### `_gap_negatives` gap midpoint sampling allows duplicates

The gap midpoint sampling logic (assembler.py lines 513-527) samples indices with replacement but uses `seen_positions` to deduplicate only when `len(gap_midpoints) >= n`. When there are few gaps, you get repeated midpoints that produce identical or near-identical candidates (same `start`, same `end`, different `cid`). These won't cause correctness failures — they're still valid negatives at valid positions — but they reduce diversity.

---

### TEST QUALITY

#### Tests are honest and non-greenwashing

`TestJitteredCount` correctly asserts the expected count and would fail if jitter logic changed. `TestNoRecordingLeakage` checks all 6 pairwise split combinations. `TestReportAccuracy` cross-validates report counts against actual CSV row counts. `TestSpectrogramPathsExist` reads every CSV row and checks file existence on disk. These are substantive behavioral tests, not "assert no exception" smoke tests.

#### Missing test: jitter failure for long USVs

The test fixture uses only 15-30ms USVs. There is no test that verifies what happens when a USV has duration >= 40ms (jitter-window = jitter-min-overlap formula). This leaves a behavioral gap — the silent `return []` path is never exercised.

**Fix:** Add a test case with a 50ms detection and verify that `report.total_positives` equals `n_originals * 1` (no jitter), not `n_originals * (1 + jitter_n_samples)`.

#### Missing test: all-labels-deleted JSON file

`_collect_labels` skips `user_action == "deleted_by_user"` but there is no test for a JSON file where every detection has been deleted. This path leads to the `if not rows: raise ValueError(...)` branch. Test 7 covers the empty-dir case but not the all-deleted case.

**Fix (optional):** Add a test that creates a JSON where all detections are `"user_action": "deleted_by_user"` and verifies `ValueError` is raised with message "No valid detections found."

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| `docs/modules/dataset-assembler.md` | MISSING | Must be created per CLAUDE.md and completion-sequence protocol |
| `docs/architecture/patterns.md` | UP TO DATE | No new patterns introduced |
| `DECISIONS.md` | UP TO DATE | No new ADRs needed (uses ADR-001, 002, 004, 008, 010) |
| `IMPLEMENTATION_PROGRESS.md` | NOT UPDATED | Phase 9.1 not recorded; must be updated |
| `docs/reviews/dataset-assembler-handoff.md` | EXISTS | Handoff present and accurate, but understates jitter failure threshold |

---

## Knowledge Graph Notes Checked

- `[[split ratio inconsistency between DECISIONS.md 80-10-10 and ROADMAP Phase 9 70-15-15 needs resolution]]` — The implementation correctly chose 70/15/15 per ROADMAP spec. This open question remains unresolved at the ADR level; ADR-004 should be updated.
- `[[constrained jittering generates diverse positive training examples by shifting detection boundaries within overlap constraints]]` — Implementation matches the described approach (N=5, 40ms window, 50% overlap). The note correctly describes the mechanism; this review adds the finding that USVs >= 40ms get no jitter.
- `[[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]]` — All 3 sources implemented and verified by test.
- `[[recording-level splits prevent data leakage in USV classification]]` — Correctly implemented including jitter-before-split.

---

## Fix Documentation Requirement

After applying all fixes listed above, the implementor MUST:
1. Add a "## Fixes Applied" section to this review file (`docs/reviews/dataset-assembler-review.md`)
2. For each fix: state what was changed, which file:line, and why
3. Re-run the affected tests and record pass/fail counts:
   - `pytest tests/test_dataset_assembler.py -v` (must remain 8/8)
   - `pytest tests/ -v` (must remain 432+/0 failed)
4. Update `IMPLEMENTATION_PROGRESS.md` with a dated entry for Phase 9.1
5. Create `docs/modules/dataset-assembler.md` with public interface documentation
6. Self-verify against each WARNING above by reading the changed code

Priority order for fixes:
1. WARNING-4 (sample rate early return) — prevents silent DSP corruption on wrong-sr files
2. WARNING-1 (document jitter threshold accurately in code and handoff) — prevents misdiagnosis in production
3. WARNING-3 (create module doc) — required by project workflow
4. WARNING-2 (log negative count overshoot) — observability improvement
5. Style fixes (lazy imports, sys.path guard) — low priority, fix opportunistically

---

## Verdict

**CHANGES NEEDED**

No blockers. Four warnings must be addressed before Phase 10.

Blocking on proceeding to Phase 10:
- WARNING-3: Create `docs/modules/dataset-assembler.md` (required by workflow)
- Update `IMPLEMENTATION_PROGRESS.md` with Phase 9.1 entry

Can be fixed in the same PR (no re-review required for WARNING-1, 2, 4):
- WARNING-4: Add early return in `_low_energy_negatives` for wrong sample rate
- WARNING-1: Add formula comment in `_jitter_candidate` and update handoff note
- WARNING-2: Add overshoot warning log
- Style: Move `import soundfile as sf` to top-level imports
- Style: Add `if str(SRC_ROOT) not in sys.path:` guard in CLI script

---

## Fixes Applied

### WARNING-4: Sample rate early return
**File:** `assembler.py:572-577`
**Change:** Replaced `logger.warning()` + continue with `logger.warning()` + `return []`. Recordings with wrong sample rate now skip low-energy negative generation entirely, preventing mis-timed candidates and downstream extraction failures.

### WARNING-1: Jitter threshold documented accurately
**File:** `assembler.py:322-330` (code comment), `dataset-assembler-handoff.md` (handoff update)
**Change:** Added formula comment: "Critical duration threshold: jitter is impossible when `usv_dur >= window_ms / (2 * min_overlap)`. With defaults (40ms window, 0.5 overlap), USVs >= 40ms get no jitter." Updated handoff from "~80ms+" to the correct ">= 40ms" threshold. Added test `TestJitterFailureForLongUSVs` verifying the silent-return path.

### WARNING-3: Module documentation created
**File:** `docs/modules/dataset-assembler.md` (NEW)
**Change:** Created module doc covering: public interface (`AssemblyConfig`, `AssemblyReport`, `DatasetAssembler.assemble()`), 7 key decisions, output structure, DSP parameters, and ADR references.

### WARNING-2: Negative count overshoot logging
**File:** `assembler.py:423-428`
**Change:** Added warning log when actual negative count exceeds `n_total * 1.2`: "Negative count overshoot: requested N, generating M (due to min-1 floor across K recordings)".

### Style: Lazy imports moved to top level
**File:** `assembler.py:24`
**Change:** Added `import soundfile as sf` to module-level imports. Removed two `import soundfile as sf` lines inside `_create_positive_candidates` (line 290) and `_create_negative_candidates` (line 401).

### Style: sys.path guard in CLI script
**File:** `scripts/assemble_training_data.py:20-22`
**Change:** Added `if str(SRC_ROOT) not in sys.path:` guard per Pattern 8.

### Additional: Two new tests
**File:** `tests/test_dataset_assembler.py`
**Change:** Added `TestJitterFailureForLongUSVs` (50ms USV -> 1 positive, no jitter) and `TestAllLabelsDeleted` (all `deleted_by_user` -> ValueError "No valid detections found"). Test count: 8 -> 10.

### Documentation updates
- `IMPLEMENTATION_PROGRESS.md`: Added Phase 9.1 entry with full details
- `docs/reviews/dataset-assembler-handoff.md`: Corrected jitter threshold from "~80ms+" to ">= 40ms"

### Test results after fixes
```
pytest tests/test_dataset_assembler.py -v
10 passed in 6.59s

pytest tests/ --tb=short
434 passed, 52 warnings in 19.89s
```
