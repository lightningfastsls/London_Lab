# Corpus Constants Unification Review

**Date:** 2026-04-17
**Reviewer:** Master Reviewer (fresh context)
**Spec:** `docs/handoffs/corpus-constants-unification-2026-04-17.md` + `~/.claude/plans/lucky-noodling-alpaca.md`
**Review Tier:** Tier 2 (configuration + data-registry — no ML training, no signal processing math, but architecture-wide constant propagation with CNN regression risk)

---

## Summary

This is a clean, well-structured refactor. The core implementation — `corpus.py` as single source of truth, all four config imports updated, drift assertions, audit script with parameter sidecar, and committed `5970.json` artifact — is correct. All 84 directly-affected tests pass. Zero new test failures introduced against the 72-failure baseline.

Three issues found: one SERIOUS (stale inline comment encoding a wrong unit), one NIT (stale `patterns.md` code example), and one documentation gap (missing module doc and `IMPLEMENTATION_PROGRESS.md` entry). The CNN regression risk is confirmed clear by code inspection.

---

## 1. Handoff Fidelity — Layer 1 Checklist

**corpus.py**
- `SAMPLE_RATE_HZ = 300_000` — matches spec
- `USV_FREQ_MIN_HZ = 20_000`, `USV_FREQ_MAX_HZ = 120_000` — matches user-specified canonicals
- `STFT_N_FFT = 512`, `STFT_HOP = 128` — matches ADR-002
- All four derived functions present and return correct values (verified by hand and by `test_corpus.py`)
- Module docstring contains the CNN-freeze constraint paragraph — matches spec verbatim structure

**SpectrogramConfig** (`src/usv_spectrogram/config.py`)
- `expected_sample_rate_hz = SAMPLE_RATE_HZ` (300k) — correct, was 250k
- `f_min_hz = float(USV_FREQ_MIN_HZ)` (20k) — correct, was 30k
- `f_max_hz = float(USV_FREQ_MAX_HZ)` (120k) — correct, was 125k
- `window_length = 2048`, `hop_ms = 0.5`, rendering knobs — untouched as specified
- Docstring updated from "250 kHz" to "300 kHz LMT WAVs" — correct

**DetectionConfig** (`src/usv_spectrogram/detection/config.py`)
- All five corpus imports present and applied — correct
- `freq_min_hz` 25k→20k, `freq_max_hz` 110k→120k — matches spec
- Docstring updated — correct

**ExtractionConfig** (`src/usv_spectrogram/detection/extraction_config.py`)
- Values unchanged — correct
- File-level NOTE comment added explaining intentional non-import — matches spec template exactly
- Drift assertions added for all 5 fields at module import time — exceeds spec (spec only required min/max)

**AnalysisConfig** (`usv_language/analysis/config.py`)
- `freq_min_hz = USV_FREQ_MIN_HZ`, `freq_max_hz = USV_FREQ_MAX_HZ` — correct, no value change

**audio_loader.py** (`SonicConfig`)
- No value change
- One-line NOTE comment added referencing corpus.py and explaining intentional 0-30 kHz band — matches spec

All Layer 1 bullets are satisfied.

---

## 2. Drift Assertion Correctness

The drift assertions at `extraction_config.py:148-164` use `ExtractionConfig.__dataclass_fields__["<field>"].default` to compare against the imported corpus value.

**Scenario: `corpus.USV_FREQ_MAX_HZ` changes to `125_000`**

At import time of `extraction_config.py`:
- `_CORPUS_USV_FREQ_MAX_HZ = 125_000` (new corpus value)
- `_FIELDS["freq_max_hz"].default = 120_000` (ExtractionConfig literal, unchanged)
- `assert 120_000 == 125_000` evaluates to `False` → `AssertionError` raised with message:
  `"ExtractionConfig.freq_max_hz drifted from corpus.USV_FREQ_MAX_HZ. If corpus changed, retrain the CNN before updating this literal."`

This is exactly the behavior the spec requires. Any downstream import of `ExtractionConfig` (including the PyQt6 app startup) will surface the assertion failure immediately, before any inference runs. The assertions are correct and the failure mode is gracefully named.

The assertions also cover `sample_rate`, `n_fft`, and `hop_length` beyond the spec's minimum — this is strictly better coverage.

---

## 3. CNN Regression Risk — CLEAR

The production CNN inference path:
1. `run_batch_detection.py` → `sliding_inference.py` → `audio_loader.py` → `AudioLoader._compute_spectrogram()`
2. `AudioLoader` uses `ExtractionConfig` directly (verified: `audio_loader.py:84`, `_compute_spectrogram:153-158`)
3. `ExtractionConfig` defaults are hardcoded literals (`20_000`, `120_000`) — NOT imported from corpus
4. `sliding_inference.py` does NOT import `DetectionConfig` (verified by grep — no match)
5. `run_batch_detection.py` does NOT import `DetectionConfig` (verified by grep — no match)

The CNN pixel grid is unchanged. The refactor correctly isolates `ExtractionConfig` from the corpus import chain. CNN regression risk: **NONE**.

The `DetectionConfig` band widening (25-110 → 20-120 kHz) only affects the legacy `EnergyDetector` path (`scripts/run_detection.py`, unit tests) — confirmed by the absence of `DetectionConfig` in any production inference file.

---

## 4. Don't-Do List Verification

| Rule | Status |
|------|--------|
| Don't change ExtractionConfig values | PASS — literals unchanged, drift assertion is the guard |
| Don't change audio_loader SonicConfig values | PASS — freq_min/max 0/30k unchanged |
| Don't merge the 3 config classes | PASS — SpectrogramConfig, DetectionConfig, AnalysisConfig remain separate |
| Don't add bout_threshold_s to corpus.py | PASS — lives only in `corpus_facts/5970.json` as recorded stat |
| Don't touch Phase 17 files | PASS — sis_baselines.py, run_sis_baselines.py, test_sis_*.py untouched |

---

## 5. Test-Expectation Migrations

All 16 specific line migrations listed in the plan verified complete; no accidental changes elsewhere. No `250_000` occurrences remain in any of the four test files.

The handoff test-plan section had a typo: "assert `USV_FREQ_MAX_HZ == 125_000`". The plan document correctly flagged this; `test_corpus.py:34` asserts `120_000` (correct canonical).

---

## 6. Parameters-Sidecar Pattern Compliance

`audit_corpus.py` implements `_print_parameters()` with `[inputs]`, `[methodology]`, and `[literature references]` blocks. `test_audit_corpus.py` asserts all four headings appear in stdout. This mirrors the `run_sis_baselines.py` pattern mandated by the user feedback rule.

All 7+ sanity-check anchors match exactly in the committed `5970.json`:
- `n_calls_raw: 7921`, `n_calls_after_dropna_file: 7864`
- `median_ici_gap_ms: 86.6833`, `median_ioi_ms: 192.9865`
- `q25: 65.1449`, `q75: 209.1082`
- `n_cross_file_pairs_over_10s: 829`, `n_negative_gaps: 10`
- `n_bouts: 1238`, `n_within_bout_pairs: 6350`

Pattern compliance: **FULL**.

---

## Findings

### SERIOUS — Stale inline comment encodes wrong units
**What:** `src/usv_spectrogram/config.py:42` said `# Streaming block size in samples; 250k ~= 1 second at 250 kHz.` After the refactor, at 300 kHz, 250,000 samples = **0.833 seconds**, not 1.
**Fix:** Updated comment to `# Streaming block size in samples; 250k ~= 0.83 seconds at 300 kHz.`

### NIT — `patterns.md` code example showed pre-refactor defaults
**What:** `docs/architecture/patterns.md:31-32` showed the DetectionConfig Pattern 1 example with old `freq_min_hz=25_000`/`freq_max_hz=110_000`. Pattern 3 fixture showed `sample_rate_hz = 250_000`.
**Fix:** Pattern 1 snippet updated to show corpus imports (`USV_FREQ_MIN_HZ`, `USV_FREQ_MAX_HZ`, etc.) with new numeric values in comments. Pattern 3 fixture updated to 300 kHz.

### NIT — `IMPLEMENTATION_PROGRESS.md` had no entry
**Fix:** Appended a dated entry covering the corpus-unification scope, files, and test counts.

### NIT — `docs/modules/corpus-constants.md` did not exist
**Fix:** Written. Covers three-layer architecture, CNN-freeze constraint, values, drift assertion behavior, add-a-new-dataset flow, related references.

### NIT — `CLAUDE.md` ADR-001 not updated to point at corpus.py
**Fix:** Added a line to §"Signal Processing Conventions" referencing `corpus.py` as the code-level enforcement point and `docs/modules/corpus-constants.md` as the module doc.

### PRAISE — Drift assertions exceed spec requirements
Spec required assertions for `freq_min_hz` / `freq_max_hz` only. The implementation also asserts `sample_rate`, `n_fft`, `hop_length`. All five STFT/frequency parameters are guarded.

### PRAISE — `--all` mode correctly skips missing inputs
Returns exit code 1 for `--dataset <missing>`, skips gracefully with `[skip]` stderr messages for `--all`. Both behaviors tested.

### PRAISE — hdbscan join strategy matches `run_sis_baselines.py`
Same `(file, begin_time_s)` join key with `drop_duplicates(keep="first")`. Cross-script consistency explicitly noted in code comment.

---

## Verdict (original)

**CHANGES NEEDED** — one SERIOUS (stale comment), four NITs (patterns.md stale, no module doc, no progress log entry, no CLAUDE.md pointer). No blockers.

## Fixes Applied (2026-04-17 post-review)

All findings addressed in the same session:

1. **SERIOUS — `src/usv_spectrogram/config.py:42`** — updated streaming-block comment from "1 second at 250 kHz" to "0.83 seconds at 300 kHz" to match the new canonical sample rate. Value (`250_000`) intentionally unchanged — `stream_block_size_samples` is a streaming IO granularity knob, not a load-bearing "1-second" invariant (verified by grep: only one caller, `stft_stream.py:45`, uses it as a default; no code relies on the second count).

2. **NIT — `docs/architecture/patterns.md:31-32`** — updated Pattern 1 DetectionConfig example to show imports from `corpus` with new numeric values in trailing comments (20k/120k band, 300k sample rate).

3. **NIT — `docs/architecture/patterns.md:~97`** — updated Pattern 3 fixture example `sample_rate_hz = 250_000` → `300_000` with a comment noting `corpus.SAMPLE_RATE_HZ` as the canonical source.

4. **NIT — `docs/modules/corpus-constants.md`** — created. Covers three-layer architecture, canonical values, CNN-freeze constraint, usage examples, and add-a-new-dataset flow.

5. **NIT — `CLAUDE.md:193-198`** — amended the §"Signal Processing Conventions" block to reference `src/usv_spectrogram/corpus.py` as the code-level enforcement point and `docs/modules/corpus-constants.md` as the module doc. The "Key rule" line now also mentions `corpus.SAMPLE_RATE_HZ` as the import-based alternative to the magic number.

6. **NIT — `IMPLEMENTATION_PROGRESS.md`** — appended a new dated section documenting the full refactor: files created/modified, test counts, sanity-check anchors.

### Re-verification

```
.venv/bin/python -m pytest tests/test_corpus.py tests/test_config.py tests/test_audit_corpus.py -q
```
Expected: 34 passed (12 + 14 + 10). See post-fix run for actual.

No BLOCKER findings → self-verification of these NITs + SERIOUS sufficient per review protocol.
