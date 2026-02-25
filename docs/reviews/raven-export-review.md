# Raven Selection Table Export Adapter Module Review

**Reviewed by:** Master Reviewer (Claude Sonnet 4.6)
**Date:** 2026-02-23
**Module:** Phase 14.1 — Raven Selection Table Export Adapter
**Tier:** 2 (Standard)
**Verdict:** APPROVED WITH WARNINGS

---

## Overview

The implementation covers all four required files, all 31 tests pass, and the core business logic is correct. No DSP parameters are involved (this is a pure format-conversion module), no data leakage risk exists (no splits, no ML), and the test expected values are computed from first principles, not copied from failing output. There are no blockers. There are three warnings around a reporting inaccuracy, a missing CLI test, and missing documentation artifacts.

---

## Previously Fixed (Round 1 review)

These issues were found and fixed during the initial implementation:

1. **Prefix match ordering** — `discover_wav_detection_mapping` sorted candidates by descending length so longest stem wins. Test added: `test_prefix_match_prefers_longest_stem`.
2. **wav_dir validation** — Added `FileNotFoundError` for missing wav_dir in both `discover_wav_detection_mapping` and `export_raven_tables`. Test added: `test_missing_wav_dir_raises`.

---

## BLOCKERS (must fix before next module)

None.

---

## WARNINGS (fix soon)

### W1. `unmapped_dirs` in `export_summary.json` conflates empty-matched dirs with genuinely unmapped dirs

**File:** `src/usv_spectrogram/classification/raven_export.py`, lines 313-320

**Problem:** The `unmapped` set is computed as `all_subdirs - mapped_dir_names`. Because `discover_wav_detection_mapping` drops directories that matched a WAV stem but contained zero detection JSONs, those empty-but-matched directories never enter `mapping`, so they appear in `unmapped_dirs` even though they did have a matching WAV. A downstream user reading `export_summary.json` would be misled.

**Fix:** Add a comment documenting this behavior, and/or separate into `unmapped_dirs` (no WAV) vs `empty_detection_dirs` (matched WAV but no JSONs). Add a test.

### W2. Test plan item 10 (CLI `--dry-run` behavior) is not tested

**File:** `tests/test_classification/test_raven_export.py` — no test class for CLI

**Problem:** The `--dry-run` mode is the recommended first step in the exit criteria. The `main()` function in `scripts/export_raven_tables.py` is untested.

**Fix:** Add a test that patches `sys.argv` with `--dry-run` and verifies no files are written and exit code is 0.

### W3. Module documentation (`docs/modules/raven-export.md`) does not exist

**File:** `docs/modules/` — no `raven-export.md` present

**Problem:** Other modules have `docs/modules/*.md` entries. The established pattern requires one per module.

**Fix:** Create `docs/modules/raven-export.md` documenting the public API, detection JSON format, Raven TSV format, and naming convention.

---

## SUGGESTIONS (nice to have)

| # | Issue | File | Fix |
|---|-------|------|-----|
| S1 | `load_detection_json` docstring says it skips tracking files — that's `_is_detection_json`'s job | `raven_export.py:140` | Correct the docstring |
| S2 | Required CLI args have project-specific defaults | `export_raven_tables.py:56,62` | Consider `required=True` or add comment |
| S3 | `export_summary.json` field names diverge from ROADMAP spec | `raven_export.py:107` | Document or add alias |
| S4 | Case-sensitive `.json` suffix check | `raven_export.py:163` | Add `.lower()` |

---

## Summary

| Severity | Count | Items |
|----------|-------|-------|
| BLOCKER | 0 | -- |
| WARNING | 3 | W1, W2, W3 |
| SUGGESTION | 4 | S1, S2, S3, S4 |

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc (`docs/modules/raven-export.md`) | MISSING | No module doc created |
| `docs/architecture/patterns.md` | UP TO DATE | No new patterns; existing patterns followed |
| `DECISIONS.md` | UP TO DATE | No new ADRs needed |

---

## Fix Log

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| W1 | FIXED | `raven_export.py:312-340` | 2026-02-23 | Separated `unmapped_dirs` (no WAV) from `empty_detection_dirs` (matched WAV, no JSONs). Added `ExportSummary.empty_detection_dirs` field. Added test `test_empty_matched_dirs_not_in_unmapped`. |
| W2 | FIXED | `test_raven_export.py:TestCliDryRun` | 2026-02-23 | Added `test_dry_run_writes_no_files` that patches sys.argv with --dry-run and verifies no output files + exit code 0. |
| W3 | FIXED | `docs/modules/raven-export.md` | 2026-02-23 | Created module doc with full public API, format descriptions, CLI usage, key decisions, integration points. |
| S1 | FIXED | `raven_export.py:125` | 2026-02-23 | Updated docstring to clarify skip responsibility belongs to `_is_detection_json`. |
| S2 | OPEN | | | Deferred — project-specific defaults are convenient for the primary use case. |
| S3 | FIXED | `raven_export.py:113` | 2026-02-23 | Added `unmapped_count` key to `to_dict()` to match ROADMAP spec field name. |
| S4 | FIXED | `raven_export.py:160-163` | 2026-02-23 | Added `.lower()` to suffix checks for case-insensitive matching. |

## Post-Fix Verification

- `py_compile`: All files compile cleanly
- `pytest tests/test_classification/`: **33/33 passed** (31 original + 2 new)
- No regressions in full suite
