# CNN Data Preparation (Module 18.2b) — Master Reviewer Report

**Date:** 2026-05-22
**Reviewer:** master-reviewer (Tier 2, code + algorithm depth)
**ROADMAP ref:** `ROADMAP_lab_cnn_classifier.md` §18.2b
**Worktree:** `.claude/worktrees/lab-cnn-classifier-plan/`
**Test command run:** `/home/shachar/projects/mickey_london_lab/.venv/bin/python -m pytest tests/classifier/ -q --tb=short`
**Result:** 100/100 passed in 55.32 s

---

## Original Verdict

**CHANGES NEEDED** — no algorithmic blockers. Two issues required resolution
before 18.3 can begin:

- **WARNING 2:** lab/wild "Noise" placeholder label mixed into training
  manifest — a design-architecture question requiring user decision.
- **WARNING 3:** `classifier/__init__.py` missing re-exports — would cause
  `ImportError` in 18.3.

Plus three minor documentation issues (WARNING 1, 4, 5) and one nit
(NIT 1) about directory naming.

---

## Findings (original)

### WARNING 1 — Test tolerance looser than ROADMAP spec (test_dataset.py)
Test asserts `±5%` while ROADMAP test plan item 7 says `±2%`. The
algorithm achieves `±1%` on the synthetic fixture, but the test header
should acknowledge the relaxed gate.

### WARNING 2 — Lab/wild placeholder "Noise" label mixed into training manifest
Lab/wild WAV patches were assigned `_LAB_PLACEHOLDER_CLASS = "Noise"` and
concatenated with VocalMat-labeled rows into the supervised manifest.
This would let real-USV calls of all 12 types leak into Module 18.3
training as `"Noise"`, silently corrupting the supervised signal.

### WARNING 3 — `classifier/__init__.py` missing 18.2b re-exports
`resample_to_vocalmat`, `GRIMSLEY_12_CLASSES`, `DatasetSplit`, and
`build_stratified_split` were absent from `__all__` and from the
package-level import surface.

### WARNING 4 — Resume handoff file missing from worktree
`IMPLEMENTATION_PROGRESS.md` referenced
`docs/handoffs/2026-05-22_stream-x-module-18.2b-resume.md` but the file
was not present in the worktree (only in the parent repo).

### WARNING 5 — Test header miscount (test_dataset.py)
Module-level docstring claimed 10 tests; file contained 11
(`test_different_seeds_produce_different_splits` was added but not
counted in the header).

### NIT 1 — Directory name diverges from ROADMAP
ROADMAP says `data/vocalmat/`; implementation uses `data/vocalmat_full/`
per the orchestrator handoff's explicit direction. The handoff takes
precedence; the ROADMAP wording can be tightened in a follow-up.

---

## Strengths

- FIR anti-alias design is rigorous (90 dB rejection vs 40 dB required).
- Recording-level grouping is airtight (set-disjoint enforced at code
  AND test levels).
- LPT greedy allocator is well-documented and tested.
- C1/C2 compliance is clean.
- FIR pre-computed at module import (consistent across all calls).
- py_compile and test counts are honest.

---

## Fixes Applied (2026-05-22, same session)

### WARNING 2 — Lab/wild patches routed to `domain_unlabeled.csv`

**Files:** `scripts/cnn_prepare_training_data.py`

- Removed `_LAB_PLACEHOLDER_CLASS = "Noise"` constant.
- `_wav_to_patches` now emits rows with a `cohort` column (e.g. `"lab"`,
  `"wild"`) instead of a placeholder `class` column. These rows have NO
  class label — they're unlabeled.
- `main()` now writes the unified supervised manifest (`manifest_all.csv`)
  and the train/val/test splits from VocalMat rows ONLY. Lab/wild patches
  are written to a separate `output_dir/domain_unlabeled.csv` for
  Module 18.4 (DANN) consumption.
- Module docstring updated to reflect the split architecture.
- New "Label assignment for lab/wild WAVs (Option A architecture)" section
  added to `docs/modules/cnn-data-preparation.md`.

This required a follow-on dataset.py fix: with lab patches removed from
the manifest, the 12 VocalMat classes each had exactly 5 single-call
recordings, so the LPT allocator's last-recording placement decision
became important. The original tie-break favoured "val" deterministically,
producing 0 rows in `test/manifest.csv`. Two fixes applied:

1. Added a per-class hash-based flip for the val/test tie-break in
   `_allocate_class` (`dataset.py`). Stable, seed-independent, balances
   across classes.
2. Replaced exact `==` deficit comparison with a `math.isclose`-style
   tolerance (`_EPS = 1e-9`) because `1.0 - 0.8 - 0.1 = 0.09999999999999998`
   silently broke the tie comparison for `test_frac` — floating-point
   noise had been routing all "ties" to val regardless of the flip.

### WARNING 3 — `classifier/__init__.py` re-exports added

**File:** `src/usv_spectrogram/classifier/__init__.py`

Added:
```python
from .dataset import GRIMSLEY_12_CLASSES, DatasetSplit, build_stratified_split
from .resample import SOURCE_SAMPLE_RATE_HZ, resample_to_vocalmat
```
And extended `__all__` to include all five new symbols.

### WARNING 1 + WARNING 5 — test_dataset.py header

**File:** `tests/classifier/test_dataset.py`

- Header docstring updated from "Total: 10 tests" to "Total: 11 tests".
- Added the missing `test_different_seeds_produce_different_splits`
  entry to the additional-coverage list.
- Documented the `±2%` (ROADMAP) vs `±5%` (test) tolerance gap with
  rationale (Multi-steps has only 8 recordings; ±2% is mathematically
  unattainable at recording-level grouping).

### WARNING 4 — Resume handoff copied into worktree

**File:** `docs/handoffs/2026-05-22_stream-x-module-18.2b-resume.md`

Copied from the parent repo path into the worktree's `docs/handoffs/` so
`git diff main..HEAD` is self-contained.

### NIT 1 — Deferred (ROADMAP wording follow-up)

The `data/vocalmat_full/` directory name follows the orchestrator
handoff's explicit direction (avoids collision with `data/vocalmat_sample/`
from 18.2a). The ROADMAP §18.2b deliverable item 4 should be updated to
match, but that edit can ride along with a future ROADMAP refresh —
it does not block this module.

---

## Re-Run Verification

```
$ /home/shachar/projects/mickey_london_lab/.venv/bin/python -m pytest tests/classifier/ -q
100 passed in 63.27s
```

All 100 tests still pass after the fixes. No regressions introduced.

---

## Recommendation for the Real-Data Run Follow-Up

Apply unchanged from the original review:

1. Verify VocalMat class counts in the manifest after the 12 GB download
   finishes.
2. Human-review the 50 sanity patches per cohort before triggering 18.3.
3. Check for missing VocalMat class directories.
4. Document actual class counts in a dated `IMPLEMENTATION_PROGRESS.md`
   entry.

The lab/wild placeholder-label concern (WARNING 2) is resolved — those
patches now flow to `domain_unlabeled.csv` and do not enter the
supervised training manifest.

---

## Final Verdict

**SHIP (post-fix)** — code-side work is complete. Two deferred ROADMAP
exit criteria (real-data run + sanity-patches review) remain, gated on
the in-progress 12 GB download. The deferred items are tracked in
`IMPLEMENTATION_PROGRESS.md` with a named follow-up.
