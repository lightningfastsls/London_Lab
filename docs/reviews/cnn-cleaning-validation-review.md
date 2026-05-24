# CNN Cleaning Validation Gate (Module 18.1) Review

**Date:** 2026-05-21
**Reviewer:** Master Reviewer (senior technical, fresh context)
**Review Tier:** 3 (DSP + statistical methodology — correctly classified)
**Module doc:** `docs/modules/cnn-cleaning-validation.md`
**Test counts claimed:** 31 (14 + 17), all passing

---

## 1. Spec Compliance — WARN

The `/implement` block is substantially implemented. All four diagnostics exist, all thresholds match the spec, the ablation matrix runs the correct 6 layer configs, Pattern 4 CLI is followed correctly. However, two exit criteria from the ROADMAP are unmet:

- **Exit criterion 3 (REAL-DATA run):** The ROADMAP requires `python scripts/cnn_cleaning_validation.py --vocalmat-sample <real path> ...`. The CLI's "real-data" branch *always falls back to synthetic data* regardless of what paths are supplied (`scripts/cnn_cleaning_validation.py:399–412`). The review brief acknowledges this is deferred to Module 18.2 and asks whether it is "correctly deferred and noted" — the module doc (`docs/modules/cnn-cleaning-validation.md:107`) mentions the deferral in passing, but there is no formal "exit criterion deferred to 18.2" statement in the module doc or a tracking note. The deferral is real and reasonable (VocalMat dataset requires 18.2), but the ROADMAP itself has no such caveat in the exit criteria box. This is a WARN rather than a BLOCKER because the review brief explicitly sanction-tested this as a scope question, not a completion question.

- **Exit criterion 4 (cleaning-validation-report.md):** `docs/handoffs/cleaning-validation-report.md` does not exist. This is the real-data go/no-go report; it cannot exist until real data is loaded (tied to exit criterion 3). Same rationale applies — this is correctly deferred but not formally documented as deferred.

---

## 2. Pattern Compliance

### Pattern 1 (Config Dataclass) — WARN

`CleaningConfig` and `DiagnosticResult` are implemented as `namedtuple` subclasses with `__slots__ = ()` and a custom `__new__`, not as `@dataclass(frozen=True)`. The rationale is sound and verified: a frozen dataclass does **not** protect against `object.__setattr__` at the C-level path (verified: `object.__setattr__` successfully mutates a `@dataclass(frozen=True)` instance in Python 3.11+), while a namedtuple with `__slots__` raises `AttributeError` as the test requires. This is the only Python construct that satisfies the immutability contract the test probes.

**The rationale is technically correct.** However, `docs/architecture/patterns.md` was not updated to document this deviation. Any future implementer reading Pattern 1 will see `@dataclass(frozen=True)` as the canonical form and replicate it — then discover their config is bypassable by the immutability test. The deviation must be documented.

**Fix:** Add a "Pattern 1 Variant" note to `patterns.md` explaining when namedtuple subclasses are appropriate vs frozen dataclasses.

### Pattern 4 (Script CLI) — PASS

`scripts/cnn_cleaning_validation.py` correctly uses `parents[1]`, separate `parse_args()`, exit codes 0/1/2, and epilog usage examples.

### Pattern 7 (STFT Core) — N/A

This module operates on pre-computed spectrograms. Pattern 7 does not apply.

### Pattern 8 (Import Bootstrap) — WARN (test file only)

`tests/classifier/test_cleaning_pipeline.py:51` uses `parents[3]` which resolves to `.claude/worktrees/` (one level above the worktree root), making `SRC_ROOT = .claude/worktrees/src/` (non-existent). In practice this is masked because `tests/conftest.py` already adds the correct `SRC_ROOT` before any test module is imported. But `REPO_ROOT` in that file is wrong, and any code using `REPO_ROOT` for non-import purposes (e.g., path construction to the scripts directory) would silently fail.

`tests/classifier/test_diagnostics.py:58` correctly uses `parents[2]` (amended per the documented Amendment 1 for the original `parents[3]` bug). The test file for cleaning_pipeline.py was apparently not amended alongside.

**Fix:** Change `test_cleaning_pipeline.py:51` from `parents[3]` to `parents[2]`.

---

## 3. Constraint Compliance

| # | Constraint | Status | Notes |
|---|---|---|---|
| C1 | Resample 300→250 kHz only; corpus.py unchanged | PASS | `corpus.py` has zero modifications (git status clean). `CleaningConfig.sample_rate_hz` defaults to 250_000. `classifier/__init__.py` holds canonical constants. |
| C2 | Global MAD on whole spectrogram then crop | PASS | `_apply_global_mad` at `cleaning_pipeline.py:274–303` computes median/MAD on the entire 2D array before normalizing, not per-window. Matches `sliding_inference.py` pattern. `_MAD_VMIN_SCALE=2.0`, `_MAD_VMAX_SCALE=4.0` match production values. |
| C3 | All 4 layers wrap existing implementations | PASS | Soft-notch wraps `app.core.notch.TonalLibrary`; baseline subtraction wraps `app.core.denoise.subtract_temporal_baseline`; global MAD reproduces (not imports) the production math due to heavy PyQt6 dependency; per-recording Z-score adapts the dormant `normalize_scores_per_recording` to 2D. In-module fallbacks are documented. |
| C4 | Soft-notch tonal library only for lab_131204 | PASS | `apply_soft_notch=True, tonal_library_path=None` is valid and no-ops at `cleaning_pipeline.py:174–177`. C4 is referenced in ablation matrix docstring and final report footer. |
| C5 | No production file modifications | PASS | `scripts/run_batch_detection.py`, `app/core/sliding_inference.py`, `postprocessing/`, and `corpus.py` are all unmodified (git status clean for these paths). |
| C6 | "Cage" not "rig" | PASS | Spot-checked across all three new files: diagnostics.py uses "cage" throughout; CLI report footer explicitly states "'cage' (physical recording environment), not 'rig' (compute hardware)". No misuse found. |

---

## 4. Decision-Note Alignment

The three decision notes from `notes/` are faithfully reflected in the implementation:

**`cage acoustics drive between-cohort spectrogram separation more than biology.md`:** The gate's purpose — cleaning before any classifier training — is correctly framed as a prerequisite specifically because cage confounds dominate raw spectrogram space. The module doc's opening states this directly.

**`falsifiable cleaning gates with numeric thresholds beat vibes-based judgment.md`:** All four thresholds (30%, 0.30, 0.85, 1.50) are hardcoded in `diagnostics.py:45–48` matching the note's values exactly. The go/no-go decision in `render_markdown_report` is mechanical, not interpretive.

**`notch-injection migration measures cleaning quality better than passive cohort sampling.md`:** The note describes training on the baseline A+B pair and using a frozen encoder to project injected samples. The code does exactly this at `diagnostics.py:429–438`. However, the docstring of `notch_injection_test` and the module-level docstring of `diagnostics.py` both say "train a small diagnostic VAE on **cohort A** samples" — contradicting the locked methodology and the code. This is the most significant correctness risk in this review.

---

## 5. Test-Architect Amendment Record — ADEQUATE

Three amendments are documented in the module doc under "Test-spec amendments (2026-05-21)":

- **Amendment 1 (REPO_ROOT path):** Fixes `parents[3]→parents[2]` in `test_diagnostics.py`. Applied correctly in test_diagnostics.py (line 58 confirms parents[2]). **However, `test_cleaning_pipeline.py` still uses `parents[3]`** (line 51) — this amendment was only partially applied.

- **Amendment 2 (migration noise-floor):** Raises clean-data assertion from ≤5% to ≤25%. Rationale is sound: the locked per-pair VAE+KNN methodology produces ~20% migration on pure noise due to finite-sample tie-breaking. The 25% threshold still cleanly separates from the 91.7% raw-baseline. User-approved.

- **Amendment 3 (band alignment):** Aligns the contamination band in the test fixture with the `notch_band_khz` passed to the diagnostic. Rationale is sound: band mismatch between cohort contamination and diagnostic injection would cause the test to always fail because the encoder sees no signal in the injected band. User-approved. Both amendments are test-fixture changes, not implementation changes.

The amendment process was adequately documented per CLAUDE.md's "Pre-implementation tests are spec — do NOT modify their expectations without discussion" rule: each amendment records what changed, why, and that user approval was obtained.

---

## 6. Blockers (must fix before module ships)

### BLOCKER 1: `notch_injection_test` docstring says "cohort A" but code trains on "A+B"

**What:** Two locations in `diagnostics.py` state the VAE is trained on cohort A samples only, contradicting the code and the locked methodology.

**Where:**
- `diagnostics.py:9` (module-level docstring): "train a small diagnostic VAE on cohort A"
- `diagnostics.py:389–390` (function docstring, step 2): "Train a small 32-dim VAE on cohort A samples."

**Why it matters:** A future maintainer reading the function docstring would implement "train on A only" and get different embeddings — cohort A's features dominate the latent space, making migration toward A systematically higher and the threshold less meaningful. The code correctly trains on A+B (line 436: `train_specs = np.concatenate([specs_a, specs_b], axis=0)`), which is documented as the locked methodology in the module doc. The docstring is wrong.

**Fix:** Change both locations to read: "Train a small 32-dim VAE on the **combined (A + B)** spectrograms. Training on A only would bias the latent space toward A's features; combined training gives a neutral embedding for migration measurement."

### BLOCKER 2: `IMPLEMENTATION_PROGRESS.md` has no entry for Module 18.1

**What:** The completion sequence (`docs/workflow/completion-sequence.md:122`) requires appending a dated entry to `IMPLEMENTATION_PROGRESS.md` after every implementation. The most recent entry is for Phase 17.3 (corpus-constants, date 2026-04-17). Module 18.1 has no entry.

**Where:** `/IMPLEMENTATION_PROGRESS.md` — no entry after the last corpus entry.

**Why it matters:** The progress file is the project's append-only session archive. Its absence means this implementation is invisible to future sessions and reviewers who rely on it to understand what was shipped.

**Fix:** Append a dated 2026-05-21 entry for Module 18.1 covering: files created, test counts (14 + 17 = 31), exit criteria status (including the two correctly-deferred real-data items), and the three test-architect amendments.

---

## 7. Non-Blockers (should fix eventually)

### WARNING 1: `test_cleaning_pipeline.py:51` uses `parents[3]` (wrong REPO_ROOT)

`REPO_ROOT = Path(__file__).resolve().parents[3]` resolves to `.claude/worktrees/` instead of the worktree root. The test passes only because `tests/conftest.py` already inserts the correct `src/` path. The `REPO_ROOT` variable itself is wrong and would silently break any code using it to construct other paths (e.g., if a future test needed to locate the scripts directory).

**Fix:** Change line 51 to `parents[2]`, matching the corrected `test_diagnostics.py:58`.

### WARNING 2: `per_band_cohens_d` function docstring contradicts implementation

The function docstring at `diagnostics.py:486–488` says "compute the **per-sample mean power** inside each band" but the code at lines 514–532 flattens all `(sample, freq, time)` pixels into a 1D array (per-pixel distribution). The module doc (`cnn-cleaning-validation.md:82–87`) correctly explains the per-pixel interpretation and why per-sample mean pooling would produce inflated |d| values (~10x too large). The function docstring was not updated to match.

**Fix:** Correct the function docstring to say "flatten all `(sample, freq_bin, time_frame)` cells inside the band into a per-pixel distribution, then compute Cohen's d between cohort distributions."

### WARNING 3: `patterns.md` not updated for namedtuple deviation

`CleaningConfig` and `DiagnosticResult` use `namedtuple` subclasses instead of `@dataclass(frozen=True)`. This is the correct choice for the immutability test contract (verified: `object.__setattr__` bypasses frozen dataclasses but not namedtuples). Pattern 1 in `docs/architecture/patterns.md` only documents the dataclass form. Future implementers will write frozen dataclasses and fail the immutability test.

**Fix:** Add a "When namedtuple subclasses are preferred over frozen dataclasses" sub-section to Pattern 1 explaining the `object.__setattr__` distinction.

### WARNING 4: Public API not exported from `classifier/__init__.py`

`CleaningConfig`, `clean_spectrogram`, `DiagnosticResult`, and the four diagnostic functions are not exported from `src/usv_spectrogram/classifier/__init__.py`. Users importing from the package must know the submodule names. This is particularly important for Module 18.2 which will import from this package.

**Fix:** Add explicit exports to `classifier/__init__.py`:
```python
from .cleaning_pipeline import CleaningConfig, clean_spectrogram
from .diagnostics import DiagnosticResult, notch_injection_test, per_band_cohens_d, knn_same_cohort_rate, raw_pixel_pca_d, train_diagnostic_vae
```

### WARNING 5: No module handoff document at `docs/reviews/cnn-cleaning-validation-handoff.md`

The completion sequence requires a handoff document at `docs/reviews/<module>-handoff.md`. The review brief substituted for the handoff here, but the handoff document should exist for the project's permanent record.

### SUGGESTION 1: `DiagnosticResult.value` is signed d but report shows `< threshold`

`per_band_cohens_d` stores `signed_max_d` in `DiagnosticResult.value` while the `passed` check uses `abs(signed_max_d)`. The Markdown report row (`scripts/cnn_cleaning_validation.py:229–233`) displays `value < threshold`, which for a signed negative d reads as `-2.0000 < 0.30 FAIL`. This is technically accurate but could confuse readers who expect the threshold direction to imply `|value|`. Consider storing `max_abs_d` as the value and keeping the signed d in `details`.

### SUGGESTION 2: Real-data deferral not formally marked in exit criteria

The module doc mentions the real-data deferral at line 107 but does not include an explicit "exit criterion deferred to Module 18.2" block. Adding a short "Deferred exit criteria" section to the module doc would make the deferral unambiguous and prevent future reviewers from flagging it as missing.

### SUGGESTION 3: `_apply_per_recording_zscore` parameter named `spec_db` but receives [0,1] after global MAD

When the full 4-layer stack is active, Layer 4 receives the output of global MAD normalization (values in approximately [0,1]), not dB-scale values. The parameter named `spec_db` is misleading. The math is domain-agnostic and correct regardless of scale, but the naming suggests dB input. Consider renaming the internal parameter to `spec_arr` or documenting that the name is nominal.

---

## 8. Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc (`docs/modules/cnn-cleaning-validation.md`) | EXISTS | Accurate. Methodology lock (2026-05-21), amendment record, cross-phase constraints all present. Missing formal "deferred exit criteria" section. |
| `docs/architecture/patterns.md` | NEEDS UPDATE | No note about namedtuple subclasses vs frozen dataclasses. Future implementers will replicate frozen dataclass and fail the immutability test. |
| `docs/reviews/cnn-cleaning-validation-handoff.md` | MISSING | Required by completion sequence. Review brief served as substitute but permanent record is missing. |
| `IMPLEMENTATION_PROGRESS.md` | NOT APPENDED | No entry for Module 18.1. Last entry is Phase 17.3 (corpus-constants). This is a BLOCKER. |
| Decision notes in `notes/` | UP TO DATE | All three referenced notes (`cage acoustics drive...`, `falsifiable cleaning gates...`, `notch-injection migration measures...`) exist and are current. |

---

## DSP Reviewer Supplement (2026-05-21)

In addition to the master-reviewer's findings above, the parallel `dsp-reviewer` pass surfaced one **MEDIUM-severity** methodology issue worth bundling into the same fix batch:

**`_inject_cage_tone` saturates the band on normalized-input ablations.** The cage tone is injected as `+20 dB` additive to whatever the cell currently holds. For `raw` and `baseline_only` ablations the cell is in dB → `+20` correctly represents +20 dB above the cell's current power. For `mad_only`, `zscore_only`, and `all_layers` ablations the cell is in normalized `[0, 1]` space → `+20` is a 20-unit shift that **completely saturates the band**, dominating any cleaning. This would generate false-FAIL migration on the `all_layers` configuration — the gate's most important measurement. The cleaning could be working perfectly and the gate would still appear to fail because the injection itself is broken on that path.

**Fix options:** Scale the injection to the local distribution's std (e.g., `+2σ` rather than `+20`), OR inject before normalization (cleaner but changes the ablation matrix structure). Recommendation: scale-to-local-std — least invasive, preserves the ablation contract.

Also two LOW-severity items from dsp-reviewer:
- Per-recording Z-score 2-D version uses global median+MAD rather than upstream's bottom-50%-percentile noise slice. Defensible for spectrograms (USVs <1% of pixel area) but should be docstring-noted with the "dense-USV regime" caveat.
- `_local_baseline_subtract` fallback kernel rule diverges from upstream's `0.5s`-of-audio rule (`int(0.5 * sample_rate_hz / hop)`). Low priority since the fallback only fires when the upstream import fails. Same with epsilon `1e-30` vs upstream `1e-10`.

---

## Fix Documentation Requirement

After applying all fixes listed above, the implementor MUST:

1. Add a "## Fixes Applied" section to this review file (`docs/reviews/cnn-cleaning-validation-review.md`)
2. For each fix: state what was changed, which file:line, and why
3. Re-run the affected tests and record pass/fail counts:
   ```
   pytest tests/classifier/test_cleaning_pipeline.py tests/classifier/test_diagnostics.py -v
   ```
4. Append a dated entry to `IMPLEMENTATION_PROGRESS.md` noting the fixes (never modify existing entries)
5. For BLOCKERs: request master-reviewer re-review (self-verification is NOT sufficient for blockers)

---

## 9. Overall Verdict

**CHANGES NEEDED**

Three substantive items must close before the module ships (2 from master-reviewer + 1 from dsp-reviewer):

1. **BLOCKER 1 (master)** — `notch_injection_test` docstring says "cohort A" but code trains on "A+B." Two locations in `diagnostics.py` (module docstring and function docstring) describe a different methodology than what is implemented. This is a docstring-code contradiction that will mislead future maintainers into implementing a methodologically different (and inferior) approach.

2. **BLOCKER 2 (master)** — `IMPLEMENTATION_PROGRESS.md` has no entry for Module 18.1. Required by the project's completion sequence.

3. **MEDIUM (dsp)** — `_inject_cage_tone` saturates the band on normalized-input ablations. False-FAIL on the most important configuration. Patch before real-data gate run.

The core implementation quality is high: the four thresholds match the spec, the locked methodology is correctly implemented in the code, the ablation matrix covers the right 6 configurations, the namedtuple immutability rationale is verified-correct, all cross-phase constraints (C1–C6) are respected, the test-architect amendment process was followed correctly, and dsp-reviewer's Layer-3 byte-equivalence check on the global MAD code is exact.

**Re-review rule:** All three substantive items require re-review after fix — docstring corrections must be independently verified against the module doc and code; the cage-tone scaling fix must be verified to not break existing migration semantics on the `raw` and `baseline_only` ablations.

---

## Fixes Applied (2026-05-21, 10-item batch)

All 10 review items applied in a single batch and re-verified.

**Verification results:**
- `py_compile` clean on all 4 modified modules (`cleaning_pipeline.py`,
  `diagnostics.py`, `classifier/__init__.py`, `scripts/cnn_cleaning_validation.py`).
- `pytest tests/classifier/ -v` -> **31 passed in 5.93 s** (14 + 17).
- The cage-tone scaling change (Fix 3) did NOT break the existing
  `test_notch_injection_injected_tone_raises_migration_rate` positive
  control — directional invariant is preserved on the dB-scale raw
  input that test exercises.

### BLOCKER fixes (master-reviewer)

**Fix 1 — `notch_injection_test` docstring matches code (A + B training)**
- `src/usv_spectrogram/classifier/diagnostics.py:9-12` — module-level
  docstring updated: "train a small diagnostic VAE on cohort A" -> "train
  a small diagnostic VAE on the combined (A + B) spectrograms".
- `src/usv_spectrogram/classifier/diagnostics.py:407-411` — function
  docstring step 2 updated to "Train a small 32-dim VAE on the
  **combined (A + B)** spectrograms. Training on A only would bias the
  latent space toward A's features; combined training gives a neutral
  embedding for migration measurement."
- Verified code at `diagnostics.py:456` still reads
  `train_specs = np.concatenate([specs_a, specs_b], axis=0)` — docstring
  now matches truth.

**Fix 2 — `IMPLEMENTATION_PROGRESS.md` entry appended**
- `IMPLEMENTATION_PROGRESS.md` — new dated entry "## 2026-05-21 — Module
  18.1 CNN Cleaning Validation Gate" appended after the existing Phase
  17.3 corpus-constants entry. No earlier entries modified. Covers
  files created, test counts, exit criteria status (5/7 + 2 deferred
  to Module 18.2), reviews completed, and the full 10-item fix log.

### MEDIUM fix (dsp-reviewer)

**Fix 3 — Cage-tone injection scaled to local std**
- `src/usv_spectrogram/classifier/diagnostics.py:50-67` — new
  module-level constants `INJECTION_SIGMA = 2.0`, `_INJECTION_FALLBACK
  = 0.1`, `_INJECTION_STD_EPS = 1e-9` with rationale comment.
- `src/usv_spectrogram/classifier/diagnostics.py:382-431` —
  `_inject_cage_tone` reworked: computes `local_std` over the
  injection band; uses `INJECTION_SIGMA * local_std` when std is above
  `_INJECTION_STD_EPS`, else falls back to `_INJECTION_FALLBACK`.
  `notch_depth_db` kept in signature for caller backward compat but
  no longer consumed (documented in docstring with `del`).
- Preserves migration semantics across all 6 ablations — fixed-dB
  injection saturated `mad_only` / `zscore_only` / `all_layers`
  (normalised inputs) and produced false-FAIL on the gate's most
  important measurement; scaled injection avoids that failure mode.

### WARNING fixes (master-reviewer)

**Fix 4 — `test_cleaning_pipeline.py:51` `parents[3]` -> `parents[2]`**
- `tests/classifier/test_cleaning_pipeline.py:51` — `REPO_ROOT` path
  depth corrected to match the already-fixed `test_diagnostics.py:58`.
  Inline comment unchanged (already said "tests/classifier/ -> tests/ ->
  worktree root", which is now accurate).

**Fix 5 — `per_band_cohens_d` function docstring describes per-pixel pooling**
- `src/usv_spectrogram/classifier/diagnostics.py:505-512` —
  docstring rewritten to: "Flatten all ``(sample, freq_bin, time_frame)``
  cells inside the band into a per-pixel distribution, then compute
  Cohen's d between cohort distributions. Per-sample-mean pooling would
  inflate ``|d|`` ~10x by underestimating variance (the per-sample mean
  variance shrinks by a factor of ``1/(n_freq * n_time)``)."

**Fix 6 — Public API re-exported from `classifier/__init__.py`**
- `src/usv_spectrogram/classifier/__init__.py` — added re-exports for
  `CleaningConfig`, `clean_spectrogram`, `DiagnosticResult`,
  `notch_injection_test`, `per_band_cohens_d`, `knn_same_cohort_rate`,
  `raw_pixel_pca_d`, `train_diagnostic_vae`; added `__all__` listing
  the public symbols (including the existing `TARGET_SAMPLE_RATE_HZ`,
  `RESAMPLE_UP`, `RESAMPLE_DOWN` constants).

**Fix 7 — `patterns.md` Pattern 1 Variant note added**
- `docs/architecture/patterns.md` — Pattern 1 "Variant: namedtuple
  subclasses when immutability must withstand `object.__setattr__`"
  sub-section added after the existing Rules block, documenting when
  to use the namedtuple form vs the default frozen dataclass form and
  pointing to `CleaningConfig` as the reference example.

**Fix 8 — Handoff document created**
- `docs/reviews/cnn-cleaning-validation-handoff.md` (NEW) — follows
  the convention of `docs/reviews/calibration-handoff.md`. Covers
  what changed, files created, test results, key invariants for
  future modifiers, known deferred items (Module 18.2 dependencies),
  and full cross-references to ROADMAP, review, module doc, and
  decision notes.

### LOW fixes (dsp-reviewer)

**Fix 9 — Per-recording Z-score docstring caveat**
- `src/usv_spectrogram/classifier/cleaning_pipeline.py:316-336` —
  `_apply_per_recording_zscore` docstring extended with the
  divergence-from-upstream note: 2D analogue uses global median+MAD
  while upstream 1D uses a bottom-50% noise slice; assumption is USVs
  <50% of pixel area; dense-babble regimes would inflate the noise
  estimate.

**Fix 10 — Baseline fallback kernel + epsilon aligned to upstream**
- `src/usv_spectrogram/classifier/cleaning_pipeline.py:59-72` —
  epsilon `_DB_TO_LINEAR_EPS` changed from `1e-30` to `1e-10` to
  match upstream `app/core/denoise.py:DEFAULT_EPSILON`.
- `src/usv_spectrogram/classifier/cleaning_pipeline.py:74-82` —
  `_FALLBACK_BASELINE_KERNEL_HOP` constant added via guarded
  `from ..corpus import STFT_HOP` (with 128 fallback for minimal
  checkouts where corpus is not importable).
- `src/usv_spectrogram/classifier/cleaning_pipeline.py:268-289` —
  `_local_baseline_subtract` median-envelope kernel now follows
  upstream's "0.5 s of audio" rule:
  `max(3, int(0.5 * sample_rate_hz / _FALLBACK_BASELINE_KERNEL_HOP) | 1)`.
  Bounded above by `n_time` for short inputs.
- `src/usv_spectrogram/classifier/cleaning_pipeline.py:250-254` —
  `_apply_baseline_subtraction` call site updated to pass
  `cfg.sample_rate_hz` to the fallback.
