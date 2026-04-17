# Supplement: Additional Parameters for Corpus Unification

**Date:** 2026-04-17
**Parent handoff:** `docs/handoffs/corpus-constants-unification-2026-04-17.md`
**Purpose:** Extend the corpus-refactor scope with three additional categories discovered after the parent handoff was dispatched. Add these as **Phase 2** once Phase 1 lands.

---

## Relationship to the parent handoff

The parent handoff (Phase 1) covered sample rate + USV frequency band + STFT params. It explicitly locks `corpus.py` values to the CNN's training grid (20-120 kHz). Phase 2 (this doc) adds three more categories of shared constants that don't affect the CNN but ARE scattered across many files.

**Do NOT start Phase 2 until Phase 1 is merged** — the file `src/usv_spectrogram/corpus.py` must already exist. Phase 2 is purely additive.

---

## Phase 2 additions

### 2A. Scattoni-7 syllable taxonomy (high value)

**Problem:** The 7-type list is hardcoded in 11+ files with subtle risk of order/case/spelling drift.

Affected files (verified 2026-04-17):
- `scripts/analyze_temporal_dynamics.py`
- `scripts/analyze_sequential_structure.py`
- `scripts/analyze_acoustic_features.py`
- `scripts/classify_traditional_taxonomy.py` (the classifier — source of truth for label NAMES)
- `scripts/run_sis_baselines.py`
- `src/usv_spectrogram/classification/sis_baselines.py`
- `tests/test_traditional_taxonomy.py`
- `tests/test_traditional_taxonomy_adversarial.py`
- `tests/test_analyze_detection_confidence.py`
- `tests/test_analyze_acoustic_features.py`
- `tests/test_sis_baselines.py`

**Fix:** create `src/usv_spectrogram/classification/taxonomy.py`:

```python
"""Scattoni-7 syllable taxonomy — single source of truth.

Seven call types from Scattoni et al. 2008 adapted for CNN-based
classification. Order is STABLE and must not change — downstream code
uses positional integer codes (type_code = list.index(name)).
"""

from __future__ import annotations
from typing import Final

# Canonical ordering for Scattoni-7 syllable types.
# DO NOT REORDER — breaks every saved CSV with integer type_code columns.
SCATTONI_SYLLABLE_TYPES: Final[tuple[str, ...]] = (
    "Flat",
    "Down",
    "Chevron",
    "Short",
    "Complex",
    "Frequency_Jump",
    "Up",
)

SCATTONI_TYPE_TO_CODE: Final[dict[str, int]] = {
    t: i for i, t in enumerate(SCATTONI_SYLLABLE_TYPES)
}
SCATTONI_CODE_TO_TYPE: Final[dict[int, str]] = {
    i: t for t, i in SCATTONI_TYPE_TO_CODE.items()
}

# Display palette — shared across all Phase-A analysis scripts.
# Tol-friendly colors; do not change casually (every plot in results/
# depends on these).
SCATTONI_TYPE_COLORS: Final[dict[str, str]] = {
    "Flat": "#4477AA",
    "Down": "#EE6677",
    "Chevron": "#228833",
    "Short": "#CCBB44",
    "Complex": "#AA3377",
    "Frequency_Jump": "#66CCEE",
    "Up": "#BBBBBB",
}
```

**Before making the refactor**: verify the ordering in every file MATCHES. If any file uses a different order, downstream integer codes will silently shift. Compare against `classify_traditional_taxonomy.py` as the source of truth. If a file diverges, fix the file — don't adjust the canonical list.

**Tests**: add `tests/test_taxonomy.py` that asserts `len(SCATTONI_SYLLABLE_TYPES) == 7`, codes are 0..6, and `SCATTONI_CODE_TO_TYPE[SCATTONI_TYPE_TO_CODE[t]] == t` for every t.

---

### 2B. Reproducibility seed (low risk, high coverage)

**Problem:** `seed = 42` / `random_state = 42` / `np.random.seed(42)` / `torch.manual_seed(42)` appears in 15+ files. Changing one-off does nothing; there's no way to sweep the seed without touching every file.

Affected files (verified 2026-04-17):
- `src/usv_spectrogram/models/config.py` (TrainingConfig.seed = 42)
- `scripts/train_cnn.py`
- `scripts/analyze_acoustic_features.py`
- `usv_language/tests/test_bout_dataset.py`
- `usv_language/tests/test_spectrogram_transformer.py`
- `usv_language/tests/test_train_pipeline.py`
- `usv_language/tests/test_dataset.py`
- `usv_language/tests/test_hidden_state_vqvae.py`
- `tests/test_sis_baselines.py`, `tests/test_sim_optimizer.py`, `tests/test_cluster_sweep.py`, `tests/test_amvoc_autoencoder.py`
- `tests/test_recluster_umap_hdbscan.py`, `tests/test_recluster_umap_hdbscan_adversarial.py`
- `tests/test_traditional_taxonomy_adversarial.py`
- `tests/test_dataset_assembler.py`

**Fix:** add to `src/usv_spectrogram/corpus.py`:

```python
# Reproducibility seed — single source of truth for all deterministic
# operations (numpy, torch, scikit-learn, UMAP, HDBSCAN).
# Chosen 42 (Hitchhiker's Guide convention, no special numerical property).
# If you want to test seed-sensitivity of results, override locally — but
# NEVER modify this value; it invalidates every cached experiment.
REPRODUCIBILITY_SEED: Final[int] = 42
```

Migrate all 15+ files to import from corpus. This is a **mechanical refactor** — every site already uses 42. A successful migration produces zero numerical change in any test.

**Note on `TrainingConfig.seed`**: do NOT remove this field. Keep the dataclass API stable; just default to `corpus.REPRODUCIBILITY_SEED`. Old code passing `seed=43` still works for experiments.

---

### 2C. USV duration limits (physical claims, move to corpus)

**Problem:** `min_duration_ms = 10.0` and `max_duration_ms = 500.0` are treated as detection-config parameters, but they're actually **physical claims** about the USV range ("any detection < 10ms or > 500ms is not a real USV"). They're duplicated in 4 files:
- `src/usv_spectrogram/detection/config.py` (DetectionConfig defaults)
- `src/usv_spectrogram/detection/energy_detector.py` (uses them via config)
- `src/usv_spectrogram/app/main_window.py:797-799` (app hardcodes 10/500 literals — should use config)
- `src/usv_spectrogram/app/core/detection_logic.py`

**Semantic clarification — these are SINGLE-CALL bounds, not bout/gap thresholds.**
- `max_duration_ms = 500` filters out segments longer than 500ms as "non-USV vocalizations" (confirmed from `energy_detector.py:573-576`).
- The 300ms comment in DetectionConfig refers to the typical upper bound of real mouse USVs; 500ms is a 200ms safety margin.
- **Do NOT** confuse with "inter-bout gap" (250-500ms literature range) — that's a completely separate concept the code doesn't have a name for yet.

**Fix:** add to `src/usv_spectrogram/corpus.py`:

```python
# Single-call duration bounds. Mouse USVs are empirically 10-300 ms; the
# max is set to 500 ms as a safety margin to avoid rejecting unusually-long
# real calls. Detections outside these bounds are treated as non-USV
# (noise, pup calls, or ambient vocalizations).
#
# NOT a bout/gap threshold — see bout detection in scripts/analyze_*.py
# for inter-call timing heuristics.
USV_MIN_DURATION_MS: Final[float] = 10.0
USV_MAX_DURATION_MS: Final[float] = 500.0
```

Update `DetectionConfig` defaults to import from corpus. Fix `main_window.py:797-799` to read from the DetectionConfig instance instead of hardcoded literals.

**CNN-freeze caveat:** duration bounds affect which detection candidates reach the CNN. Changing them (e.g., 500→400) would reduce the CNN's input distribution slightly. Not strictly a CNN-training change (the CNN scores candidates it gets; it doesn't know about duration filtering), but conservative practice: **don't change the values** in this refactor, only centralize them.

---

## Phase 2 completion sequence

1. Start only AFTER Phase 1 (parent handoff) is merged on main.
2. **2A first** — taxonomy centralization, new file `classification/taxonomy.py`. Write test `tests/test_taxonomy.py`. Commit.
3. **2C second** — duration-limit migration into `corpus.py`. Update DetectionConfig + main_window.py. Commit.
4. **2B last** — seed centralization. Mechanical replace in ~15 files. Commit.
5. Run full test suite after each commit. Expect zero numerical change.
6. Spawn `master-reviewer`.

## Don't do these

- **Don't reorder SCATTONI_SYLLABLE_TYPES** — every saved CSV has integer codes tied to this order.
- **Don't change any duration value** — centralize location, not meaning.
- **Don't change the seed value** — centralize location, not value. Swapping seeds invalidates every cached experiment.
- **Don't centralize clustering hyperparameters** (UMAP n_neighbors, HDBSCAN min_cluster_size). Those are experiment choices per-run, captured by the `parameters.json` sidecar pattern already landed 2026-04-17.
- **Don't centralize DeepSqueak match tolerance (75ms)** — it's in one file (`deepsqueak_import.py`), no benefit to moving, but add a comment citing the methodology reference.

## Context for the fresh session

This supplement was written in the same session as the parent handoff. The user ran a conversation-context check (`/context`) and asked "anything else along these lines of unification of parameters you think we missed?" The audit surfaced these three categories. The user had already dispatched the parent handoff to a different chat; rather than interrupting, they preferred a separate supplementary doc that can be given to a followup chat.

The parent handoff's CNN-freeze rationale and empirical-data registry (Layer 2) still hold. This supplement only adds the taxonomy/seed/duration trio — it does not modify the three-layer architecture.

## Expected effort

~90 min — all three are low-risk mechanical refactors. No CNN regression risk (values unchanged), no tests should break (same values flowing through same code).
