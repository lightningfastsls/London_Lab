# Handoff: Rerun SIS 17.1 with a Bout Filter

**Date:** 2026-04-19
**From:** Claude Code (session 20260419)
**To:** Claude Code (future chat — fresh context recommended)
**Status:** Ready — two design decisions to make + implementation + rerun
**Prior handoff:** `docs/handoffs/sis-baselines-mi-reconciliation.md` (the registry work that uncovered this gap)

---

## TL;DR

Module 17.1 (`scripts/run_sis_baselines.py`) was approved and implemented with **no bout filter** (explicit in the spec: "bout detection = NONE"). Follow-up reconciliation work (2026-04-18 → 2026-04-19) uncovered that:

1. **Hertz et al. 2020 uses a 160 ms ISI filter** — MI is computed only within silence-bounded sequences. Verified in the paper's Methods: *"we divided the labeled syllables into sequences, based on their ISI (with 160 ms as a threshold). … An ISI of more than 160 ms represented the end of the current sequence and the beginning of a new one."*
2. **Phase A2 uses a 600 ms IOI-derived filter** — bout-aware MI = 0.0921 bits for Scattoni-7 (registered as `canonical_for_downstream: true` in `data/corpus_facts/5970.json`).
3. **SIS 17.1's no-filter MI = 0.0758 bits** — this is NOT apples-to-apples vs Hertz's benchmark lines (0.10 / 0.13 / 0.22 / 0.23 bits at depth 1) that module 17.9 will plot.

**Your job:** apply a bout filter to 17.1's MI computation so `results/sis_baselines/baselines.csv` contains methodology-aligned values. Two decisions to make + a code change.

## Why this isn't trivial

You need to decide **what threshold** to use and **how to share code** with Phase A2 before implementing. Neither is obvious from first principles.

## Decision 1 — Which ISI threshold?

**Check the corpus facts before choosing** — do NOT hardcode from memory. Run:

```python
import json
d = json.load(open('data/corpus_facts/5970.json'))
print(d['timing']['median_ioi_ms'])      # onset-to-onset
print(d['timing']['median_ici_gap_ms'])  # offset-to-onset silent gap
print(d['bout_detection_a2']['threshold_s'])
```

As of 2026-04-19 the actual values are:

| Quantity | Definition | Value |
|---|---|---:|
| `median_ioi_ms` | onset-to-onset | **193 ms** |
| `median_ici_gap_ms` | offset-to-onset silent gap | **87 ms** |
| `bout_detection_a2.threshold_s` | Phase A2's current choice | **0.6 s** (≈ 3× median IOI) |

Re-read these live — they may have been recomputed when 3452/9252 were added to the corpus.

### Candidate thresholds

| Option | Value | Derivation | Pros | Cons |
|---|---:|---|---|---|
| **A** | **0.6 s** (≈ 3× median IOI) | Phase A2 current default | Consistency with existing bout-aware analyses. Uses our data. | IOI-based, not ISI-based → doesn't structurally mirror Hertz. Most permissive of the three. |
| **B** | **0.26 s** (3× median ICI gap) | Hertz's *structure* (silent-gap threshold) with Phase A2's 3× multiplier applied to our data | Structurally matches Hertz (ISI-based). Still data-adapted to our animals. | Differs from Phase A2's current 0.6 s → A2 and 17.1 would diverge unless A2 is also recomputed. |
| **C** | **0.16 s** (literal Hertz) | Hertz 2020 methods section, verbatim | Literal match to Hertz's benchmark methodology. Fully comparable to 0.10/0.13/0.22 reference lines. | Not data-adapted to our animals (wild mice ≠ Hertz's strains). Most restrictive — will exclude the most pairs. |

### My recommendation: **Option A (0.6 s)**, for now

Rationale:
- Our canonical registry already lists 0.6 s as `bout_detection_a2.threshold_s` — changing 17.1 to 0.26 s creates a new methodology divergence between 17.1 and Phase A2.
- The bout-aware MI (0.0921 bits) under 0.6 s is already `canonical_for_downstream: true`. Option A reproduces that exact number and makes 17.1 consistent with the registry.
- The *family-of-methods* argument (silence-gap segmentation → per-sequence MI) is satisfied at any threshold ≥ ~160 ms. The exact threshold matters less than the presence of filtering.

But **don't silently default** — surface the decision to the user. If they prefer Option B (restructure to mirror Hertz more closely) or C (literal Hertz), both are defensible. Option B in particular is worth raising because it fixes a real methodology inconsistency in the corpus (Phase A2 uses IOI-based bout detection, Hertz and most of the field use ISI-based).

If the user picks B, that's a bigger change: Phase A2's `bout_detection_a2` also needs recomputing to match. Flag that as a downstream task, don't do it in the same PR.

## Decision 2 — Factor out shared code or duplicate?

Phase A2 already has bout-filtered MI logic in `scripts/analyze_sequential_structure.py`. SIS 17.1 needs the same. Two paths:

### Option F — Factor out a shared helper (recommended)

Move the bout-filtering MI logic into `usv_language.analysis.sequence_analysis` (or a new submodule) as something like:

```python
def mutual_information_within_bouts(
    labels: np.ndarray,
    ici_gap_s: np.ndarray,      # offset-to-onset, per-pair
    bout_threshold_s: float,
    K: int,
) -> tuple[float, int, int]:
    """Returns (mi_bits, n_within_bout_pairs, n_excluded_pairs)."""
```

Then:
- `scripts/analyze_sequential_structure.py` calls it
- `scripts/run_sis_baselines.py` calls it
- One implementation, one test surface, zero drift risk

Cost: ~2-3 hours. Factor, migrate A2's caller, add tests covering the new helper, update SIS 17.1 to use it, verify Phase A2's output is unchanged.

### Option G — Duplicate the logic

Copy Phase A2's bout-gap filter into `run_sis_baselines.py` with a new `--bout-threshold-s` flag. Faster (~45 min) but introduces drift risk: if bout filtering ever evolves (e.g. add min-bout-length), both places need to update.

**My recommendation: F.** The factoring cost is worth it because bout filtering is a *general analytical primitive* (not specific to SIS), and Phase C+ work on 3452/9252 will want the same helper. The canonical-corpus architecture is already designed to prevent drift — factoring is consistent with that philosophy.

## Implementation plan (assuming A + F)

### Files to modify

1. **`usv_language/analysis/sequence_analysis.py`** (or a new `bout_filter.py` submodule — pick whatever matches existing layout)
   - Add `mutual_information_within_bouts(labels, ici_gap_s, bout_threshold_s, K)` that:
     - Splits pairs into within-bout (ici < threshold) and cross-bout (ici ≥ threshold)
     - Builds the K×K transition count matrix from within-bout pairs only
     - Returns (mi_bits, n_within, n_excluded)
   - Also add `filter_sequence_into_bouts(labels, ici_gap_s, bout_threshold_s) -> list[np.ndarray]` if Phase A2 needs it in that shape

2. **`scripts/analyze_sequential_structure.py`**
   - Replace its inline bout-filter logic with calls to the new helper
   - Verify output `results/sequential_structure/sequential_structure_summary.csv` is byte-identical (or only differs in rounding tolerance of ≤1e-6 bits)

3. **`scripts/run_sis_baselines.py`**
   - Add `--bout-threshold-s FLOAT` argument (default: read from `data/corpus_facts/{dataset}.json:bout_detection_a2.threshold_s`)
   - Add `--classified-csv` expansion: the script needs per-pair ICI gap. Either:
     - Load `results/sequential_structure/ici_gap.npy` directly (simplest)
     - Compute ICI gap inline from `end_time_s`/`begin_time_s` columns (more robust but duplicative)
     - Use whichever approach `analyze_sequential_structure.py` uses
   - Wire the 3 SIS computations (Scattoni-7, DeepSqueak-27, HDBSCAN-3) through the new helper
   - Parameters block (printed to stdout per `feedback_analysis_print_params`): must include the threshold value, its source (corpus vs CLI override), and the n_within / n_excluded counts per labeling

4. **`results/sis_baselines/baselines.csv`** will be regenerated
   - Expected: Scattoni-7 MI = ~0.0921 bits (matches Phase A2, validates the implementation)
   - DeepSqueak-27 and HDBSCAN-3 MI values will also shift — record their new values
   - Add a `parameters.json` sidecar (already exists) that records the bout threshold + n_pairs used

5. **`results/sis_baselines/baselines.png`** will be regenerated
   - The Hertz reference lines (0.10 / 0.13 / 0.22 / 0.23) are now methodologically matched — no caveat needed on the chart

6. **`docs/modules/sis-baselines.md`**
   - Update §Reproducibility Check: the two-row table becomes *one* row (bout-aware is canonical; no-filter is removed from the registry or demoted to a methodology-demo-only footnote)
   - Add a line citing the new shared helper
   - Remove the "rerun with filter OR caveat chart" option from the ROADMAP-gap subsection (it's resolved)

7. **`data/corpus_facts/5970.json`** will be regenerated by `audit_corpus.py`
   - The `scattoni_7_raw_consecutive` entry may be removed entirely, OR kept with `canonical_for_downstream: false` and a new `deprecated: true` flag + reason
   - The `scattoni_7_bout_aware` MI value shouldn't change (still 0.0921), but its `source` may shift from `results/sequential_structure/…` to `results/sis_baselines/baselines.csv` since the new SIS 17.1 also produces it — pick one source and be consistent
   - Round-trip test (`diff data/corpus_facts/5970.json /tmp/regen.json`) must still pass with only timestamp differing

8. **`ROADMAP_SIS_BENCHMARK.md`**
   - Remove the METHODOLOGY GAP FLAGGED line from 17.1's status (it's resolved)
   - Update 17.1's exit criterion: "Scattoni-7 MI ≈ 0.093 bits" is now *actually* the expected number, not aspirational
   - If Decision 2 chose factoring (F), add a note about the new helper location

9. **`tests/test_sis_baselines.py`**
   - New test: `test_bout_filter_matches_phase_a2_scattoni_value` — on the real dataset, with the A2 threshold, Scattoni-7 MI comes out to 0.0921 ± 0.0005 bits
   - New test: `test_bout_filter_with_all_pairs_within_bouts_equals_no_filter` — sanity check that a very large threshold (e.g. 1000 s) reproduces the old no-filter value
   - New test: `test_bout_filter_excluded_count_matches_expected` — with the A2 threshold on 5970, n_within = 6350, n_excluded = 1513 (match Phase A2)
   - Keep the existing 53 tests passing

10. **`tests/test_analyze_sequential_structure.py`** (if it exists — check)
    - Ensure refactor to use the shared helper doesn't regress A2's numbers

### Validation sequence (same as the reconciliation handoff)

1. `py_compile` on every modified file
2. `pytest tests/test_corpus.py tests/test_sis_baselines.py tests/test_analyze_sequential_structure.py -v` — all green
3. Rerun `scripts/analyze_sequential_structure.py --dataset 5970` — verify output CSV is unchanged vs current
4. Rerun `scripts/run_sis_baselines.py` with new bout filter — verify Scattoni-7 MI ≈ 0.0921 bits (matches Phase A2; that's your implementation correctness signal)
5. Rerun `scripts/audit_corpus.py --dataset 5970 --output data/corpus_facts/5970.json`
6. Round-trip check: `python scripts/audit_corpus.py --dataset 5970 --output /tmp/regen.json && diff data/corpus_facts/5970.json /tmp/regen.json` — timestamp-only diff

## Expected delta in committed artifacts

- `results/sis_baselines/baselines.csv` — 3 MI values change (Scattoni-7 ≈ 0.0758 → ~0.0921, DeepSqueak-27 and HDBSCAN-3 also shift)
- `results/sis_baselines/baselines.png` — regenerated with new bar heights; Hertz reference lines now methodology-aligned
- `results/sis_baselines/parameters.json` — gains `bout_threshold_s` field + n_within/n_excluded per labeling
- `data/corpus_facts/5970.json` — `scattoni_7_raw_consecutive` removed or marked deprecated; `scattoni_7_bout_aware` remains canonical (unchanged value)
- `docs/modules/sis-baselines.md` — reproducibility section rewritten; gap flag removed
- `ROADMAP_SIS_BENCHMARK.md` — 17.1 status line cleaned up

## Things NOT to change

- **Don't touch the MI estimator itself.** `usv_language.analysis.sequence_analysis.mutual_information_at_lag` is working correctly. This change is upstream of it (pair filtering), not inside it.
- **Don't re-run 17.4, 17.7, 17.8.** Those modules haven't been built yet. When they are built, they should use the new helper from the start — leave that for their own implementation.
- **Don't change `ExtractionConfig`, corpus.py, or any CNN-related file.** This is a pure analysis-layer change — no Layer 1 physical constants touched.
- **Don't change 3452/9252 entries in `DATASET_REGISTRY`.** They're warns-and-skips — 17.1 hasn't been run on them yet. When it is (Phase B), the bout filter is inherited for free.

## Relevant Constraints (from vault)

Run `/kcheck "sis baselines bout filter"` for a richer pull, but the key items:

1. **Corpus-canary hook will fire** on edits to `data/corpus_facts/5970.json` and `scripts/audit_corpus.py`. Answer (B) for both — you're using/extending existing empirical facts, not declaring new ones.

2. **`feedback_analysis_print_params`** — every analysis run must print its parameters. The rerun must show the threshold, its source (corpus or CLI override), the per-labeling pair counts (n_within, n_excluded), and the row-drop rule. Don't skip this even though it feels redundant.

3. **`feedback_roadmap_not_authoritative`** — `ROADMAP_SIS_BENCHMARK.md` is binding *for this module's scope*. The methodology-gap flag you'll remove after implementation IS part of that scope. Don't treat the ROADMAP text as optional to update.

4. **Module 17.1 tests were written by `test-architect` before implementation** — treat them as spec. If the bout-filter change forces a test expectation to shift, that's a discussion point, not a unilateral change. See CLAUDE.md §Test Protocol.

5. **Large-file protocol** — `scripts/analyze_sequential_structure.py` may or may not be large; check line count. `information_theory.py` and `repertoire_stats.py` are in the large-file list (1,075 and 1,142 lines respectively) — read in chunks if you touch them.

## Heads-up: new hook will fire on this work

The corpus-canary hook (from `.claude/hooks/corpus_canary.py`) will fire on:

- **`data/corpus_facts/5970.json`** — Answer **(B)** when prompted. You're using existing empirical facts (threshold from corpus, ICI gap from existing npy).
- **`scripts/audit_corpus.py`** — Answer **(B)** if you only update wiring; **(C)** if you add a new computed field.
- **`scripts/run_sis_baselines.py`** — Answer **(B)** (reading threshold from corpus) or **(D)** (unrelated to canonical params).

None of these should trigger Layer 1 drift assertions — this is pure Layer 2 work.

## Open questions for the future chat to raise with the user

1. **Decision 1 — threshold choice**: A (0.6 s, status quo), B (0.26 s, restructure), or C (0.16 s, literal Hertz)? My rec: A.
2. **Decision 2 — factor or duplicate**: F (new shared helper) or G (copy into 17.1)? My rec: F.
3. **Should we remove `scattoni_7_raw_consecutive` from the registry** after this work, or keep it as a documented-deprecated entry? My rec: keep with `deprecated: true` + reason — preserves the trail of why the flip happened.
4. **Should `ici_gap.npy`** be moved to a more general location (e.g. `data/corpus_facts/5970/ici_gap.npy`) since both 17.1 and Phase A2 consume it? Orthogonal to this task; flag only.

## Source artifacts (already read — paths for reference)

- `scripts/analyze_sequential_structure.py` — Phase A2 bout-filter logic to factor out
- `scripts/run_sis_baselines.py:264-265` — current "bout detection = NONE" declaration to replace
- `src/usv_spectrogram/classification/sis_baselines.py` — 17.1 module itself; probably doesn't need changes if filtering happens upstream
- `results/sequential_structure/ici_gap.npy` — per-pair ICI gap array, input for bout filtering
- `results/sequential_structure/sequential_structure_summary.csv` — Phase A2 reference: mi_lag1_bits=0.0921, n_within_bout_pairs=6350, n_cross_bout_gaps_excluded=1513
- `results/sis_baselines/baselines.csv` — current 17.1 output (to be regenerated)
- `data/corpus_facts/5970.json:sequential_structure_mi` — registry block to update
- `docs/modules/sis-baselines.md` — module doc, §Reproducibility Check
- `docs/handoffs/sis-baselines-mi-reconciliation.md` — prior handoff, covers the registry work
- Hertz 2020 (open-access PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC7320152/ — Methods section quote: *"we divided the labeled syllables into sequences, based on their ISI (with 160 ms as a threshold)"*

## Background — why this handoff exists at all

You could ask: why wasn't 17.1 implemented with a bout filter from day one? Answer: the ROADMAP spec (line 20-28) never stated a filter should be applied. The implementer (correctly) treated the spec as authoritative and produced a no-filter implementation. The exit criterion said "Scattoni-7 MI ≈ 0.093 bits" but didn't say that number was bout-derived, so the 0.0758 result was accepted with a tentative note.

The 2026-04-18 reconciliation work promoted both MI values into the canonical registry, which made the discrepancy visible side-by-side. The 2026-04-19 audit of Hertz 2020's Methods confirmed that bout filtering is integral to the benchmark methodology — not an optional variant. That's the finding this handoff operationalizes.
